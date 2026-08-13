# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeepSeek-V4 fp8 2-buffer KV cache (ATOM op4/op5 layout).

Port of ATOM's V4 fp8 KV-cache design (``atom/plugin/vllm/deepseek_v4_bridge.py``
and ``atom/model_ops/v4_kernels/v4_quant.py``): per layer a flat uint8 byte pool
holding, in region-contiguous order, ``swa_nope | main_nope | swa_rope |
main_rope``. The NoPE plane rows are 512B (448 fp8 NoPE + 14 duplicated e8m0
scale bytes + 50 pad); the parallel RoPE plane rows are 128B (64 bf16 -- RoPE
is never quantized). Attention runs on the aiter op4 (prefill) / op5 (decode)
kernels, which consume each plane as a dense contiguous tensor, so the
region-contiguous layout is load-bearing.

This module is pure math / spec / view slicing. The attention backend and
metadata builder classes live in :mod:`vllm.models.deepseek_v4.amd.rocm` to
avoid an import cycle (they subclass the ROCm sparse backend defined there).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import MLAAttentionSpec, get_kv_quant_mode

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# ---- Layout constants (ATOM v4_quant.py) ----
V4_DIM_NOPE = 448
V4_DIM_ROPE = 64
V4_NUM_TILES = 7  # V4_DIM_NOPE // 64
V4_DIM_SCALE_DUP = 14  # V4_NUM_TILES * 2 duplicated e8m0 scale bytes
V4_NOPE_ROW_BYTES = 512  # V4_DIM_NOPE + V4_DIM_SCALE_DUP + 50 pad
V4_ROPE_ROW_BYTES = 128  # V4_DIM_ROPE * 2 (bf16)
V4_ENTRY_BYTES = V4_NOPE_ROW_BYTES + V4_ROPE_ROW_BYTES

# aiter's V4 native 2buff fp8 prefill (op4) / decode (op5) kernels exist only
# on gfx950 / gfx1250 (mirrors ATOM's guard).
_V4_FP8_SUPPORTED_GFX = ("gfx950", "gfx1250")

_ATOM2BUFF_KERNELS_OK: bool | None = None


def _atom2buff_kernels_available() -> bool:
    """Env + arch + aiter-import check for the op4/op5 kernels.

    Cached: environment and installed packages do not change mid-process.
    """
    global _ATOM2BUFF_KERNELS_OK
    if _ATOM2BUFF_KERNELS_OK is not None:
        return _ATOM2BUFF_KERNELS_OK

    ok = envs.VLLM_ROCM_USE_AITER_DSV4_FP8
    if not ok:
        _ATOM2BUFF_KERNELS_OK = False
        return False

    try:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    except Exception:
        gfx = None
    if gfx not in _V4_FP8_SUPPORTED_GFX:
        logger.warning_once(
            "VLLM_ROCM_USE_AITER_DSV4_FP8: aiter op4/op5 fp8 kernels are only "
            "supported on %s (got gfx=%r); falling back to the Triton path.",
            "/".join(_V4_FP8_SUPPORTED_GFX),
            gfx,
        )
        _ATOM2BUFF_KERNELS_OK = False
        return False

    try:
        from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus  # noqa: F401
        import aiter.mla  # noqa: F401
        from aiter.ops.fused_qk_norm_rope_cache_quant import (  # noqa: F401
            fused_qk_norm_rope_group_quant,
        )
    except Exception as e:
        logger.warning_once(
            "VLLM_ROCM_USE_AITER_DSV4_FP8: aiter op4/op5 kernels unavailable "
            "(%s); falling back to the Triton path.",
            e,
        )
        _ATOM2BUFF_KERNELS_OK = False
        return False

    _ATOM2BUFF_KERNELS_OK = True
    return True


def atom2buff_available(kv_cache_dtype: str | None) -> bool:
    """Single authority for the 2-buffer fp8 decision.

    True iff the resolved cache dtype is the fp8_ds_mla spelling AND the
    environment/arch/aiter kernel check passes. The kv_cache_dtype check is
    intentionally uncached so tests can flip it freely.
    """
    if kv_cache_dtype != "fp8_ds_mla":
        return False
    return _atom2buff_kernels_available()


def atom2buff_reject_prefix_caching(enable_prefix_caching: bool) -> None:
    """Fail fast when prefix caching and the 2buff path are both enabled.

    The 2buff fp8 pool uses ATOM ring addressing instead of vLLM
    block-managed SWA, so prefix reuse would serve stale ring rows.
    """
    if not enable_prefix_caching:
        return
    raise ValueError(
        "VLLM_ROCM_USE_AITER_DSV4_FP8 does not support prefix caching: the "
        "2-buffer fp8 pool uses ATOM ring addressing instead of vLLM "
        "block-managed SWA. Start with --no-enable-prefix-caching or unset "
        "the env var."
    )


def atom2buff_ring_rows(
    max_num_seqs: int, window_size: int, spec_steps: int = 0
) -> int:
    """SWA ring rows per layer.

    One contiguous window's worth of slots per concurrent sequence, widened by
    speculative steps so draft tokens can also write their KV (ATOM
    ``_v4_win_with_spec``).
    """
    return max_num_seqs * (window_size + spec_steps)


def v4_atom2buff_ring_rows_from_config(vllm_config: VllmConfig) -> int:
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    window_size = int(vllm_config.model_config.hf_config.sliding_window)
    spec_config = vllm_config.speculative_config
    spec_steps = (
        int(getattr(spec_config, "num_speculative_tokens", 0) or 0)
        if spec_config is not None
        else 0
    )
    return atom2buff_ring_rows(max_num_seqs, window_size, spec_steps)


def atom2buff_head_size(block_size: int, min_blocks: int, ring_rows: int) -> int:
    """Per-token byte size of the flat pool (the uint8 ``head_size``).

    ``V4_ENTRY_BYTES`` per KV entry plus the SWA ring bytes amortized into
    every token slot (ATOM ``_proxy_page_bytes``): with ``num_blocks >=
    min_blocks`` the pool holds the full ring plus all paged KV rows.
    """
    ring_bytes = ring_rows * V4_ENTRY_BYTES
    ring_share = -(-ring_bytes // (min_blocks * block_size))
    return V4_ENTRY_BYTES + ring_share


def v4_atom2buff_head_size_from_config(vllm_config: VllmConfig) -> int:
    cache_config = vllm_config.cache_config
    block_size = cache_config.block_size
    max_model_len = vllm_config.model_config.max_model_len
    min_blocks = max(1, (max_model_len + block_size - 1) // block_size)
    ring_rows = v4_atom2buff_ring_rows_from_config(vllm_config)
    return atom2buff_head_size(block_size, min_blocks, ring_rows)


@dataclass(frozen=True, kw_only=True)
class V4AtomFp82BuffSpec(MLAAttentionSpec):
    """Flat byte-pool KV cache spec for the ATOM V4 fp8 2-buffer layout.

    The pool is a single uint8 tensor of shape
    ``(num_blocks, block_size, head_size)`` where ``head_size`` already
    includes the amortized SWA ring bytes (see
    :func:`v4_atom2buff_head_size_from_config`). The layer slices dense
    per-plane views (:func:`slice_atom2buff_pool_views`); block ``b`` maps to
    ``rows_per_block = block_size // compress_ratio`` rows in the main region,
    matching the existing fp8_ds_mla block geometry.
    """

    @property
    def real_page_size_bytes(self) -> int:
        # One flat byte row per token: head_size bytes (NoPE + RoPE + ring
        # share) for each of the block_size tokens in a page. The parent's
        # formula (storage_block_size-based, 584B special case) does not
        # apply to the byte-pool layout.
        return self.block_size * self.num_kv_heads * self.head_size


def v4_atom2buff_spec(
    vllm_config: VllmConfig, compress_ratio: int
) -> V4AtomFp82BuffSpec:
    block_size = vllm_config.cache_config.block_size
    head_size = v4_atom2buff_head_size_from_config(vllm_config)
    return V4AtomFp82BuffSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
        kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
    )


@dataclass
class Atom2BuffPoolViews:
    """Dense per-plane views over one layer's flat 2-buffer pool."""

    unified_nope: torch.Tensor  # [ring_rows + main_rows, 512] fp8
    unified_rope: torch.Tensor  # [ring_rows + main_rows, 64] bf16
    ring_rows: int
    rows_per_block: int

    @property
    def ring_nope(self) -> torch.Tensor:
        return self.unified_nope[: self.ring_rows]

    @property
    def main_nope(self) -> torch.Tensor:
        return self.unified_nope[self.ring_rows :]

    @property
    def ring_rope(self) -> torch.Tensor:
        return self.unified_rope[: self.ring_rows]

    @property
    def main_rope(self) -> torch.Tensor:
        return self.unified_rope[self.ring_rows :]


def slice_atom2buff_pool_views(
    pool: torch.Tensor, *, ring_rows: int, rows_per_block: int
) -> Atom2BuffPoolViews:
    """Carve dense per-plane views from one layer's flat uint8 pool.

    Region-contiguous layout (mirrors ATOM's bridge slicing minus the indexer
    region, which stays on the fork's existing indexer cache):
    ``swa_nope | main_nope | swa_rope | main_rope``. Every plane view is a
    dense contiguous tensor -- the aiter op4/op5 asm kernels index rows by
    stride and assume no interleaving.

    ``pool`` is the runner-bound cache tensor of shape
    ``(num_blocks, block_size, head_size)`` (uint8).
    """
    if pool.dtype != torch.uint8:
        raise ValueError(
            f"ATOM 2-buffer pool must be uint8, got {pool.dtype}; "
            "is --kv-cache-dtype fp8 in effect?"
        )
    if not pool.is_contiguous():
        pool = pool.contiguous()
    flat = pool.reshape(-1)
    main_rows = pool.shape[0] * rows_per_block
    total_rows = ring_rows + main_rows
    nope_bytes = total_rows * V4_NOPE_ROW_BYTES
    rope_bytes = total_rows * V4_ROPE_ROW_BYTES
    if flat.numel() < nope_bytes + rope_bytes:
        raise ValueError(
            f"ATOM 2-buffer pool too small: need {nope_bytes + rope_bytes} "
            f"bytes for {ring_rows} ring + {main_rows} main rows, pool has "
            f"{flat.numel()}."
        )

    unified_nope = (
        flat[:nope_bytes].view(torch.float8_e4m3fnuz).view(total_rows, V4_NOPE_ROW_BYTES)
    )
    unified_rope = (
        flat[nope_bytes : nope_bytes + rope_bytes]
        .view(torch.bfloat16)
        .view(total_rows, V4_DIM_ROPE)
    )
    return Atom2BuffPoolViews(
        unified_nope=unified_nope,
        unified_rope=unified_rope,
        ring_rows=ring_rows,
        rows_per_block=rows_per_block,
    )


# =============================================================================
# Kernel layer (ports of ATOM's op4/op5 call sites).
# =============================================================================

_MAX_KV_SPLITS = 64


def _cu_count() -> int:
    try:
        from aiter.ops.triton.utils.device_info import get_num_sms

        return int(get_num_sms())
    except Exception:
        return 128


def kv_splits_heuristic(
    T: int,
    H: int,
    block_h: int | None = None,
    num_cu: int | None = None,
    target_wg_per_cu: float = 2.0,
    max_kv_splits: int = _MAX_KV_SPLITS,
) -> int:
    """Split-K factor for the op5 decode kernel (ATOM ``_kv_splits_heuristic``).

    CUDAGraph-safe: depends only on capture-time scalars (T, H, block_h), never
    on runtime tensor values.
    """
    if block_h is None:
        block_h = triton.next_power_of_2(min(H, 64))
    block_h = max(block_h, 16)  # AMD MFMA min tile
    if num_cu is None:
        num_cu = _cu_count()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    head_blocks = max(1, (H + block_h - 1) // block_h)
    base_ctas = max(1, T * head_blocks)
    if base_ctas >= target_wg:
        return 1
    splits_to_fill = max(1, target_wg // base_ctas)
    prev_pow2 = 1 << (min(splits_to_fill, max_kv_splits).bit_length() - 1)
    return prev_pow2


def rocm_fp8_2buff_qk_norm_rope_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    n_local_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Q/K RMSNorm + GPT-J RoPE + e8m0 group-quant into the 2buff layout.

    Port of ATOM ``qk_norm_rope_maybe_quant_fp8_2buff`` (qk_norm_rope_maybe_quant.py):
    one aiter launch producing ``q_packed`` [T, H, 512] fp8, ``q_rope``
    [T, H, 64] bf16, ``k_packed`` [T, 1, 512] fp8, ``k_rope`` [T, 1, 64] bf16
    consumed directly by op4/op5. ``q_weight=None`` = V4 weightless Q norm;
    ``is_neox=False`` = GPT-J adjacent-pair RoPE.

    ``cos_sin_cache`` is the fork's rotary cache [max_pos, 2*rd] (first rd
    cos, last rd sin, per-dim). aiter expects per-PAIR tables [max_pos, rd//2];
    the even-index entries of the per-dim halves are exactly that.
    """
    from aiter import dtypes
    from aiter.ops.fused_qk_norm_rope_cache_quant import (
        fused_qk_norm_rope_group_quant,
    )

    rd = V4_DIM_ROPE
    assert cos_sin_cache.shape[-1] == 2 * rd, (
        f"expected cos_sin_cache [max_pos, {2 * rd}], got {tuple(cos_sin_cache.shape)}"
    )
    cache = cos_sin_cache.squeeze(-2).squeeze(-2)  # tolerate 4D [., 2rd, 1, 1]
    cos_2d = cache[:, 0 : rd : 2]  # [max_pos, rd//2] per-pair cos
    sin_2d = cache[:, rd : 2 * rd : 2]  # [max_pos, rd//2] per-pair sin
    num_tokens = q.shape[0]
    q_packed, q_rope, k_packed, k_rope = fused_qk_norm_rope_group_quant(
        q.view(num_tokens, n_local_heads, head_dim),
        kv,
        kv_weight,
        positions,
        cos_2d,
        sin_2d,
        eps,
        is_neox=False,
        q_out_dtype=dtypes.fp8,
        q_weight=None,
        quant_group_size=64,
        scale_dtype="e8m0",
    )
    return q_packed, q_rope, k_packed, k_rope


def rocm_fp8_2buff_prefill(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    unified_nope: torch.Tensor,
    unified_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """V4 sparse prefill via aiter op4 (ATOM ``paged_prefill.py`` fp8 branch).

    Prefix (ring + compressed pool rows) is read directly from the fp8 NoPE /
    bf16 RoPE planes -- no dequant, no torch quant; the in-chunk extend K is
    the freshly quantized ``k_packed`` / ``k_rope``. Returns [T, H, head_dim]
    bf16.
    """
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    return pa_sparse_prefill_fp8_opus(
        q_packed,
        q_rope,
        unified_nope,
        unified_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        k_packed.view(k_packed.shape[0], -1),
        k_rope.view(k_rope.shape[0], -1),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
    )  # [T, H, head_dim] bf16


def rocm_fp8_2buff_decode(
    q_packed: torch.Tensor,
    q_rope: torch.Tensor,
    unified_nope: torch.Tensor,
    unified_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    qo_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    num_kv_splits: int | None = None,
    *,
    q_packed_pad: torch.Tensor | None = None,
    q_rope_pad: torch.Tensor | None = None,
    sink_pad: torch.Tensor | None = None,
    out_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """V4 sparse decode via aiter op5 (ATOM ``_sparse_attn_v4_paged_decode_asm``).

    GQA head-count padding: the asm kernel only ships gqa in {16, 32, 64, 128};
    a TP sharding with fewer local heads (TP8 -> 8) pads to 16 and slices the
    discardable padded-head output afterwards. Preallocated pad buffers make
    the path allocation-free under cudagraph capture; ``F.pad`` is the eager
    fallback. Trims the per-seq index tensors to the real query count N so an
    eager forward smaller than the staged T_pad never drives an OOB write.
    """
    import aiter.mla

    N, H, _ = q_packed.shape
    H_real = H
    _GQA_MIN = 16
    if H < _GQA_MIN:
        pad_h = _GQA_MIN - H
        if q_packed_pad is not None:
            q_packed_pad[:N].zero_()
            q_packed_pad[:N, :H].copy_(q_packed)
            q_packed = q_packed_pad[:N]
        else:
            q_packed = torch.nn.functional.pad(q_packed, (0, 0, 0, pad_h))
        if q_rope_pad is not None:
            q_rope_pad[:N].zero_()
            q_rope_pad[:N, :H].copy_(q_rope)
            q_rope = q_rope_pad[:N]
        else:
            q_rope = torch.nn.functional.pad(q_rope, (0, 0, 0, pad_h))
        if sink_pad is not None:
            sink_pad.zero_()
            sink_pad[:H].copy_(attn_sink)
            attn_sink = sink_pad
        else:
            attn_sink = torch.nn.functional.pad(attn_sink, (0, pad_h))
        H = _GQA_MIN

    nhead_kv = 1
    page_size = 1
    kv_packed = unified_nope.view(-1, page_size, nhead_kv, V4_NOPE_ROW_BYTES)
    kv_rope = unified_rope.view(-1, page_size, nhead_kv, V4_DIM_ROPE)

    # Trim to the actual query count: the staged (padded) tail carries 0-length
    # slots and must not drive the kernel's seq count past the real N.
    qo_indptr = qo_indptr[: N + 1]
    kv_indptr = kv_indptr[: N + 1]
    kv_page_indices = kv_indices.contiguous()

    if out_pad is not None:
        output = out_pad[:N]
    else:
        output = torch.empty(
            (N, H, V4_DIM_NOPE + V4_DIM_ROPE), dtype=torch.bfloat16,
            device=q_packed.device,
        )

    aiter.mla.mla_decode_fwd_v4_nm(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        output,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        1,  # max_seqlen_q
        sink=attn_sink,
        sm_scale=softmax_scale,
        num_kv_splits=num_kv_splits,
    )
    # Drop padded heads. The slice is a non-contiguous view, so .contiguous()
    # gives downstream a dense tensor; no-op when H was unpadded.
    return output[:, :H_real].contiguous() if H_real != H else output


@triton.jit
def _swa_ring_scatter_2buff_kernel(
    nope_src,  # [T, 512] raw bytes (uint8 view of the fp8 plane)
    rope_src,  # [T, 128] raw bytes (uint8 view of the bf16 plane)
    pos_ptr,  # [T] int64
    cu_ptr,  # [bs+1] int32
    slot_ptr,  # [bs] int32
    nope_dst,  # [ring_rows, 512] raw bytes
    rope_dst,  # [ring_rows, 128] raw bytes
    RING_SLOTS: tl.constexpr,  # window_size + spec_steps
    WRITE_PER_BATCH: tl.constexpr,
    NOPE_BYTES: tl.constexpr,
    ROPE_BYTES: tl.constexpr,
):
    """Scatter the LAST ``WRITE_PER_BATCH`` tokens of every sequence into that
    request's SWA ring rows (ATOM ``swa_write`` semantics, per-layer ring).

    Ring row = ``slot * RING_SLOTS + (pos % RING_SLOTS)``. Pure byte copy --
    the K is already in the 2buff layout (fp8 NoPE + bf16 RoPE), so no
    quantization happens here.
    """
    b = tl.program_id(0)
    w = tl.program_id(1)
    start = tl.load(cu_ptr + b)
    end = tl.load(cu_ptr + b + 1)
    tok_n = end - start
    if w >= tok_n:
        return
    t = end - 1 - w
    pos = tl.load(pos_ptr + t)
    slot = tl.load(slot_ptr + b)
    row = slot * RING_SLOTS + (pos % RING_SLOTS)

    offs = tl.arange(0, NOPE_BYTES)
    tl.store(
        nope_dst + row * NOPE_BYTES + offs,
        tl.load(nope_src + t * NOPE_BYTES + offs),
    )
    offs_r = tl.arange(0, ROPE_BYTES)
    tl.store(
        rope_dst + row * ROPE_BYTES + offs_r,
        tl.load(rope_src + t * ROPE_BYTES + offs_r),
    )


def swa_ring_scatter_2buff(
    k_packed: torch.Tensor,
    k_rope: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    state_slot_per_seq: torch.Tensor,
    pool_nope: torch.Tensor,
    pool_rope: torch.Tensor,
    ring_slots: int,
    write_per_batch: int,
) -> None:
    """SWA ring write of the per-forward tail into both planes.

    ``k_packed`` [T, 512] fp8 / ``k_rope`` [T, 64] bf16 are the freshly
    quantized extend K (2buff layout); ``pool_nope`` / ``pool_rope`` are this
    layer's dense plane views. ``write_per_batch`` = min(max_q_len,
    ring_slots): prefill writes the chunk's window tail, decode writes the one
    new token.
    """
    if k_packed.shape[0] == 0:
        return
    bs = int(state_slot_per_seq.shape[0])
    nope_src = k_packed.contiguous().view(torch.uint8).view(-1, V4_NOPE_ROW_BYTES)
    rope_src = k_rope.contiguous().view(torch.uint8).view(-1, V4_ROPE_ROW_BYTES)
    nope_dst = pool_nope.view(torch.uint8)
    rope_dst = pool_rope.view(torch.uint8)
    _swa_ring_scatter_2buff_kernel[(bs, write_per_batch)](
        nope_src,
        rope_src,
        positions,
        cu_seqlens_q.to(torch.int32),
        state_slot_per_seq.to(torch.int32),
        nope_dst,
        rope_dst,
        RING_SLOTS=ring_slots,
        WRITE_PER_BATCH=write_per_batch,
        NOPE_BYTES=V4_NOPE_ROW_BYTES,
        ROPE_BYTES=V4_ROPE_ROW_BYTES,
    )


# =============================================================================
# Compressor 2buff store (adaptation of the fork's fused compress kernel).
# Dispatched from an explicit hook in compressor.py when the layer's kv_cache
# is a 2buff pool.
# =============================================================================


@triton.jit
def _fused_kv_compress_norm_rope_insert_2buff(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output (2buff planes) ──
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    nope_plane_ptr,  # [ring+main rows, 512] raw bytes
    rope_plane_ptr,  # [ring+main rows, 64] bf16
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,  # 448.0
    QUANT_BLOCK: tl.constexpr,  # 64 for DeepseekV4
    ROWS_PER_BLOCK: tl.constexpr,  # kv_cache_block_size // COMPRESS_RATIO
    NOPE_ROW_BYTES: tl.constexpr,  # 512
    DIM_SCALE_DUP: tl.constexpr,  # 14 (7 tiles x 2)
    PACK_OFF_SCALE: tl.constexpr,  # 448
):
    """Fused compress → RMSNorm → FP8 quant (nope) → RoPE → 2buff store.

    Identical to the fork's ``_fused_kv_compress_norm_rope_insert_sparse_attn``
    except for the final store: the 448 fp8 NoPE values, the 7 e8m0 scale bytes
    (duplicated x2, inline at offset 448), a 50-byte zero pad, and the 64 bf16
    RoPE values go to the layer's dense NoPE / RoPE planes at
    ``row = block_id * ROWS_PER_BLOCK + pos_in_block // COMPRESS_RATIO``.
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── 2buff row addressing ──────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    row = kv_block_idx * ROWS_PER_BLOCK + kv_pos_in_block // COMPRESS_RATIO
    nope_row_ptr = nope_plane_ptr + row.to(tl.int64) * NOPE_ROW_BYTES
    rope_row_ptr = (rope_plane_ptr + row.to(tl.int64) * ROPE_HEAD_DIM).to(
        tl.pointer_type(tl.bfloat16)
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(nope_row_ptr + block, x_uint8_flat, mask=nope_mask)

    # e8m0 scales duplicated inline at offset 448 (14 bytes), 50-byte zero pad.
    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    encoded_u8 = encoded.to(tl.uint8)
    dup_offs = PACK_OFF_SCALE + scale_idx * 2
    tl.store(nope_row_ptr + dup_offs, encoded_u8, mask=scale_idx < N_NOPE_BLOCKS)
    tl.store(
        nope_row_ptr + dup_offs + 1, encoded_u8, mask=scale_idx < N_NOPE_BLOCKS
    )
    pad_offs = tl.arange(0, 64)
    tl.store(
        nope_row_ptr + PACK_OFF_SCALE + DIM_SCALE_DUP + pad_offs,
        tl.zeros((), dtype=tl.uint8),
        mask=pad_offs < (NOPE_ROW_BYTES - PACK_OFF_SCALE - DIM_SCALE_DUP),
    )

    # Register-based GPT-J RoPE in fp32.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32

    # Store the rotated rope portion as bf16 into the RoPE plane row.
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(rope_row_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


def compress_norm_rope_store_2buff(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    **kwargs: Any,
) -> None:
    """2buff-layout launcher mirroring ``compress_norm_rope_store_triton``.

    The 2buff planes ride on the per-step metadata (attached by the attention
    layer, whose ``get_kv_cache_spec`` sized the pool); the row base is the
    layer's ``main_nope`` / ``main_rope`` views, so no ``ring_rows`` offset is
    needed inside the kernel.
    """
    views = getattr(k_cache_metadata, "atom2buff_views", None)
    if views is None:
        raise ValueError(
            "2buff compressor store requires atom2buff_views on the metadata"
        )
    kv_slot_mapping = k_cache_metadata.slot_mapping
    kv_cache_block_size = kv_cache.shape[1]
    rows_per_block = kv_cache_block_size // compress_ratio
    # Raw-byte view of the NoPE plane: the kernel stores uint8 (fp8 payload,
    # e8m0 scales, zero pad) and uint8->uint8 keeps Triton pointer types
    # happy. The RoPE plane stays bf16.
    nope_plane = views.main_nope.view(torch.uint8)
    rope_plane = views.main_rope

    _fused_kv_compress_norm_rope_insert_2buff[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # 2buff planes
        kv_slot_mapping,
        kv_cache_block_size,
        nope_plane,
        rope_plane,
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        ROWS_PER_BLOCK=rows_per_block,
        NOPE_ROW_BYTES=V4_NOPE_ROW_BYTES,
        DIM_SCALE_DUP=V4_DIM_SCALE_DUP,
        PACK_OFF_SCALE=V4_DIM_NOPE,
        num_warps=4,
    )


# =============================================================================
# Ring slot allocation + CPU-side index translation (pure, unit-testable).
# =============================================================================


class V4RingSlotAllocator:
    """Host-side per-request ring slot allocator (ATOM ``_V4StateSlotAllocator``).

    Slots are stable while a request is resident and recycled when it leaves
    the batch. Stale ring rows from a recycled slot are never read: every
    prefill writes its last ``min(chunk_len, ring_slots)`` tokens and the
    window of the next chunk starts at ``chunk_start - win + 1 >= chunk_start
    - ring_slots``, so all rows a query can read were rewritten by its own
    request (see the invariant in the ring-scatter docstring). Runs outside
    the captured graph.
    """

    def __init__(self, max_slots: int):
        self._free: list[int] = list(range(max_slots))
        self._slot: dict[str, int] = {}

    def slot_for(self, req_ids: list[str]) -> np.ndarray:
        active = set(req_ids)
        for rid in [r for r in self._slot if r not in active]:
            self._free.append(self._slot.pop(rid))
        for rid in req_ids:
            if rid not in self._slot:
                if not self._free:
                    raise RuntimeError(
                        f"V4 ring slot allocator exhausted ({len(self._slot)} "
                        "resident > max_num_seqs); resize the pool or reduce "
                        "concurrency."
                    )
                self._slot[rid] = self._free.pop()
        return np.array([self._slot[rid] for rid in req_ids], dtype=np.int32)


def build_ring_indices_cpu(
    positions_np: np.ndarray,
    cu_seqlens_np: np.ndarray,
    num_computed_np: np.ndarray,
    slots_np: np.ndarray,
    ring_slots: int,
    window_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Per-token SWA ring row lists (decode: full window; prefill: pre-chunk
    prefix) -- pure numpy, mirrors ATOM's window arithmetic.

    Returns ``(ring_rows_per_token, prefix_ring_rows_per_token, batch_np)``:
      - ring_rows_per_token[t]: rows for q in [max(0, p-win+1), p] (decode) --
        the full window the query attends via the ring.
      - prefix_ring_rows_per_token[t]: rows for q in [max(0, p-win+1), s) with
        s = num_computed (prefill) -- the window part already in the ring;
        the in-chunk remainder is covered by the extend indices.
    Ring row = ``slot * ring_slots + (q % ring_slots)``.
    """
    bs = num_computed_np.shape[0]
    batch_np = np.zeros(positions_np.shape[0], dtype=np.int32)
    ring_rows_per_token: list[np.ndarray] = []
    prefix_ring_rows_per_token: list[np.ndarray] = []
    for b in range(bs):
        lo, hi = int(cu_seqlens_np[b]), int(cu_seqlens_np[b + 1])
        batch_np[lo:hi] = b
        slot = int(slots_np[b])
        chunk_start = int(num_computed_np[b])
        for t in range(lo, hi):
            p = int(positions_np[t])
            qs_full = np.arange(max(0, p - window_size + 1), p + 1, dtype=np.int64)
            ring_rows_per_token.append(
                (slot * ring_slots + qs_full % ring_slots).astype(np.int32)
            )
            qs_prefix = np.arange(
                max(0, p - window_size + 1), chunk_start, dtype=np.int64
            )
            prefix_ring_rows_per_token.append(
                (slot * ring_slots + qs_prefix % ring_slots).astype(np.int32)
            )
    return ring_rows_per_token, prefix_ring_rows_per_token, batch_np


def build_extend_indices_cpu(
    positions_np: np.ndarray,
    cu_seqlens_np: np.ndarray,
    num_computed_np: np.ndarray,
    window_size: int,
) -> list[np.ndarray]:
    """Per-token in-chunk window lists (prefill extend): flat token offsets
    into the forward's k_packed, i.e. ``cu_seqlens[b] + (q - chunk_start)``
    for q in [max(chunk_start, p-win+1), p).
    """
    bs = num_computed_np.shape[0]
    extend_per_token: list[np.ndarray] = []
    for b in range(bs):
        lo, hi = int(cu_seqlens_np[b]), int(cu_seqlens_np[b + 1])
        chunk_start = int(num_computed_np[b])
        for t in range(lo, hi):
            p = int(positions_np[t])
            qs = np.arange(max(chunk_start, p - window_size + 1), p, dtype=np.int64)
            extend_per_token.append(
                (lo + (qs - chunk_start)).astype(np.int32)
            )
    return extend_per_token


def ragged_from_lists(
    lists: list[np.ndarray], pad_to: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-token arrays into one ragged array + indptr, padded with a
    repeating tail so cudagraph-padded slots carry zero-length segments."""
    counts = np.array([len(x) for x in lists], dtype=np.int32)
    indptr = np.zeros(pad_to + 1, dtype=np.int32)
    if len(counts):
        indptr[1 : len(counts) + 1] = np.cumsum(counts)
        if pad_to > len(counts):
            indptr[len(counts) + 1 :] = indptr[len(counts)]
    flat = np.concatenate(lists).astype(np.int32) if lists else np.zeros(0, np.int32)
    return flat, indptr


@triton.jit
def _merge_ragged_indices_kernel(
    a_ptr,  # [sum_a] int32
    a_indptr_ptr,  # [T+1]
    b_ptr,  # [sum_b] int32
    b_indptr_ptr,  # [T+1]
    out_ptr,  # [sum_a + sum_b]
    out_indptr_ptr,  # [T+1]
    MAX_SEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Merge two ragged per-token segments (a then b) into one array."""
    t = tl.program_id(0)
    a_start = tl.load(a_indptr_ptr + t)
    a_end = tl.load(a_indptr_ptr + t + 1)
    b_start = tl.load(b_indptr_ptr + t)
    b_end = tl.load(b_indptr_ptr + t + 1)
    o_start = tl.load(out_indptr_ptr + t)
    a_len = a_end - a_start
    b_len = b_end - b_start
    for i in range(0, MAX_SEG, BLOCK):
        offs = tl.arange(0, BLOCK) + i
        am = offs < a_len
        tl.store(
            out_ptr + o_start + offs,
            tl.load(a_ptr + a_start + offs, mask=am, other=0),
            mask=am,
        )
        bm = offs < b_len
        tl.store(
            out_ptr + o_start + a_len + offs,
            tl.load(b_ptr + b_start + offs, mask=bm, other=0),
            mask=bm,
        )


def merge_ragged_indices(
    a: torch.Tensor,
    a_indptr: torch.Tensor,
    b: torch.Tensor,
    b_indptr: torch.Tensor,
    out: torch.Tensor,
    out_indptr: torch.Tensor,
    num_tokens: int,
    max_seg: int,
) -> None:
    """Merge ``a`` then ``b`` ragged per-token segments into ``out``.

    ``out_indptr`` is precomputed on the host as the elementwise sum of the
    two indptrs (both zero-based). ``max_seg`` is a host-known bound on the
    per-token segment width (max comp width, window size) -- no device
    reduction, capture-safe.
    """
    if num_tokens == 0:
        return
    _merge_ragged_indices_kernel[(num_tokens,)](
        a,
        a_indptr,
        b,
        b_indptr,
        out,
        out_indptr,
        MAX_SEG=max_seg,
        BLOCK=64,
    )
