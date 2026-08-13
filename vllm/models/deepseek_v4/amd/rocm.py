# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.amd.fp8_2buff import (
    Atom2BuffPoolViews,
    V4RingSlotAllocator,
    atom2buff_available,
    atom2buff_reject_prefix_caching,
    build_extend_indices_cpu,
    build_ring_indices_cpu,
    kv_splits_heuristic,
    merge_ragged_indices,
    ragged_from_lists,
    rocm_fp8_2buff_decode,
    rocm_fp8_2buff_prefill,
    rocm_fp8_2buff_qk_norm_rope_quant,
    slice_atom2buff_pool_views,
    swa_ring_scatter_2buff,
    v4_atom2buff_ring_rows_from_config,
    v4_atom2buff_spec,
)
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    nnz = ragged_indices.numel()
    if nnz > 0:
        ragged_out[:nnz].copy_(ragged_indices, non_blocking=True)
    return ragged_out, indptr_out


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4Atom2BuffFp8Metadata(DeepseekV4ROCMAiterMLASparseMetadata):
    """2-buffer fp8 additions on top of the ROCm sparse metadata.

    The ring/extend index streams are built host-side by the builder; the
    compressed (CSA topk / HCA) part is translated in-forward (the topk buffer
    is only written during forward), so the merged index arrays and indptrs
    are assembled at forward time from the two components.
    """

    # Attached by the attention layer at forward time (before the compressor
    # store hook reads them): dense plane views + ring geometry.
    atom2buff_views: Atom2BuffPoolViews | None = None
    atom2buff_ring_rows: int = 0
    atom2buff_ring_slots: int = 0

    # Decode (op5): per-token SWA ring rows + merged [comp | ring] buffers.
    decode_ring_indices: torch.Tensor | None = None
    decode_ring_indptr: torch.Tensor | None = None
    decode_merged_buf: torch.Tensor | None = None
    decode_merged_indptr_buf: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    state_slot_per_seq: torch.Tensor | None = None

    # Prefill (op4): pre-chunk ring prefix + in-chunk extend offsets.
    prefix_ring_indices: torch.Tensor | None = None
    prefix_ring_indptr: torch.Tensor | None = None
    prefix_merged_buf: torch.Tensor | None = None
    prefix_merged_indptr_buf: torch.Tensor | None = None
    prefill_extend_indices: torch.Tensor | None = None
    prefill_extend_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if dense_decode is not None and decode_lens is not None:
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                dense_decode.reshape(dense_decode.shape[0], -1),
                decode_lens,
            )
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                dense_decode.shape[0],
                self.c128a_max_compressed,
            )

        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    # Keep fused multi-step decode disabled until update_draft_decode_metadata()
    # also refreshes the ROCm-specific ragged SWA indices and indptrs.
    supports_draft_decode_metadata_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(base.num_decode_tokens, -1),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                # Actual dense width for this build: window_size (causal) or
                # noncausal_index_width (DSpark non-causal draft).
                base.decode_swa_indices.shape[-1],
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


class DeepseekV4Atom2BuffFp8MetadataBuilder(
    DeepseekV4ROCMAiterMLASparseMetadataBuilder
):
    """Metadata builder for the ATOM 2-buffer fp8 path.

    Builds (host-side, outside the captured graph) the ring/extend index
    streams the op4/op5 kernels consume: per-token SWA ring rows (full window
    for decode, pre-chunk prefix for prefill), in-chunk extend offsets for
    prefill, per-request ring slots (keyed by the request's first block id --
    stable while resident, unique across concurrent requests, no prefix
    caching on this path), and ``qo_indptr = arange(N+1)``. The compressed
    (CSA topk / HCA) part is translated in-forward with the fork's existing
    ``compute_global_topk_ragged_indices_and_indptr`` (the topk buffer is only
    written during forward), and the merged indptr is a plain GPU add of the
    two component indptrs -- no host/device length parity to keep in sync.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.vllm_config
        sched = cfg.scheduler_config
        hf = cfg.model_config.hf_config
        self._2b_win = int(hf.sliding_window)
        spec = cfg.speculative_config
        spec_steps = (
            int(getattr(spec, "num_speculative_tokens", 0) or 0)
            if spec is not None
            else 0
        )
        self._2b_ring_slots = self._2b_win + spec_steps
        self._2b_ring_rows = sched.max_num_seqs * self._2b_ring_slots
        self._2b_max_comp = (
            self.c128a_max_compressed
            if self.compress_ratio == 128
            else self.topk_tokens
        )
        max_tokens = sched.max_num_batched_tokens
        max_seg = self._2b_win + self._2b_max_comp
        dev = self.device
        self._2b_decode_ring_buf = torch.empty(
            max_tokens * self._2b_win, dtype=torch.int32, device=dev
        )
        self._2b_decode_ring_indptr_buf = torch.empty(
            max_tokens + 1, dtype=torch.int32, device=dev
        )
        self._2b_decode_merged_buf = torch.empty(
            max_tokens * max_seg, dtype=torch.int32, device=dev
        )
        self._2b_decode_merged_indptr_buf = torch.empty(
            max_tokens + 1, dtype=torch.int32, device=dev
        )
        self._2b_prefix_ring_buf = torch.empty(
            max_tokens * self._2b_win, dtype=torch.int32, device=dev
        )
        self._2b_prefix_ring_indptr_buf = torch.empty(
            max_tokens + 1, dtype=torch.int32, device=dev
        )
        self._2b_prefix_merged_buf = torch.empty(
            max_tokens * max_seg, dtype=torch.int32, device=dev
        )
        self._2b_prefix_merged_indptr_buf = torch.empty(
            max_tokens + 1, dtype=torch.int32, device=dev
        )
        self._2b_extend_buf = torch.empty(
            max_tokens * self._2b_win, dtype=torch.int32, device=dev
        )
        self._2b_extend_indptr_buf = torch.empty(
            max_tokens + 1, dtype=torch.int32, device=dev
        )
        self._2b_qo_indptr_buf = torch.empty(
            max_tokens + 1, dtype=torch.int32, device=dev
        )
        self._2b_slots_buf = torch.empty(
            sched.max_num_seqs, dtype=torch.int32, device=dev
        )
        self._2b_slot_alloc = V4RingSlotAllocator(sched.max_num_seqs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "DeepseekV4Atom2BuffFp8Metadata":
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        md = DeepseekV4Atom2BuffFp8Metadata(
            **vars(base),
            atom2buff_ring_slots=self._2b_ring_slots,
        )
        num_actual = common_attn_metadata.num_actual_tokens
        num_reqs = common_attn_metadata.num_reqs
        if num_actual == 0 or num_reqs == 0:
            return md

        qsl = common_attn_metadata.query_start_loc_cpu.tolist()
        cu_np = np.asarray(qsl, dtype=np.int64)
        num_computed = common_attn_metadata._num_computed_tokens_cpu
        if num_computed is None:
            num_computed = common_attn_metadata.seq_lens.cpu()
        num_computed_np = np.asarray(num_computed, dtype=np.int64)[:num_reqs]
        positions = common_attn_metadata.positions
        assert positions is not None, "V4 2buff metadata requires positions"
        positions_np = positions[:num_actual].cpu().numpy().astype(np.int64)
        block_table = common_attn_metadata.block_table_tensor
        first_blocks_np = block_table[:num_reqs, 0].cpu().numpy()
        slots_np = self._2b_slot_alloc.slot_for([int(b) for b in first_blocks_np])

        ring_lists, prefix_lists, _batch_np = build_ring_indices_cpu(
            positions_np,
            cu_np,
            num_computed_np,
            slots_np,
            self._2b_ring_slots,
            self._2b_win,
        )
        extend_lists = build_extend_indices_cpu(
            positions_np, cu_np, num_computed_np, self._2b_win
        )

        ring_flat, ring_indptr = ragged_from_lists(ring_lists, num_actual)
        prefix_flat, prefix_indptr = ragged_from_lists(prefix_lists, num_actual)
        extend_flat, extend_indptr = ragged_from_lists(extend_lists, num_actual)
        qo_indptr = np.arange(num_actual + 1, dtype=np.int32)

        def _stage(dst: torch.Tensor, np_arr: np.ndarray) -> torch.Tensor:
            n = min(np_arr.shape[0], dst.shape[0])
            dst[:n].copy_(torch.from_numpy(np_arr[:n]).to(dst.device))
            return dst[: np_arr.shape[0]]

        md.decode_ring_indices = _stage(self._2b_decode_ring_buf, ring_flat)
        md.decode_ring_indptr = _stage(self._2b_decode_ring_indptr_buf, ring_indptr)
        md.prefix_ring_indices = _stage(self._2b_prefix_ring_buf, prefix_flat)
        md.prefix_ring_indptr = _stage(
            self._2b_prefix_ring_indptr_buf, prefix_indptr
        )
        md.prefill_extend_indices = _stage(self._2b_extend_buf, extend_flat)
        md.prefill_extend_indptr = _stage(self._2b_extend_indptr_buf, extend_indptr)
        md.qo_indptr = _stage(self._2b_qo_indptr_buf, qo_indptr)
        md.state_slot_per_seq = _stage(self._2b_slots_buf, slots_np)
        md.decode_merged_buf = self._2b_decode_merged_buf
        md.prefix_merged_buf = self._2b_prefix_merged_buf
        md.decode_merged_indptr_buf = self._2b_decode_merged_indptr_buf
        md.prefix_merged_indptr_buf = self._2b_prefix_merged_indptr_buf
        return md


class DeepseekV4Atom2BuffFp8Backend(DeepseekV4ROCMAiterMLASparseBackend):
    """Backend for the ATOM 2-buffer fp8 path.

    Reuses the existing backend name on purpose: v1 dispatch is
    ``backend_cls``-based (``AttentionLayerBase.get_attn_backend``), so no
    registry entry is needed and ``--attention-backend`` override bookkeeping
    keeps resolving to the registered sparse backend.
    """

    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4Atom2BuffFp8MetadataBuilder"]:
        return DeepseekV4Atom2BuffFp8MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Flat byte pool: head_size already carries NoPE + RoPE + amortized
        # ring bytes per token (see fp8_2buff.v4_atom2buff_spec), so the
        # fp8_ds_mla 584B special case of the sparse backend does not apply.
        return (num_blocks, block_size, head_size)


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Block scale for the preshuffled weight; None = not preshuffled.
        self._wqa_wkv_scale: torch.Tensor | None = None
        self._wo_b_scale: torch.Tensor | None = None

        # ---- ATOM 2-buffer fp8 gate (env + arch + aiter kernels) ----
        # Compressed (CSA/HCA) layers only: dense (ratio<=1) layers keep the
        # existing SWA-cache + Triton path untouched.
        vllm_config = kwargs.get("vllm_config") or args[0]
        self._atom_2buff = self.compress_ratio > 1 and atom2buff_available(
            self.kv_cache_dtype
        )
        self._atom2buff_ring_rows = 0
        self._atom2buff_views_cache: tuple[int, Atom2BuffPoolViews] | None = None
        if self._atom_2buff:
            cache_config = vllm_config.cache_config
            atom2buff_reject_prefix_caching(
                getattr(cache_config, "enable_prefix_caching", False)
            )
            self._atom2buff_ring_rows = v4_atom2buff_ring_rows_from_config(
                vllm_config
            )
            hf = vllm_config.model_config.hf_config
            spec = vllm_config.speculative_config
            spec_steps = (
                int(getattr(spec, "num_speculative_tokens", 0) or 0)
                if spec is not None
                else 0
            )
            self._atom2buff_ring_slots = int(hf.sliding_window) + spec_steps
            self._2b_topk_tokens = int(hf.index_topk)
            # Mirror DeepseekV4SparseMLAMetadataBuilder.c128a_max_compressed
            # (sparse_mla.py): cdiv(max_model_len, 128), 128-aligned.
            self._2b_max_comp = (
                (self.max_model_len + 127) // 128 + 127
            ) // 128 * 128
            self.backend_cls = DeepseekV4Atom2BuffFp8Backend
        self._qkn_2buff: tuple[torch.Tensor, ...] | None = None
        self._2b_decode_scratch: dict[str, torch.Tensor] | None = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def prepare_attn_preshuffle(self) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )
        from vllm.model_executor.utils import replace_parameter

        def _prep(linear) -> torch.Tensor | None:
            w = getattr(linear, "weight", None)
            if w is None or w.dim() != 2:
                return None
            # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
            if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
                return None
            ws = getattr(linear, "weight_scale_inv", None)  # per-block scale
            if ws is None:
                return None
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _upcast_e8m0_to_fp32(ws).contiguous()
            # Shuffle the weight in place (single weight, no unshuffled copy).
            replace_parameter(
                linear,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
            return ws

        self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)
        self._wo_b_scale = _prep(self.wo_b)

    def _bpre_attn_gemm(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        reduce_tp: bool,
    ) -> torch.Tensor:
        from vllm._aiter_ops import rocm_aiter_ops

        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, weight, x_scale, scale, output_dtype=x.dtype
        )
        if reduce_tp and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
            return self._bpre_attn_gemm(
                self.fused_wqa_wkv.weight, self._wqa_wkv_scale, hidden_states, False
            )
        return super()._fused_wqa_wkv_gemm(hidden_states)

    def get_kv_cache_spec(self, vllm_config):
        if self._atom_2buff:
            return v4_atom2buff_spec(vllm_config, self.compress_ratio)
        return super().get_kv_cache_spec(vllm_config)

    def _atom2buff_pool_views(self) -> Atom2BuffPoolViews:
        """Dense plane views over the runner-bound 2-buffer pool.

        v1 re-binds ``self.kv_cache`` every step; slicing is idempotent and
        cached by data pointer so the views are computed once per allocation.
        """
        pool = self.kv_cache
        key = pool.data_ptr()
        cached = self._atom2buff_views_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        block_size = pool.shape[1]
        views = slice_atom2buff_pool_views(
            pool,
            ring_rows=self._atom2buff_ring_rows,
            rows_per_block=block_size // self.compress_ratio,
        )
        self._atom2buff_views_cache = (key, views)
        return views

    def _fused_qnorm_rope_kv_insert(
        self, q, kv, positions, attn_metadata
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict) or not self._atom_2buff:
            # Profile/dummy runs take the base path (which pads q and skips the
            # kernels); the 2buff branch needs real per-step metadata.
            return super()._fused_qnorm_rope_kv_insert(
                q, kv, positions, attn_metadata
            )

        md = cast(DeepseekV4Atom2BuffFp8Metadata, attn_metadata[self.prefix])
        swa_md = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata,
            attn_metadata[self.swa_cache_layer.prefix],
        )
        views = self._atom2buff_pool_views()
        md.atom2buff_views = views
        md.atom2buff_ring_rows = self._atom2buff_ring_rows

        q_packed, q_rope, k_packed, k_rope = rocm_fp8_2buff_qk_norm_rope_quant(
            q,
            kv,
            self.kv_norm.weight.data,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            self.n_local_heads,
            self.head_dim,
        )
        self._qkn_2buff = (q_packed, q_rope, k_packed, k_rope)

        # Decode: scatter every token (verified + drafts) into the ring BEFORE
        # attention -- the draft tokens attend rows written this same step.
        # Runs inside the eager break (attention_impl), so real batch counts
        # and dynamic shapes are fine here.
        num_decode_tokens = swa_md.num_decode_tokens
        if num_decode_tokens > 0:
            num_decodes = swa_md.num_decodes
            cu = swa_md.query_start_loc[: num_decodes + 1]
            slots = md.state_slot_per_seq[:num_decodes]
            write_per_batch = min(
                swa_md.max_decode_query_len, self._atom2buff_ring_slots
            )
            swa_ring_scatter_2buff(
                k_packed[:num_decode_tokens].squeeze(1),
                k_rope[:num_decode_tokens].squeeze(1),
                positions[:num_decode_tokens],
                cu,
                slots,
                views.ring_nope,
                views.ring_rope,
                self._atom2buff_ring_slots,
                write_per_batch,
            )
        return q

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        zf = z.flatten(1)
        if self._wo_b_scale is not None and zf.dim() == 2:
            return self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        return self.wo_b(zf)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        if self._atom_2buff:
            self._forward_2buff(q, kv, positions, output, attn_metadata)
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_2buff(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: dict,
    ) -> None:
        md = cast(DeepseekV4Atom2BuffFp8Metadata, attn_metadata[self.prefix])
        swa_md = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata,
            attn_metadata[self.swa_cache_layer.prefix],
        )
        qkn = self._qkn_2buff
        assert qkn is not None
        q_packed, q_rope, k_packed, k_rope = qkn
        views = md.atom2buff_views
        assert views is not None

        num_decode_tokens = swa_md.num_decode_tokens
        if num_decode_tokens > 0:
            self._forward_2buff_decode(
                q_packed[:num_decode_tokens],
                q_rope[:num_decode_tokens],
                views,
                md,
                swa_md,
                output[:num_decode_tokens],
            )
        if swa_md.num_prefills > 0:
            self._forward_2buff_prefill(
                q_packed[num_decode_tokens:],
                q_rope[num_decode_tokens:],
                k_packed[num_decode_tokens:],
                k_rope[num_decode_tokens:],
                positions[num_decode_tokens:],
                views,
                md,
                swa_md,
                output[num_decode_tokens:],
            )

    def _2b_get_decode_scratch(
        self, q_packed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Preallocated GQA pad scratch for op5 (H < 16, e.g. TP8 -> 8 heads).

        Runs in the eager break, so the lazy allocation is capture-safe;
        cached on the layer for reuse. The fp8 scratch follows the quant
        kernel's actual q_packed dtype (aiter dtypes.fp8), captured on first
        use.
        """
        if self._2b_decode_scratch is None:
            dev = q_packed.device
            max_tokens = self.max_num_batched_tokens
            pad_heads = 16
            self._2b_decode_scratch = {
                "q_packed_pad": torch.empty(
                    (max_tokens, pad_heads, self.head_dim),
                    dtype=q_packed.dtype,
                    device=dev,
                ),
                "q_rope_pad": torch.empty(
                    (max_tokens, pad_heads, self.rope_head_dim),
                    dtype=torch.bfloat16,
                    device=dev,
                ),
                "sink_pad": torch.empty(
                    (pad_heads,), dtype=torch.float32, device=dev
                ),
                "out_pad": torch.empty(
                    (max_tokens, pad_heads, self.head_dim),
                    dtype=torch.bfloat16,
                    device=dev,
                ),
            }
        return self._2b_decode_scratch

    def _forward_2buff_decode(
        self,
        q_packed: torch.Tensor,
        q_rope: torch.Tensor,
        views: Atom2BuffPoolViews,
        md: DeepseekV4Atom2BuffFp8Metadata,
        swa_md: DeepseekV4ROCMAiterSparseSWAMetadata,
        output: torch.Tensor,
    ) -> None:
        ndec = q_packed.shape[0]
        num_decodes = swa_md.num_decodes
        block_size = md.block_size // self.compress_ratio
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            is_valid = swa_md.is_valid_token[:ndec]
            comp, comp_indptr, _ = compute_global_topk_ragged_indices_and_indptr(
                self.topk_indices_buffer[:ndec],
                swa_md.token_to_req_indices,
                md.block_table[:num_decodes],
                block_size,
                is_valid,
            )
        else:
            comp = md.c128a_decode_topk_ragged_indices
            comp_indptr = md.c128a_decode_topk_ragged_indptr
        comp = comp + self._atom2buff_ring_rows

        ring_indices = md.decode_ring_indices
        ring_indptr = md.decode_ring_indptr
        assert ring_indices is not None and ring_indptr is not None
        merged_indptr = md.decode_merged_indptr_buf[: ndec + 1]
        torch.add(
            comp_indptr[: ndec + 1],
            ring_indptr[: ndec + 1],
            out=merged_indptr,
        )
        merged = md.decode_merged_buf
        assert merged is not None
        max_seg = self._atom2buff_ring_slots + (
            self._2b_topk_tokens if self.compress_ratio == 4 else self._2b_max_comp
        )
        merge_ragged_indices(
            comp,
            comp_indptr[: ndec + 1],
            ring_indices,
            ring_indptr[: ndec + 1],
            merged,
            merged_indptr,
            ndec,
            max_seg,
        )

        num_kv_splits = kv_splits_heuristic(ndec, self.n_local_heads)
        scratch = (
            self._2b_get_decode_scratch(q_packed) if self.n_local_heads < 16 else None
        )
        out = rocm_fp8_2buff_decode(
            q_packed,
            q_rope,
            views.unified_nope,
            views.unified_rope,
            merged,
            merged_indptr,
            md.qo_indptr,
            self.attn_sink,
            self.scale,
            num_kv_splits,
            **(scratch or {}),
        )
        output.copy_(out)

    def _forward_2buff_prefill(
        self,
        q_packed: torch.Tensor,
        q_rope: torch.Tensor,
        k_packed: torch.Tensor,
        k_rope: torch.Tensor,
        positions: torch.Tensor,
        views: Atom2BuffPoolViews,
        md: DeepseekV4Atom2BuffFp8Metadata,
        swa_md: DeepseekV4ROCMAiterSparseSWAMetadata,
        output: torch.Tensor,
    ) -> None:
        npref = q_packed.shape[0]
        num_decodes = swa_md.num_decodes
        num_decode_tokens = swa_md.num_decode_tokens
        block_size = md.block_size // self.compress_ratio
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            topk = self.topk_indices_buffer[num_decode_tokens:][:npref]
        else:
            topk = md.c128a_prefill_topk_indices
        is_valid = torch.ones(npref, dtype=torch.bool, device=q_packed.device)
        comp, comp_indptr, _ = compute_global_topk_ragged_indices_and_indptr(
            topk,
            swa_md.token_to_req_indices[num_decode_tokens:],
            md.block_table[num_decodes:],
            block_size,
            is_valid,
        )
        comp = comp + self._atom2buff_ring_rows

        # Rebase the prefix-ring stream to the prefill token range.
        ring_indptr_all = md.prefix_ring_indptr
        ring_indices_all = md.prefix_ring_indices
        assert ring_indptr_all is not None and ring_indices_all is not None
        base = ring_indptr_all[num_decode_tokens]
        ring_indptr = (
            ring_indptr_all[num_decode_tokens : num_decode_tokens + npref + 1]
            - base
        )
        end = ring_indptr_all[num_decode_tokens + npref]
        ring_indices = ring_indices_all[base:end]

        # Same rebase for the extend stream (built over all tokens, decode
        # first; op4 consumes the prefill range as [0, npref)).
        ext_indptr_all = md.prefill_extend_indptr
        ext_indices_all = md.prefill_extend_indices
        assert ext_indptr_all is not None and ext_indices_all is not None
        ext_base = ext_indptr_all[num_decode_tokens]
        ext_indptr = (
            ext_indptr_all[num_decode_tokens : num_decode_tokens + npref + 1]
            - ext_base
        )
        ext_end = ext_indptr_all[num_decode_tokens + npref]
        ext_indices = ext_indices_all[ext_base:ext_end]

        merged_indptr = md.prefix_merged_indptr_buf[: npref + 1]
        torch.add(comp_indptr[: npref + 1], ring_indptr, out=merged_indptr)
        merged = md.prefix_merged_buf
        assert merged is not None
        max_seg = self._atom2buff_ring_slots + (
            self._2b_topk_tokens if self.compress_ratio == 4 else self._2b_max_comp
        )
        merge_ragged_indices(
            comp,
            comp_indptr[: npref + 1],
            ring_indices,
            ring_indptr,
            merged,
            merged_indptr,
            npref,
            max_seg,
        )

        out = rocm_fp8_2buff_prefill(
            q_packed,
            q_rope,
            views.unified_nope,
            views.unified_rope,
            merged,
            merged_indptr,
            k_packed.squeeze(1),
            k_rope.squeeze(1),
            ext_indices,
            ext_indptr,
            self.attn_sink,
            self.scale,
        )
        output.copy_(out)

        # Post-attention SWA ring write of the chunk tail (the in-chunk window
        # part must not overwrite ring rows the prefill itself still reads).
        num_prefills = swa_md.num_prefills
        cu = swa_md.query_start_loc[num_decodes : num_decodes + num_prefills + 1]
        cu = cu - num_decode_tokens
        slots = md.state_slot_per_seq[num_decodes:]
        assert swa_md.prefill_query_lens_cpu is not None
        max_prefill_q = int(swa_md.prefill_query_lens_cpu.max())
        write_per_batch = min(max_prefill_q, self._atom2buff_ring_slots)
        swa_ring_scatter_2buff(
            k_packed.squeeze(1),
            k_rope.squeeze(1),
            positions,
            cu,
            slots,
            views.ring_nope,
            views.ring_rope,
            self._atom2buff_ring_slots,
            write_per_batch,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )
