# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the ATOM 2-buffer fp8 KV-cache layout (P1 skeleton).

CPU-safe: the layout math, spec, and view slicing do not touch the GPU, and
the aiter gate is exercised through fake modules.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.models.deepseek_v4.amd.fp8_2buff as f2b


@pytest.fixture(autouse=True)
def _reset_gate_cache():
    f2b._ATOM2BUFF_KERNELS_OK = None
    yield
    f2b._ATOM2BUFF_KERNELS_OK = None


def _fake_vllm_config(
    block_size=256,
    max_model_len=16384,
    max_num_seqs=256,
    window=128,
    spec_steps=0,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            hf_config=SimpleNamespace(sliding_window=window),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_steps) if spec_steps else None
        ),
    )


def _install_fake_aiter(monkeypatch, gfx):
    for name in (
        "aiter",
        "aiter.jit",
        "aiter.jit.utils",
        "aiter.ops",
        "aiter.ops.pa_sparse_prefill_opus",
        "aiter.ops.fused_qk_norm_rope_cache_quant",
        "aiter.mla",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    chip = types.ModuleType("aiter.jit.utils.chip_info")
    chip.get_gfx = lambda: gfx
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", chip)
    monkeypatch.setattr(
        sys.modules["aiter.ops.pa_sparse_prefill_opus"],
        "pa_sparse_prefill_fp8_opus",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        sys.modules["aiter.ops.fused_qk_norm_rope_cache_quant"],
        "fused_qk_norm_rope_group_quant",
        lambda *a, **kw: None,
    )


def test_ring_rows_and_head_size_math():
    ring_rows = f2b.atom2buff_ring_rows(max_num_seqs=256, window_size=128)
    assert ring_rows == 256 * 128

    # Spec steps widen the ring so draft tokens can write their KV.
    ring_rows_spec = f2b.atom2buff_ring_rows(256, 128, spec_steps=1)
    assert ring_rows_spec == 256 * 129

    # block 256, max_model_len 16384 -> min_blocks 64; ring amortized evenly.
    head_size = f2b.atom2buff_head_size(
        block_size=256, min_blocks=64, ring_rows=ring_rows
    )
    ring_bytes = ring_rows * f2b.V4_ENTRY_BYTES
    assert head_size == f2b.V4_ENTRY_BYTES + ring_bytes // (64 * 256)
    # The pool must hold the full ring once num_blocks >= min_blocks.
    assert 64 * 256 * (head_size - f2b.V4_ENTRY_BYTES) >= ring_bytes


def test_head_size_from_config_rounds_up():
    cfg = _fake_vllm_config(max_model_len=16384, max_num_seqs=256)
    head_size = f2b.v4_atom2buff_head_size_from_config(cfg)
    # Exact division for these numbers; spot-check the composition instead.
    assert head_size > f2b.V4_ENTRY_BYTES
    assert head_size == f2b.atom2buff_head_size(256, 64, 256 * 128)


def test_head_size_from_config_not_divisible():
    # ring_bytes = 7 * 640 = 4480; min_blocks*block = 2*256 = 512 -> share 9.
    cfg = _fake_vllm_config(block_size=256, max_model_len=512, max_num_seqs=7)
    assert f2b.v4_atom2buff_head_size_from_config(cfg) == f2b.V4_ENTRY_BYTES + 9


def test_spec_fields():
    cfg = _fake_vllm_config()
    spec = f2b.v4_atom2buff_spec(cfg, compress_ratio=4)
    assert spec.block_size == 256
    assert spec.num_kv_heads == 1
    assert spec.dtype == torch.uint8
    assert spec.cache_dtype_str == "fp8_ds_mla"
    assert spec.model_version == "deepseek_v4"
    assert spec.compress_ratio == 4
    assert spec.storage_block_size == 64
    # One flat byte row per token: block_size * head_size bytes per page.
    assert spec.real_page_size_bytes == 256 * spec.head_size
    assert spec.page_size_bytes == spec.real_page_size_bytes


def test_spec_is_registry_resolvable():
    # The KVCacheSpecRegistry walks the MRO: the subclass must resolve to the
    # MLAAttentionSpec registration without an explicit register call.
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    cfg = _fake_vllm_config()
    spec = f2b.v4_atom2buff_spec(cfg, compress_ratio=4)
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is not None
    assert KVCacheSpecRegistry.get_manager_class(spec) is not None


def test_pool_slicing_dense_views():
    head_size = 1920
    ring_rows = 4
    pool = torch.zeros((2, 256, head_size), dtype=torch.uint8)
    views = f2b.slice_atom2buff_pool_views(pool, ring_rows=ring_rows, rows_per_block=64)

    main_rows = 2 * 64
    assert views.unified_nope.shape == (ring_rows + main_rows, 512)
    assert views.unified_rope.shape == (ring_rows + main_rows, 64)
    assert views.ring_nope.shape == (ring_rows, 512)
    assert views.main_nope.shape == (main_rows, 512)
    assert views.unified_nope.dtype == torch.float8_e4m3fnuz
    assert views.unified_rope.dtype == torch.bfloat16
    # The aiter asm kernels require dense, uninterleaved plane rows.
    assert views.unified_nope.is_contiguous()
    assert views.unified_rope.is_contiguous()
    assert views.unified_nope.stride() == (512, 1)
    assert views.unified_rope.stride() == (64, 1)

    # Region order: swa_nope | main_nope | swa_rope | main_rope.
    flat = pool.reshape(-1)
    flat[: ring_rows * 512] = 7
    assert views.ring_nope.view(torch.uint8).eq(7).all()
    nope_bytes = (ring_rows + main_rows) * 512
    flat[ring_rows * 512 + 5 * 512 + 3] = 9
    assert views.main_nope.view(torch.uint8)[5, 3] == 9
    # RoPE plane starts right after the NoPE plane; row 6 = main_rope row 2
    # (ring rows 0..3 occupy the head of the unified RoPE plane).
    rope_base = nope_bytes
    flat[rope_base + 6 * 128] = 0xC0
    flat[rope_base + 6 * 128 + 1] = 0x3F  # bf16 1.5, little-endian
    assert views.unified_rope[6, 0].item() == 1.5
    assert views.main_rope[2, 0].item() == 1.5
    assert views.ring_rope.view(torch.uint8).eq(0).all()


def test_pool_slicing_rejects_undersized_pool():
    # Need (4 ring + 1 * 64 main) * 640 = 43520 bytes; pool has 40960.
    pool = torch.zeros((1, 64, 640), dtype=torch.uint8)
    with pytest.raises(ValueError, match="pool too small"):
        f2b.slice_atom2buff_pool_views(pool, ring_rows=4, rows_per_block=64)


def test_pool_slicing_rejects_non_uint8():
    pool = torch.zeros((1, 256, 1920), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be uint8"):
        f2b.slice_atom2buff_pool_views(pool, ring_rows=4, rows_per_block=64)


def test_gate_env_off(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_DSV4_FP8", "0")
    assert not f2b.atom2buff_available("fp8_ds_mla")
    assert not f2b.atom2buff_available("auto")


def test_gate_wrong_dtype(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_DSV4_FP8", "1")
    _install_fake_aiter(monkeypatch, gfx="gfx950")
    assert not f2b.atom2buff_available("auto")
    assert not f2b.atom2buff_available("bfloat16")


def test_gate_enabled_on_gfx950(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_DSV4_FP8", "1")
    _install_fake_aiter(monkeypatch, gfx="gfx950")
    assert f2b.atom2buff_available("fp8_ds_mla")


def test_gate_unsupported_arch_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_DSV4_FP8", "1")
    _install_fake_aiter(monkeypatch, gfx="gfx90a")
    assert not f2b.atom2buff_available("fp8_ds_mla")


def test_prefix_caching_rejection():
    f2b.atom2buff_reject_prefix_caching(False)  # no-op
    with pytest.raises(ValueError, match="does not support prefix caching"):
        f2b.atom2buff_reject_prefix_caching(True)


def test_mtp_decode_ring_indices():
    # MTP decode: one request at main position 10 with k=2 drafts. Per-token
    # positions are main_pos + t; every draft's window must include the main
    # token's ring row (written by the pre-attention scatter of this step).
    positions = np.array([10, 11, 12], dtype=np.int64)
    cu = np.array([0, 3], dtype=np.int64)
    num_computed = np.array([10], dtype=np.int64)
    slots = np.array([3], dtype=np.int32)
    ring_slots, win = 129, 128
    ring_lists, _prefix, batch = f2b.build_ring_indices_cpu(
        positions, cu, num_computed, slots, ring_slots, win
    )
    base = 3 * 129
    assert ring_lists[0].tolist() == [base + q % 129 for q in range(0, 11)]
    assert ring_lists[1].tolist() == [base + q % 129 for q in range(0, 12)]
    assert ring_lists[2].tolist() == [base + q % 129 for q in range(0, 13)]
    # The main token's row (position 10) is inside every draft's window.
    main_row = base + 10 % 129
    assert main_row in ring_lists[1].tolist()
    assert main_row in ring_lists[2].tolist()
    assert batch.tolist() == [0, 0, 0]


def test_gate_missing_aiter_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_DSV4_FP8", "1")
    # A fake "aiter" package with an empty __path__ makes every submodule
    # import fail deterministically, on machines with and without aiter.
    fake_aiter = types.ModuleType("aiter")
    fake_aiter.__path__ = []
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    for name in list(sys.modules):
        if name.startswith("aiter."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    assert not f2b.atom2buff_available("fp8_ds_mla")
