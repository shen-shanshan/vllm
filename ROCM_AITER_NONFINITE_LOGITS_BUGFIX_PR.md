# [ROCm][Bugfix] Sanitize AITER paged-MQA logits before sparse top-k

## Summary

This change prevents a GPU memory access fault when serving DeepSeek V4 with
ROCm sparse attention on gfx950 under concurrent mixed prefill/decode workloads.

The ROCm AITER `deepgemm_fp8_paged_mqa_logits` kernel can intermittently return
`NaN` or infinite values within the valid logits range. The generic histogram
top-k kernel assumes finite, orderable inputs. Non-finite logits can leave some
of its shared-memory output slots unwritten, after which uninitialized values
are copied to the global top-k index buffer. Downstream sparse attention treats
large non-negative values as valid token indices and uses them for block-table
lookup, eventually causing an out-of-bounds GPU access.

This PR sanitizes the AITER decode logits at the ROCm call site before invoking
the generic top-k kernel. It does not modify `sampler.cu` or any NVIDIA path.

Related issue:
[Fangzhou-Ai/vllm#13](https://github.com/Fangzhou-Ai/vllm/issues/13)

## Root cause analysis

Runtime instrumentation captured the following failure sequence:

1. The valid range of an affected AITER paged-MQA output contained `NaN` and
   infinite logits.
2. `top_k_per_row_decode` then returned uninitialized values, including large
   positive indices and values below the `-1` sentinel.
3. The row bounds were valid, but the positive indices were far beyond their
   corresponding sequence lengths.
4. Sparse decode used those indices to address the block table and KV cache,
   causing the GPU memory fault.

The investigation ruled out the vLLM inputs and metadata:

- `q`, weights, FP8 KV-cache values, and KV scales were finite;
- the referenced physical blocks were within the allocated cache range;
- `seq_lens`, row bounds, `next_n`, and `num_rows` were consistent;
- graph padding and output views matched the actual token count;
- the output workspace was initialized to `-Inf`, so observed `NaN` values were
  not simply unwritten workspace entries;
- a direct FP32 computation using the same q/KV/scales/weights was finite where
  AITER returned `NaN`.

Changing graph mode did not eliminate the failure. Explicit `max_model_len`
values changed whether or when it was exposed, but several different values
failed, so sequence-length configuration was a trigger rather than the source
of the invalid address.

These results isolate the first non-finite values to the gfx950 AITER paged-MQA
output. The exact arithmetic or scheduling defect inside AITER remains to be
fixed upstream; this PR enforces the vLLM-side input contract for the shared
top-k consumer.

## Fix

Immediately after `rocm_fp8_paged_mqa_logits` and before
`top_k_per_row_decode`, normalize non-finite values in place:

- `NaN` and `-Inf` become the dtype's finite minimum;
- `+Inf` becomes the dtype's finite maximum.

This preserves the ordering direction of infinities while ensuring every input
to histogram top-k is finite and comparable. The top-k operator still uses
`seq_lens` to restrict each row's valid range, so converting padding from
`-Inf` to the finite minimum does not make padding selectable.

Keeping the workaround in `rocm_aiter_sparse_attn_indexer` limits it to the
affected ROCm AITER decode path and avoids changing the stable cross-platform
sampler kernel.

## Test plan and results

Before the fix:

- The concurrent serving workload failed with explicit `max_model_len` values
  of 8192, 9472, 9473, and 16384.
- The failure reproduced with `FULL_AND_PIECEWISE`, `PIECEWISE`, and `NONE`
  graph modes.
- Instrumentation observed non-finite AITER logits followed by out-of-range
  top-k indices and a GPU memory access fault.

After the fix:

- Instrumentation still observed non-finite values in the raw AITER output.
- The normalized logits contained no non-finite values.
- No top-k index was below `-1` or outside its row's sequence length.
- Three concurrent end-to-end runs completed without a GPU memory access
  fault.

Test workload: the exact server and client stress commands published in
[Fangzhou-Ai/vllm#13](https://github.com/Fangzhou-Ai/vllm/issues/13).

Model quality evaluation has not been run yet. It should be completed before
submission because replacing anomalous non-finite logits can affect token
selection even though all originally finite logits remain unchanged. The
underlying non-finite AITER output should also be investigated upstream.

## Duplicate-work check

Draft PR
[vllm-project/vllm#49666](https://github.com/vllm-project/vllm/pull/49666)
was an earlier investigation of the same report by the same author. It changed
the generic sampler kernel to initialize shared top-k output. This approach
supersedes that draft by handling the invalid AITER output at the ROCm call
site and leaving the shared NVIDIA/ROCm sampler kernel unchanged.

## AI assistance

AI assistance was used for debugging and report preparation. The human
submitter reviewed the evidence and is responsible for every changed line,
the test results, and the conclusions in this description.
