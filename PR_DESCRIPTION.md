## Summary

Enable and document ViT encoder CUDA graph (`cudagraph_mm_encoder`) validation on AMD MI350X (gfx950), addressing [#38175](https://github.com/vllm-project/vllm/issues/38175).

- Extend ROCm coverage in `test_vit_cudagraph.py` and `test_encoder_cudagraph.py` by gating on `is_cuda_alike()` instead of `is_cuda()`.
- Add `qwen3_5_moe` test config and reorder/enable `gemma4` in the core-model pytest suite.
- Skip `llama4` on ROCm (gated HuggingFace config).
- Add `test_vit_cudagraph.py` to the MI355/gfx950 AMD Buildkite multimodal job.
- Document MI350X manual and pytest validation results in `docs/design/cuda_graphs_multimodal.md`.

No functional changes to encoder CUDA graph capture/replay logic were required; existing ROCm handling in `gpu_model_runner.py` is sufficient.

## Motivation

ViT full CUDA graph support was validated on NVIDIA Blackwell but lacked documented end-to-end validation on AMD MI350X. This PR adds ROCm pytest coverage, CI wiring, and a validation results table so MI350X users can rely on `--compilation-config '{"cudagraph_mm_encoder": true}'` for supported architectures.

## Duplicate-work check

- No open PR found that adds MI350X ViT CUDA graph validation, ROCm pytest enablement, and MI355 CI for `test_vit_cudagraph.py` as a combined change for #38175.

## Test plan

### Pre-commit

```bash
pre-commit run --files \
  tests/models/multimodal/generation/test_vit_cudagraph.py \
  tests/v1/cudagraph/test_encoder_cudagraph.py \
  .buildkite/test-amd.yaml \
  docs/design/cuda_graphs_multimodal.md
```

Result: **All hooks passed.**

### Pytest on MI350X (ROCm, gfx950)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_VISIBLE_DEVICES=2
unset CUDA_VISIBLE_DEVICES

.venv/bin/python -m pytest \
  tests/models/multimodal/generation/test_vit_cudagraph.py \
  -m core_model -v
```

Result: **`17 passed, 5 skipped, 0 failed`** (~29 min)

| Test | Result |
| ---- | ------ |
| `gemma4` image + video | PASSED |
| `qwen2_vl` image + video | PASSED |
| `qwen2_5_vl` image + video | PASSED |
| `kimi_vl` image | PASSED |
| `qwen3_vl` image + video | PASSED |
| `qwen3_5` image + video | PASSED |
| `qwen3_5_moe` image + video | PASSED |
| `internvl` image + video | PASSED |
| `glm4_1v` image + video | PASSED |
| `llama4` | SKIPPED (gated HF config on ROCm) |
| `deepseek_ocr` | SKIPPED (existing OOM skip) |
| `kimi_vl` video | SKIPPED (no video modality in config) |
| `llama4` video | SKIPPED |

### Manual end-to-end validation on MI350X

Used local-only script (not committed) wrapping
`examples/generate/multimodal/vision_language_offline.py` with
`--enable-vit-cuda-graph`, comparing outputs to eager baseline.
Models stored under `/shared/models/`. Attn backend: `FLASH_ATTN`.

| Model | Image | Video | Status |
| ----- | ----- | ----- | ------ |
| Qwen3.5-0.8B | Pass | Pass | Verified |
| Qwen2-VL-2B | Pass | Pass | Verified |
| Qwen2.5-VL-3B | Pass | Pass | Verified |
| Qwen3-VL-2B | Pass | Pass | Verified |
| Qwen3.5-35B-A3B (MoE) | Pass | Pass | Verified |
| GLM-4.6V-Flash | Pass | Pass | Verified |
| Gemma-4-E2B | Pass | — | Verified (image only) |
| InternVL3-1B | — | — | Not verified (manual skipped) |
| Kimi-VL-A3B | — | — | Not verified (manual skipped) |
| DeepSeek-OCR | — | — | Not verified (manual skipped) |
| Step3-VL-10B | — | — | Not verified (manual skipped) |
| Gemma3 | — | — | Not verified (gated weights) |
| Llama 4 | — | — | Not verified (gated weights) |

Architectures marked "Not verified (manual skipped)" were not re-run manually;
several (`gemma4` video, `internvl`, `kimi_vl`) are covered by the ROCm pytest
suite above.

## AI assistance disclosure

This PR was prepared with AI assistance. The submitting author has reviewed all
changed lines and validation results.
