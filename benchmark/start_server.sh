#!/usr/bin/env bash
# Start DeepSeek-V4-Pro on ROCm — aligned with InferenceX dsv4_fp4_mi355x_vllm.sh
# Uses editable vLLM from /home/amd/vllm/.venv (pip install -e .).
#
# Default: no --max-model-len / --max-num-seqs / --max-num-batched-tokens so vLLM
# auto-tunes on MI355 (max_num_batched_tokens=8192, max_num_seqs=1024, max_model_len=1M).
# Optional overrides via env for high-conc OOM workarounds:
#   MAX_MODEL_LEN=9472 MAX_NUM_SEQS=512 MAX_NUM_BATCHED_TOKENS=16384 ./start_server.sh
#
# Override binary for container-only runs (e.g. nightly image sweep):
#   VLLM_BIN=/usr/local/bin/vllm ./start_server.sh
set -euo pipefail

VLLM_ROOT=/home/shashen/vllm/vllm
VLLM_BIN="${VLLM_BIN:-${VLLM_ROOT}/.venv/bin/vllm}"
VLLM_PY="${VLLM_PY:-$(dirname "${VLLM_BIN}")/python}"

# Avoid cwd shadowing; venv editable install resolves the repo via .pth.
unset PYTHONPATH VIRTUAL_ENV
unset HSA_NO_SCRATCH_RECLAIM
cd /tmp

# export MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Pro}
export MODEL=${MODEL:-/shared/models/deepseek-ai/DeepSeek-V4-Pro}
export TP=${TP:-8}
export PORT=${PORT:-8333}
export VLLM_ROCM_USE_AITER=1
unset VLLM_ROCM_USE_AITER_MOE VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
if [[ "${LEGACY_SHARED_FUSION:-0}" == "1" ]]; then
  export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
fi
if [[ -n "${GPU_MAX_HW_QUEUES:-}" ]]; then
  if ! [[ "${GPU_MAX_HW_QUEUES}" =~ ^[0-9]+$ ]] || ((GPU_MAX_HW_QUEUES < 1)); then
    echo "ERROR: GPU_MAX_HW_QUEUES must be a positive integer" >&2
    exit 1
  fi
  export GPU_MAX_HW_QUEUES
fi
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
else
  export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
fi

# OUTDIR=${OUTDIR:-/home/shashen/vllm/vllm/dsv4_rocm_bench/inferencex_tests}
OUTDIR=${OUTDIR:-/home/shashen/vllm/vllm/benchmark}
# mkdir -p "${OUTDIR}"

VLLM_MODULE="$("${VLLM_PY}" -c 'import vllm; print(vllm.__file__)' 2>/dev/null || true)"
VLLM_VERSION="$("${VLLM_BIN}" --version 2>&1 | head -1 || true)"

# Profiling
export PROFILE_STEPS=6
# export MAX_NUM_SEQS=64
# export MAX_NUM_BATCHED_TOKENS=8192
export PROFILE_ROOT=/home/shashen/vllm/profiles/deepseek-v4
mkdir -p "${PROFILE_ROOT}"

# export CONC=$1
export PROFILE_DIR="${PROFILE_ROOT}/torch/conc_${CONC}"
rm -rf "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"

export PROFILER_CONFIG=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","delay_iterations":5,"max_iterations":%d,"ignore_frontend":true,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"torch_profiler_with_stack":%s,"torch_profiler_with_flops":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true}' "${PROFILE_DIR}" "${PROFILE_STEPS}" "${WITH_STACK:-false}")

{
  echo "MODEL=${MODEL}"
  echo "TP=${TP}"
  echo "PORT=${PORT}"
  echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
  echo "VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}"
  echo "LEGACY_SHARED_FUSION=${LEGACY_SHARED_FUSION:-0}"
  echo "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-<unset>}"
  echo "GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-<unset>}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-<vllm-default>}"
  echo "MAX_NUM_SEQS=${MAX_NUM_SEQS:-<vllm-default>}"
  echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-<vllm-default>}"
  echo "PROFILER_CONFIG_JSON=${PROFILER_CONFIG_JSON:-<unset>}"
  echo "OUTDIR=${OUTDIR}"
  echo "VLLM_BIN=${VLLM_BIN}"
  echo "VLLM_PY=${VLLM_PY}"
  echo "VLLM_MODULE=${VLLM_MODULE}"
  echo "VLLM_VERSION=${VLLM_VERSION}"
  echo "CWD=$(pwd)"
} | tee "${OUTDIR}/server_env.txt"

if [ -z "${COMPILATION_CONFIG:-}" ]; then
  COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
fi

SCHED_ARGS=()
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  SCHED_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  SCHED_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  SCHED_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

PROFILER_ARGS=()
if [[ -n "${PROFILER_CONFIG_JSON:-}" ]]; then
  PROFILER_ARGS+=(--profiler-config "${PROFILER_CONFIG_JSON}")
fi

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "ERROR: vLLM not found at ${VLLM_BIN}" >&2
  exit 1
fi

"${VLLM_BIN}" serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size 1 \
  "${SCHED_ARGS[@]}" \
  "${PROFILER_ARGS[@]}" \
  --async-scheduling \
  --no-enable-prefix-caching \
  --distributed-executor-backend mp \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}" \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --moe-backend aiter \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --compilation-config "${COMPILATION_CONFIG}" \
  --profiler-config "${PROFILER_CONFIG}" \
  2>&1 | tee "${OUTDIR}/server.log"

