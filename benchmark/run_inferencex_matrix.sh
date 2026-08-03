#!/usr/bin/env bash
# 8k/1k perf sweep — client aligned with InferenceX dsv4_fp4_mi355x_vllm.sh
# Benchmark client runs from /tmp so it does not pick up the mounted vLLM repo.
set -euo pipefail

unset PYTHONPATH VIRTUAL_ENV
cd /tmp

IX=${IX:-/home/shashen/vllm/InferenceX}
BENCH="${IX}/utils/bench_serving/benchmark_serving.py"
export MODEL=${MODEL:-/shared/models/deepseek-ai/DeepSeek-V4-Pro}
export PORT=${PORT:-8333}
export ISL=${ISL:-128}
export OSL=${OSL:-1024}
# export RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}
export RANDOM_RANGE_RATIO=0
# export PROMPTS_MULT=${PROMPTS_MULT:-10}
export PROMPTS_MULT=1
# CONCS=${CONCS:-"4 8 16 32 64"}
# CONCS=${CONCS:-"64"}
# export CONCS=$1
OUTDIR=${OUTDIR:-/home/shashen/vllm/vllm/benchmark/ix_8k1k/results_$(date +%Y%m%d_%H%M%S)}
mkdir -p "${OUTDIR}"

PYTHON="${PYTHON:-/home/shashen/vllm/vllm/.venv/bin/python}"

DSV4_ARGS=()
if [[ "${USE_DSV4_TEMPLATE:-0}" == "1" ]]; then
  DSV4_ARGS=(--use-chat-template --dsv4)
fi

VLLM_MODULE="$("${PYTHON}" -c 'import vllm; print(vllm.__file__)' 2>/dev/null || true)"

{
  echo "MODEL=${MODEL}"; echo "PORT=${PORT}"
  echo "ISL=${ISL}"; echo "OSL=${OSL}"; echo "RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}"
  echo "PROMPTS_MULT=${PROMPTS_MULT}"; echo "CONCS=${CONCS}"
  echo "USE_DSV4_TEMPLATE=${USE_DSV4_TEMPLATE:-0}"
  echo "BENCH=${BENCH}"; echo "OUTDIR=${OUTDIR}"; echo "PYTHON=${PYTHON}"
  echo "VLLM_MODULE=${VLLM_MODULE}"
  echo "CWD=$(pwd)"
} | tee "${OUTDIR}/bench_env.txt"

for CONC in ${CONCS}; do
  NUM_PROMPTS=$((CONC * PROMPTS_MULT))
  NUM_WARMUPS=$((CONC * 2))
  RESULT_FILENAME="dsv4_8k1k_vllm_tp8_conc${CONC}_local.json"
  echo "InferenceX 8k1k bench: conc=${CONC} num_prompts=${NUM_PROMPTS} warmups=${NUM_WARMUPS} range_ratio=${RANDOM_RANGE_RATIO}"
  "${PYTHON}" "${BENCH}" \
    --model "${MODEL}" \
    --backend vllm \
    --base-url "http://0.0.0.0:${PORT}" \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONC}" \
    --num-warmups "${NUM_WARMUPS}" \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --trust-remote-code \
    "${DSV4_ARGS[@]}" \
    --save-result \
    --result-dir "${OUTDIR}" \
    --result-filename "${RESULT_FILENAME}" \
    --profile \
    2>&1 | tee "${OUTDIR}/8k1k_conc${CONC}.log"
done

