#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT=/home/shashen/vllm/vllm
PYTHON=${PYTHON:-${VLLM_ROOT}/.venv/bin/python}
cd /tmp

# MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Pro}
MODEL=${MODEL:-/shared/models/deepseek-ai/DeepSeek-V4-Pro}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8333}
CONC=${CONC:-64}
NUM_FEWSHOT=${NUM_FEWSHOT:-5}
MAX_LENGTH=${MAX_LENGTH:-9472}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
MAX_RETRIES=${MAX_RETRIES:-5}
TIMEOUT=${TIMEOUT:-1800}
API_KEY=${API_KEY:-EMPTY}
TASK_YAML=${TASK_YAML:-/home/shashen/vllm/vllm/perf_test/gsm8k_inferencex.yaml}
OUTDIR=${OUTDIR:-/home/shashen/vllm/vllm/perf_test/gsm8k_lm_eval_inferencex/results_$(date +%Y%m%d_%H%M%S)}
LIMIT=${LIMIT:-}

mkdir -p "${OUTDIR}"

if ! "${PYTHON}" -c "import lm_eval" >/dev/null 2>&1; then
  echo "lm_eval is not installed in .venv."
  echo "Install it with: uv pip install lm-eval"
  exit 2
fi

VLLM_MODULE="$("${PYTHON}" -c 'import vllm; print(vllm.__file__)' 2>/dev/null || true)"
VLLM_VERSION="$("${PYTHON}" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"

{
  echo "MODEL=${MODEL}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"
  echo "CONC=${CONC}"
  echo "NUM_FEWSHOT=${NUM_FEWSHOT}"
  echo "MAX_LENGTH=${MAX_LENGTH}"
  echo "MAX_TOKENS=${MAX_TOKENS}"
  echo "TEMPERATURE=${TEMPERATURE}"
  echo "TOP_P=${TOP_P}"
  echo "MAX_RETRIES=${MAX_RETRIES}"
  echo "TIMEOUT=${TIMEOUT}"
  echo "TASK_YAML=${TASK_YAML}"
  echo "OUTDIR=${OUTDIR}"
  echo "LIMIT=${LIMIT}"
  echo "VLLM_MODULE=${VLLM_MODULE}"
  echo "VLLM_VERSION=${VLLM_VERSION}"
  echo "PYTHON=${PYTHON}"
  echo "GIT_COMMIT=$(git -C "${VLLM_ROOT}" rev-parse HEAD || true)"
  echo "GIT_STATUS=$(git -C "${VLLM_ROOT}" status --short || true)"
} | tee "${OUTDIR}/gsm8k_lm_eval_env.txt"

MODEL_ARGS="model=${MODEL},base_url=http://${HOST}:${PORT}/v1/chat/completions,api_key=${API_KEY},eos_string=</s>,max_retries=${MAX_RETRIES},num_concurrent=${CONC},timeout=${TIMEOUT},tokenized_requests=False,max_length=${MAX_LENGTH}"
GEN_KWARGS="max_tokens=${MAX_TOKENS},temperature=${TEMPERATURE},top_p=${TOP_P}"

CMD=(
  "${PYTHON}" -m lm_eval
  --model local-chat-completions
  --apply_chat_template
  --tasks "${TASK_YAML}"
  --num_fewshot "${NUM_FEWSHOT}"
  --output_path "${OUTDIR}"
  --log_samples
  --model_args "${MODEL_ARGS}"
  --gen_kwargs "${GEN_KWARGS}"
)

if [ -n "${LIMIT}" ]; then
  CMD+=(--limit "${LIMIT}")
fi

printf '%q ' "${CMD[@]}" | tee "${OUTDIR}/command.txt"
echo | tee -a "${OUTDIR}/command.txt"
"${CMD[@]}" 2>&1 | tee "${OUTDIR}/gsm8k_lm_eval.log"

