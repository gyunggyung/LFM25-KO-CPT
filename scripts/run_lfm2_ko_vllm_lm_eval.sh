#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_cpt}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm}"
LIQUID_ENV="${LIQUID_ENV:-$ROOT_DIR/.liquid-sft-env}"

MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
MODEL_NAME="${MODEL_NAME:-lfm25_8b_a1b_ko_cpt_full}"
TASKS="${TASKS:-ifeval,global_mmlu_full_ko,mmlu_prox_lite_ko}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/evals/$(date -u +%Y%m%dT%H%M%SZ)_vllm_lm_eval}"
TP="${TP:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
LIMIT="${LIMIT:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"

if [ ! -x "$VLLM_ENV/bin/python" ]; then
  echo "missing vLLM virtualenv python: $VLLM_ENV/bin/python" >&2
  exit 2
fi
if [ ! -d "$LIQUID_ENV/lib/python3.12/site-packages" ]; then
  echo "missing Liquid SFT site-packages: $LIQUID_ENV/lib/python3.12/site-packages" >&2
  exit 2
fi
if [ ! -e "$MODEL_PATH" ] && [[ "$MODEL_PATH" == /* ]]; then
  echo "missing model path: $MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR/logs/evals"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/huggingface/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

# vLLM in .vllm-lfm is built against the CUDA 13 runtime wheels. The dynamic
# loader does not always see those wheel libraries unless we expose them here.
VLLM_SITE_PACKAGES="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_CUDA_LIBS="$VLLM_SITE_PACKAGES/nvidia/cu13/lib:$VLLM_SITE_PACKAGES/nvidia/nccl/lib:$VLLM_SITE_PACKAGES/nvidia/nvshmem/lib:$VLLM_SITE_PACKAGES/nvidia/cudnn/lib:$VLLM_SITE_PACKAGES/nvidia/cusparselt/lib"
export LD_LIBRARY_PATH="$VLLM_CUDA_LIBS:${LD_LIBRARY_PATH:-}"

# lm_eval is installed in the user/Unsloth environment while vLLM is installed
# in .vllm-lfm. Keep this mixed PYTHONPATH until the eval environment is rebuilt.
export PYTHONPATH="$LIQUID_ENV/lib/python3.12/site-packages:/home/work/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

MODEL_ARGS="pretrained=$MODEL_PATH,tensor_parallel_size=$TP,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,max_model_len=$MAX_MODEL_LEN"

CMD=(
  "$VLLM_ENV/bin/python" -m lm_eval run
  --model vllm
  --model_args "$MODEL_ARGS"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --max_batch_size "$MAX_BATCH_SIZE"
  --output_path "$OUTPUT_DIR/$MODEL_NAME"
  --trust_remote_code
  --confirm_run_unsafe_code
  --metadata "model_name=$MODEL_NAME" "model_path=$MODEL_PATH" "tp=$TP" "max_model_len=$MAX_MODEL_LEN"
)

if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi
if [ -n "$NUM_FEWSHOT" ]; then
  CMD+=(--num_fewshot "$NUM_FEWSHOT")
fi
if [ "$APPLY_CHAT_TEMPLATE" = "1" ]; then
  CMD+=(--apply_chat_template)
fi
if [ "$LOG_SAMPLES" = "1" ]; then
  CMD+=(--log_samples)
fi

echo "time_utc: $(date -u '+%F %T UTC')"
echo "time_kst: $(TZ=Asia/Seoul date '+%F %T KST')"
echo "model_name: $MODEL_NAME"
echo "model_path: $MODEL_PATH"
echo "tasks: $TASKS"
echo "output_dir: $OUTPUT_DIR/$MODEL_NAME"
echo "tensor_parallel_size: $TP"
echo "max_model_len: $MAX_MODEL_LEN"
echo "batch_size: $BATCH_SIZE"
echo
printf '%q ' "${CMD[@]}"
echo
echo

cd "$WORK_DIR"
"${CMD[@]}" 2>&1 | tee "$WORK_DIR/logs/evals/${MODEL_NAME}_lm_eval_$(date -u +%Y%m%dT%H%M%SZ).log"
