#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_cpt}"
RUN_ID="${RUN_ID:-20260627_lfm25_8b_ko_cpt_lora}"
MODEL_PATH="${MODEL_PATH:-LiquidAI/LFM2.5-8B-A1B}"
CONFIG_PATH="${CONFIG_PATH:-$WORK_DIR/configs/ko_cpt_sources_20260627.json}"
MIX_PATH="${MIX_PATH:-$DATA_ROOT/datasets/ko_cpt_mix_seed_20260627.jsonl}"
STATS_PATH="${STATS_PATH:-$DATA_ROOT/datasets/ko_cpt_mix_seed_20260627.stats.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-LoRA-$RUN_ID}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs/$RUN_ID}"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"

mkdir -p "$DATA_ROOT/datasets" "$DATA_ROOT/models" "$LOG_DIR"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/huggingface/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$ROOT_DIR/.env" ]; then
  HF_TOKEN_LINE="$(awk -F= '/^HF_TOKEN=/{print substr($0, index($0, "=") + 1)}' "$ROOT_DIR/.env" | tail -n 1)"
  if [ -n "$HF_TOKEN_LINE" ]; then
    export HF_TOKEN="$HF_TOKEN_LINE"
  fi
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "missing virtualenv: $VENV" >&2
  exit 2
fi

source "$VENV/bin/activate"
cd "$WORK_DIR"

if [ ! -s "$MIX_PATH" ]; then
  python scripts/build_ko_cpt_mix.py \
    --config "$CONFIG_PATH" \
    --output "$MIX_PATH" \
    --stats-output "$STATS_PATH" \
    --max-docs-total "${MAX_DOCS_TOTAL:-195000}" \
    --min-chars "${MIN_CHARS:-160}" \
    --max-chars "${MAX_CHARS:-24000}" \
    2>&1 | tee "$LOG_DIR/build_dataset.log"
fi

RESUME_ARG=()
if [ "${RESUME_FROM_CHECKPOINT:-auto}" = "auto" ]; then
  LATEST_CHECKPOINT="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$LATEST_CHECKPOINT" ]; then
    RESUME_ARG=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
  fi
elif [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  RESUME_ARG=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

PUSH_ARGS=()
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
  PUSH_ARGS=(--push-to-hub --hub-model-id "${HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-LoRA}")
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  scripts/train_lfm25_ko_cpt_lora.py \
    --model-path "$MODEL_PATH" \
    --dataset-path "$MIX_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length "${MAX_SEQ_LENGTH:-8192}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
    --learning-rate "${LEARNING_RATE:-0.0001}" \
    --max-steps "${MAX_STEPS:-3000}" \
    --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --save-steps "${SAVE_STEPS:-100}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT:-5}" \
    --logging-steps "${LOGGING_STEPS:-1}" \
    --dataset-num-proc "${DATASET_NUM_PROC:-24}" \
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-8}" \
    --lora-rank "${LORA_RANK:-64}" \
    --lora-alpha "${LORA_ALPHA:-128}" \
    --target-modules "${TARGET_MODULES:-all-linear}" \
    "${RESUME_ARG[@]}" \
    "${PUSH_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/train.log"
