#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
TRAIN_LOG="${TRAIN_LOG:-$WORK_DIR/logs/20260628_lfm25_8b_ko_cpt_full_lfmstyle/train.log}"
EXPORT_LOG="${EXPORT_LOG:-$WORK_DIR/logs/20260628_tokenized_export.log}"
UPLOAD_LOG="${UPLOAD_LOG:-$WORK_DIR/logs/20260628_tokenized_upload.log}"
OUT_DIR="${OUT_DIR:-/home/work/.data/lfm2_ko_cpt/datasets/tokenized_lfm25_8k_20260628}"

mkdir -p "$WORK_DIR/logs"

echo "watch_start_kst=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$EXPORT_LOG"
while true; do
  if [ -f "$TRAIN_LOG" ] && grep -Eq 'Total optimization steps|Num Epochs|Num examples' "$TRAIN_LOG"; then
    break
  fi
  if [ -f "$TRAIN_LOG" ] && grep -Eq 'Traceback|CUDA out of memory|RuntimeError|Error' "$TRAIN_LOG"; then
    echo "training_error_detected_kst=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$EXPORT_LOG"
    tail -120 "$TRAIN_LOG" | tee -a "$EXPORT_LOG"
    exit 1
  fi
  sleep 30
done

echo "training_steps_detected_kst=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$EXPORT_LOG"
cd "$ROOT_DIR"
nice -n 10 ionice -c2 -n7 "$VENV/bin/python" lfm2_ko_cpt/scripts/export_lfm25_tokenized_dataset.py \
  --out-dir "$OUT_DIR" \
  --overwrite \
  >> "$EXPORT_LOG" 2>&1

echo "tokenized_upload_start_kst=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$UPLOAD_LOG"
HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/python" lfm2_ko_cpt/scripts/upload_cpt_dataset.py \
  --skip-corpus \
  >> "$UPLOAD_LOG" 2>&1
echo "tokenized_upload_done_kst=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$UPLOAD_LOG"
