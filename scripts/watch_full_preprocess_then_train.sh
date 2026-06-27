#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_cpt}"
FULL_MIX="${FULL_MIX:-$DATA_ROOT/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl}"
FULL_STATS="${FULL_STATS:-$DATA_ROOT/datasets/ko_cpt_mix_full_lfmstyle_20260627.stats.json}"
SESSION_NAME="${SESSION_NAME:-lfm2_ko_cpt_full_lfmstyle_20260627}"
PRACTICE_SESSION="${PRACTICE_SESSION:-lfm2_ko_cpt_full_seed_practice_20260627}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/lfm2_ko_cpt/logs/20260627_full_lfmstyle_watcher}"

mkdir -p "$LOG_DIR"

echo "watcher_start $(date -u '+%F %T UTC')"
while [ ! -s "$FULL_STATS" ] || [ ! -s "$FULL_MIX" ]; do
  ls -lh "$FULL_MIX" "$FULL_STATS" 2>/dev/null || true
  sleep 120
done

echo "full_preprocess_ready $(date -u '+%F %T UTC')"
cat "$FULL_STATS" || true

while tmux has-session -t "$PRACTICE_SESSION" 2>/dev/null; do
  echo "waiting_practice_session $PRACTICE_SESSION $(date -u '+%F %T UTC')"
  sleep 120
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "full_training_session_already_exists $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$ROOT_DIR' && RUN_ID=20260627_lfm25_8b_ko_cpt_full_lfmstyle MIX_PATH='$FULL_MIX' PER_DEVICE_TRAIN_BATCH_SIZE=\${PER_DEVICE_TRAIN_BATCH_SIZE:-2} GRADIENT_ACCUMULATION_STEPS=\${GRADIENT_ACCUMULATION_STEPS:-4} MAX_SEQ_LENGTH=\${MAX_SEQ_LENGTH:-8192} MAX_STEPS=\${MAX_STEPS:-8000} SAVE_STEPS=\${SAVE_STEPS:-100} LEARNING_RATE=\${LEARNING_RATE:-0.00002} bash lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_full.sh"

echo "started_full_training_session $SESSION_NAME $(date -u '+%F %T UTC')"
