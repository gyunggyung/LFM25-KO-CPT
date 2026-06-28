#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_cpt}"
SHARD_DIR="${SHARD_DIR:-$DATA_ROOT/datasets/shards_full_lfmstyle_20260627}"
FULL_MIX="${FULL_MIX:-$DATA_ROOT/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl}"
FULL_STATS="${FULL_STATS:-$DATA_ROOT/datasets/ko_cpt_mix_full_lfmstyle_20260627.stats.json}"
PRACTICE_LOG="${PRACTICE_LOG:-$WORK_DIR/logs/20260627_lfm25_8b_ko_cpt_full_seed_practice/train.log}"
PREPROCESS_LOG="${PREPROCESS_LOG:-$WORK_DIR/logs/20260627_full_lfmstyle_parallel/build.log}"
WATCH_LOG="${WATCH_LOG:-$WORK_DIR/logs/20260627_full_lfmstyle_watcher/watch.log}"
FULL_RUN_ID="${FULL_RUN_ID:-20260628_lfm25_8b_ko_cpt_full_lfmstyle}"
FULL_LOG="${FULL_LOG:-$WORK_DIR/logs/$FULL_RUN_ID/train.log}"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-FULL-${FULL_RUN_ID}}"
FINAL_FULL="${FINAL_FULL:-$FULL_OUTPUT_DIR/final_full}"
TOKENIZED_DIR="${TOKENIZED_DIR:-$DATA_ROOT/datasets/tokenized_lfm25_8k_20260628}"

echo "time_utc: $(date -u '+%F %T UTC')"
echo "time_kst: $(TZ=Asia/Seoul date '+%F %T KST')"
echo

echo "[sessions]"
tmux ls 2>/dev/null | grep 'lfm2_ko_cpt' || true
echo

echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo

echo "[active_full_training]"
echo "run_id: $FULL_RUN_ID"
echo "log: $FULL_LOG"
echo "output_dir: $FULL_OUTPUT_DIR"
if [ -d "$FINAL_FULL" ]; then
  echo "final_full: present"
else
  echo "final_full: not_yet"
fi
if [ -d "$FULL_OUTPUT_DIR" ]; then
  echo "output_dir_size: $(du -sh "$FULL_OUTPUT_DIR" 2>/dev/null | awk '{print $1}')"
  echo "checkpoints:"
  find "$FULL_OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '  %f\n' 2>/dev/null | sort -V | tail -8
else
  echo "output_dir_missing"
fi
if [ -s "$FULL_LOG" ]; then
  echo "latest_progress:"
  tr '\r' '\n' < "$FULL_LOG" | grep -E '[0-9]+/[0-9]+' | tail -3 || true
  echo "latest_metrics:"
  grep "{'loss'" "$FULL_LOG" | tail -8 || true
  echo "latest_errors:"
  grep -Ei 'traceback|runtimeerror|cuda out of memory|outofmemory|error' "$FULL_LOG" | tail -5 || echo "none"
else
  echo "full_log_missing_or_empty"
fi
echo

echo "[tokenized_export]"
if [ -d "$TOKENIZED_DIR" ]; then
  echo "tokenized_dir_size: $(du -sh "$TOKENIZED_DIR" 2>/dev/null | awk '{print $1}')"
  find "$TOKENIZED_DIR" -maxdepth 1 -type f | sed -n '1,10p'
else
  echo "tokenized_dir_missing: $TOKENIZED_DIR"
fi
echo

echo "[preprocess]"
if [ -d "$SHARD_DIR" ]; then
  completed_sources="$(find "$SHARD_DIR" -maxdepth 1 -name '*.stats.json' 2>/dev/null | wc -l | tr -d ' ')"
  echo "shard_dir_size: $(du -sh "$SHARD_DIR" 2>/dev/null | awk '{print $1}')"
  echo "completed_sources: ${completed_sources}/10"
  if [ ! -s "$FULL_STATS" ]; then
    echo "rough_eta_kst: 2026-06-28 00:20-00:35 KST for merged full LFM-style corpus"
  else
    echo "rough_eta_kst: full preprocess complete"
  fi
  find "$SHARD_DIR" -maxdepth 1 -name '*.stats.json' -printf '%f\n' 2>/dev/null | sort
else
  echo "shard_dir_missing: $SHARD_DIR"
fi
ls -lh "$FULL_MIX" "$FULL_STATS" 2>/dev/null || true
echo

echo "[historical_practice_tail]"
tail -8 "$PRACTICE_LOG" 2>/dev/null || true
echo

echo "[watcher_tail]"
tail -12 "$WATCH_LOG" 2>/dev/null || true
