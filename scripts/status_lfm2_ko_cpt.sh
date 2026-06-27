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

echo "time_utc: $(date -u '+%F %T UTC')"
echo "time_kst: $(TZ=Asia/Seoul date '+%F %T KST')"
echo

echo "[sessions]"
tmux ls 2>/dev/null | grep 'lfm2_ko_cpt' || true
echo

echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
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
tail -20 "$PREPROCESS_LOG" 2>/dev/null || true
echo

echo "[training_practice_tail]"
tail -40 "$PRACTICE_LOG" 2>/dev/null || true
echo

echo "[watcher_tail]"
tail -40 "$WATCH_LOG" 2>/dev/null || true
