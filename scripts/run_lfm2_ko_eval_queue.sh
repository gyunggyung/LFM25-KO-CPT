#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_FILE="${1:-$WORK_DIR/configs/eval_queue_global_mmlu_ko_refill_20260628.txt}"
STATE_FILE="${STATE_FILE:-$QUEUE_FILE.state}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-10000}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-$(date -u +%Y%m%d_%H%M%S)_queue}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

if [ ! -f "$QUEUE_FILE" ]; then
  echo "queue file not found: $QUEUE_FILE" >&2
  exit 2
fi

mkdir -p "$WORK_DIR/logs/evals"

next_index() {
  if [ -f "$STATE_FILE" ]; then
    cat "$STATE_FILE"
  else
    echo 1
  fi
}

queue_line() {
  local index="$1"
  awk 'NF && $0 !~ /^#/ { n += 1; if (n == idx) { print; exit } }' idx="$index" "$QUEUE_FILE"
}

gpu_has_session() {
  local gpu="$1"
  tmux list-sessions -F '#S' 2>/dev/null | grep -q "^lfm2ko_eval_${gpu}_"
}

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="$IDLE_MEMORY_MIB" '{ g=$1+0; m=$2+0; if (m < limit) print g }'
}

launch_one() {
  local gpu="$1"
  local spec="$2"
  local task limit fewshot
  IFS=: read -r task limit fewshot <<<"$spec"
  if [ -z "${task:-}" ]; then
    return 1
  fi

  local run_id="${RUN_ID_PREFIX}_gpu${gpu}_${task}"
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
  BATCH_SIZE="$BATCH_SIZE" \
  RUN_ID_BASE="$run_id" \
    bash "$WORK_DIR/scripts/run_lfm2_ko_parallel_eval_1gpu.sh" "${gpu}:${task}:${limit:-}:${fewshot:-}"
}

echo "queue: $QUEUE_FILE"
echo "state: $STATE_FILE"
echo "poll_seconds: $POLL_SECONDS"
echo "idle_memory_mib: $IDLE_MEMORY_MIB"

while true; do
  idx="$(next_index)"
  spec="$(queue_line "$idx")"
  if [ -z "$spec" ]; then
    echo "queue exhausted at index $idx"
    exit 0
  fi

  launched=0
  while read -r gpu; do
    [ -n "$gpu" ] || continue
    if gpu_has_session "$gpu"; then
      continue
    fi
    spec="$(queue_line "$idx")"
    if [ -z "$spec" ]; then
      echo "queue exhausted at index $idx"
      exit 0
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launching gpu=$gpu spec=$spec index=$idx"
    launch_one "$gpu" "$spec"
    idx=$((idx + 1))
    echo "$idx" > "$STATE_FILE"
    launched=1
  done < <(idle_gpus)

  if [ "$launched" -eq 0 ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) no idle GPU; next_index=$idx"
  fi
  sleep "$POLL_SECONDS"
done
