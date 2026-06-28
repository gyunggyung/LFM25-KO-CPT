#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
RUN_ID_BASE="${RUN_ID_BASE:-$(date -u +%Y%m%d_%H%M%S)_parallel_1gpu}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

mkdir -p "$WORK_DIR/logs/evals"

# Format: gpu:task:limit:num_fewshot
# Empty limit means full task. Empty num_fewshot means lm-eval task default.
DEFAULT_JOBS=(
  "0:mmlu_pro:500:"
  "1:mmlu_pro_law:500:"
  "2:mmlu_pro_economics:500:"
  "3:mmlu_prox_lite_ko:500:"
  "4:arc_challenge:500:"
  "5:gpqa_main_zeroshot:200:"
  "6:gsm8k::5"
  "7:aime24::"
)

if [ "$#" -gt 0 ]; then
  JOBS=("$@")
else
  JOBS=("${DEFAULT_JOBS[@]}")
fi

cd "$WORK_DIR"

for spec in "${JOBS[@]}"; do
  IFS=: read -r gpu task limit fewshot <<<"$spec"
  if [ -z "${gpu:-}" ] || [ -z "${task:-}" ]; then
    echo "bad job spec: $spec" >&2
    exit 2
  fi

  session="lfm2ko_eval_${gpu}_${task}_${RUN_ID_BASE}"
  log="logs/evals/${RUN_ID_BASE}_gpu${gpu}_${task}.log"

  cmd="cd $WORK_DIR && CUDA_VISIBLE_DEVICES=$gpu TP=1 TASK_SET=smoke TASKS=$task LIMIT='$limit' NUM_FEWSHOT='$fewshot' RUN_ID=${RUN_ID_BASE}_gpu${gpu}_${task} BATCH_SIZE=$BATCH_SIZE MAX_BATCH_SIZE=$MAX_BATCH_SIZE GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION bash scripts/run_lfm2_ko_eval_matrix.sh > $log 2>&1"

  tmux new-session -d -s "$session" "$cmd"
  echo "$session -> GPU $gpu task=$task limit=${limit:-full} fewshot=${fewshot:-default} log=$log"
done

echo
tmux list-sessions | grep "lfm2ko_eval_.*${RUN_ID_BASE}" || true
