#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_cpt}"

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
CPT_MODEL="${CPT_MODEL:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/evals/${RUN_ID}_vllm_matrix}"
TASK_SET="${TASK_SET:-smoke}"
LIMIT="${LIMIT:-}"

case "$TASK_SET" in
  smoke)
    TASKS="${TASKS:-ifeval,global_mmlu_full_ko,mmlu_prox_lite_ko}"
    ;;
  korean)
    TASKS="${TASKS:-global_mmlu_full_ko,mmlu_prox_lite_ko,global_mmlu_full_ko_professional_law,global_mmlu_full_ko_professional_accounting}"
    ;;
  regression)
    TASKS="${TASKS:-ifeval,mmlu_pro,gpqa_main_zeroshot,gsm8k,arc_challenge,aime24}"
    ;;
  core)
    TASKS="${TASKS:-ifeval,global_mmlu_full_ko,mmlu_prox_lite_ko,mmlu_pro,mmlu_pro_law,mmlu_pro_economics,gpqa_main_zeroshot,gsm8k,arc_challenge}"
    ;;
  full)
    TASKS="${TASKS:-ifeval,global_mmlu_full_ko,mmlu_prox_lite_ko,global_mmlu_full_ko_professional_law,global_mmlu_full_ko_professional_accounting,mmlu_pro,mmlu_pro_law,mmlu_pro_economics,gpqa_main_zeroshot,gpqa_diamond_zeroshot,gsm8k,arc_challenge,aime24,aime25}"
    ;;
  *)
    echo "unknown TASK_SET=$TASK_SET. Use smoke, korean, regression, core, or full." >&2
    exit 2
    ;;
esac

if [ ! -d "$CPT_MODEL" ]; then
  echo "CPT final model is not ready yet: $CPT_MODEL" >&2
  echo "Finish training and final_full save first." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "time_utc: $(date -u '+%F %T UTC')"
echo "time_kst: $(TZ=Asia/Seoul date '+%F %T KST')"
echo "task_set: $TASK_SET"
echo "tasks: $TASKS"
echo "output_dir: $OUTPUT_DIR"
echo "base_model: $BASE_MODEL"
echo "cpt_model: $CPT_MODEL"
echo

MODEL_PATH="$BASE_MODEL" \
MODEL_NAME="LiquidAI_LFM2.5-8B-A1B_base" \
TASKS="$TASKS" \
OUTPUT_DIR="$OUTPUT_DIR" \
LIMIT="$LIMIT" \
bash "$WORK_DIR/scripts/run_lfm2_ko_vllm_lm_eval.sh"

MODEL_PATH="$CPT_MODEL" \
MODEL_NAME="LLM-OS-Models_LFM2.5-8B-A1B-KO-CPT-FULL" \
TASKS="$TASKS" \
OUTPUT_DIR="$OUTPUT_DIR" \
LIMIT="$LIMIT" \
bash "$WORK_DIR/scripts/run_lfm2_ko_vllm_lm_eval.sh"

echo
echo "matrix_done: $OUTPUT_DIR"
