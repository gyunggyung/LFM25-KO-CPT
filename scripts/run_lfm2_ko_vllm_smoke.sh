#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_cpt}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"

BASE_MODEL="${BASE_MODEL:-LiquidAI/LFM2.5-8B-A1B}"
CPT_MODEL="${CPT_MODEL:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/evals/${RUN_ID}_vllm_smoke}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

if [ ! -x "$VLLM_ENV/bin/python" ]; then
  echo "missing vLLM virtualenv python: $VLLM_ENV/bin/python" >&2
  exit 2
fi
if [ ! -d "$CPT_MODEL" ]; then
  echo "CPT final model is not ready yet: $CPT_MODEL" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR/logs/evals"

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/huggingface/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

VLLM_SITE_PACKAGES="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_CUDA_LIBS="$VLLM_SITE_PACKAGES/nvidia/cuda_runtime/lib:$VLLM_SITE_PACKAGES/nvidia/cublas/lib:$VLLM_SITE_PACKAGES/nvidia/nccl/lib:$VLLM_SITE_PACKAGES/nvidia/nvshmem/lib:$VLLM_SITE_PACKAGES/nvidia/cudnn/lib:$VLLM_SITE_PACKAGES/nvidia/cusparselt/lib"
export LD_LIBRARY_PATH="$VLLM_CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

echo "time_utc: $(date -u '+%F %T UTC')"
echo "time_kst: $(TZ=Asia/Seoul date '+%F %T KST')"
echo "output_dir: $OUTPUT_DIR"
echo

"$VLLM_ENV/bin/python" "$WORK_DIR/scripts/vllm_lfm2_ko_smoke.py" \
  --model "$BASE_MODEL" \
  --model-name "LiquidAI/LFM2.5-8B-A1B" \
  --output "$OUTPUT_DIR/base_smoke.jsonl" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  2>&1 | tee "$WORK_DIR/logs/evals/base_smoke_${RUN_ID}.log"

"$VLLM_ENV/bin/python" "$WORK_DIR/scripts/vllm_lfm2_ko_smoke.py" \
  --model "$CPT_MODEL" \
  --model-name "LFM2.5-8B-A1B-KO-CPT-FULL" \
  --output "$OUTPUT_DIR/cpt_smoke.jsonl" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  2>&1 | tee "$WORK_DIR/logs/evals/cpt_smoke_${RUN_ID}.log"

echo
echo "smoke_done: $OUTPUT_DIR"
