# LFM2.5-8B-A1B-KO-CPT-FULL

Full-parameter Korean continued pretraining project for
[`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B).

한국어 법률, 금융, 위키 지식, 터미널/tool-use 데이터로 `LFM2.5-8B-A1B`를 full CPT한 프로젝트다.
학습은 완료됐고, vLLM 기반 base-vs-CPT 평가와 Hugging Face 업로드까지 진행 중이다.

## Links

- Model: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
- Dataset: https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-DATA
- Current status / handoff: [docs/CURRENT_STATUS_20260627.ko.md](docs/CURRENT_STATUS_20260627.ko.md)
- CPT runbook: [docs/LFM2_KO_CPT_RUNBOOK_20260627.ko.md](docs/LFM2_KO_CPT_RUNBOOK_20260627.ko.md)
- vLLM eval runbook: [docs/LFM2_KO_VLLM_EVAL_RUNBOOK_20260628.ko.md](docs/LFM2_KO_VLLM_EVAL_RUNBOOK_20260628.ko.md)
- Evaluation plan: [docs/LFM2_KO_EVAL_PLAN_20260627.ko.md](docs/LFM2_KO_EVAL_PLAN_20260627.ko.md)
- Model card source: [model_cards/LFM2.5-8B-A1B-KO-CPT-FULL.md](model_cards/LFM2.5-8B-A1B-KO-CPT-FULL.md)
- Dataset card source: [dataset_cards/LFM2.5-8B-A1B-KO-CPT-DATA.md](dataset_cards/LFM2.5-8B-A1B-KO-CPT-DATA.md)

## Status

- Base model: `LiquidAI/LFM2.5-8B-A1B`
- Final model path: `/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full`
- Training: full-parameter CPT, 1 epoch, `10196/10196` steps, final train loss `0.712`
- Training hardware: 8x H200, `torchrun --nproc_per_node=8`
- Eval runtime: vLLM `TP=8` for large single runs, `TP=1` one task per GPU for parallel benchmark sweeps
- Current artifact root: `/home/work/.data/lfm2_ko_cpt`

## English Summary

This project adapts LiquidAI LFM2.5-8B-A1B to Korean through full-parameter continued pretraining. The data mixture focuses on Korean legal text, finance text, wiki-style knowledge, terminal/tool-call behavior, and general instruction-preserving text. Evaluation is run with vLLM and lm-evaluation-harness against the original base model.

The strongest confirmed gains so far are on GSM8K, Korean Global MMLU subject slices, BoolQ, ARC-Challenge, and instruction-following. Some law-oriented MMLU-Pro and MMLU-ProX-lite-ko checks regress, so the next data iteration should add targeted Korean/legal multiple-choice remediation.

## 한국어 요약

이 작업은 LiquidAI `LFM2.5-8B-A1B`에 한국어 지식을 이식하기 위한 full CPT 실험이다. 데이터는 한국어 법률, 금융, 위키, 터미널 기록, tool-call 형식, 일반 지시문 데이터 중심으로 구성했다. 평가는 원본 base 모델과 CPT 모델을 vLLM으로 같은 조건에서 비교한다.

현재까지 GSM8K, Global MMLU Korean 일부 과목, BoolQ, ARC-Challenge, IFEval은 상승했다. 반대로 MMLU-Pro law와 MMLU-ProX-lite-ko는 하락했으므로, 다음 학습에서는 한국어/법률 객관식 풀이 형식과 선택지 추출 안정성을 보강해야 한다.

## Key Results

All numbers below are vLLM/lm-eval base-vs-CPT comparisons.

| Benchmark | Metric | Base | CPT | Delta | Relative | Note |
|---|---|---:|---:|---:|---:|---|
| IFEval full | prompt loose | 0.2921 | 0.3216 | +0.0295 | +10.10% | full task |
| IFEval full | instruction loose | 0.4341 | 0.4628 | +0.0287 | +6.61% | full task |
| GSM8K full 5-shot | exact_match flexible | 0.4845 | 0.5701 | +0.0856 | +17.67% | full task |
| GSM8K full 5-shot | exact_match strict | 0.2472 | 0.4617 | +0.2145 | +86.77% | full task |
| Global MMLU Korean `LIMIT=500` | acc | 0.2803 | 0.3086 | +0.0283 | +10.10% | limited |
| ARC-Challenge `LIMIT=500` | acc_norm | 0.3760 | 0.4140 | +0.0380 | +10.11% | limited |
| BoolQ full | acc | 0.6544 | 0.7902 | +0.1358 | +20.75% | full task |
| HellaSwag `LIMIT=1000` | acc_norm | 0.4330 | 0.5110 | +0.0780 | +18.01% | limited |
| PIQA full | acc_norm | 0.7209 | 0.7465 | +0.0256 | +3.55% | full task |
| WinoGrande full | acc | 0.5643 | 0.5699 | +0.0055 | +0.98% | full task |
| Global MMLU KO management full | acc | 0.3107 | 0.4369 | +0.1262 | +40.63% | full subject |
| Global MMLU KO world religions full | acc | 0.3450 | 0.4854 | +0.1404 | +40.70% | full subject |
| Global MMLU KO international law full | acc | 0.3223 | 0.4215 | +0.0992 | +30.77% | full subject |
| Global MMLU KO virology full | acc | 0.2831 | 0.3795 | +0.0964 | +34.05% | full subject |
| MMLU-Pro economics `LIMIT=500` | exact_match | 0.4420 | 0.4900 | +0.0480 | +10.86% | limited |
| MMLU-Pro law `LIMIT=500` | exact_match | 0.1840 | 0.1240 | -0.0600 | -32.61% | limited |
| MMLU-ProX Lite KO `LIMIT=500` | exact_match | 0.2585 | 0.1667 | -0.0918 | -35.51% | limited |

## Running Evaluation

Single matrix evaluation:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
TASKS=ifeval RUN_ID=ifeval_full bash scripts/run_lfm2_ko_eval_matrix.sh
```

One task per GPU parallel sweep:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
RUN_ID_BASE=parallel_1gpu bash scripts/run_lfm2_ko_parallel_eval_1gpu.sh
```

Custom one-GPU job specs use `gpu:task:limit:num_fewshot`:

```bash
RUN_ID_BASE=kmmlu_math bash scripts/run_lfm2_ko_parallel_eval_1gpu.sh \
  '1:kmmlu_direct:1000:' \
  '2:kmmlu_hard:1000:' \
  '3:leaderboard_math_hard:500:' \
  '4:hendrycks_math:500:'
```

Summarize an eval directory:

```bash
python scripts/summarize_lm_eval_results.py /home/work/.data/lfm2_ko_cpt/evals/<RUN_ID>_vllm_matrix
```

## Important Files

- Data sources: [configs/ko_cpt_sources_20260627.json](configs/ko_cpt_sources_20260627.json)
- Data builder: [scripts/build_ko_cpt_mix.py](scripts/build_ko_cpt_mix.py)
- Parallel data builder: [scripts/build_ko_cpt_mix_parallel.py](scripts/build_ko_cpt_mix_parallel.py)
- Full trainer: [scripts/train_lfm25_ko_cpt_full.py](scripts/train_lfm25_ko_cpt_full.py)
- Full training runner: [scripts/run_lfm25_8b_ko_cpt_full.sh](scripts/run_lfm25_8b_ko_cpt_full.sh)
- vLLM eval runner: [scripts/run_lfm2_ko_vllm_lm_eval.sh](scripts/run_lfm2_ko_vllm_lm_eval.sh)
- Base-vs-CPT matrix runner: [scripts/run_lfm2_ko_eval_matrix.sh](scripts/run_lfm2_ko_eval_matrix.sh)
- Parallel eval runner: [scripts/run_lfm2_ko_parallel_eval_1gpu.sh](scripts/run_lfm2_ko_parallel_eval_1gpu.sh)
- Model upload: [scripts/upload_full_model.py](scripts/upload_full_model.py)
- Dataset upload: [scripts/upload_cpt_dataset.py](scripts/upload_cpt_dataset.py)

## Next Evaluation Queue

- Korean language specialization: `kmmlu`, `kmmlu_direct`, `kmmlu_hard`, `kmmlu_cot_hard`
- Liquid official-style checks: `leaderboard_instruction_following`, `leaderboard_math_hard`, `hendrycks_math`
- Tool/agent checks needing separate harness: BFCLv3/v4, Tau2 Telecom/Retail, IFBench, Multi-IF
- Korean remediation checks: legal multiple-choice, Korean option extraction, tool-call JSON validity

Quick status:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
bash lfm2_ko_cpt/scripts/status_lfm2_ko_cpt.sh
```
