# LFM2.5 KO CPT 현재 상태

최종 갱신: 2026-06-28 11:00 KST
작업 repo: `/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt`
산출물 root: `/home/work/.data/lfm2_ko_cpt`

## 현재 목표

`LiquidAI/LFM2.5-8B-A1B`를 한국어 위키/법률/금융/도구 데이터로 full-parameter CPT해서 `LFM2.5-8B-A1B-KO` 계열 모델을 만든다. 현재 학습은 LoRA가 아니라 full-parameter CPT다.

HF repos:

- Model: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
- Dataset: https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-DATA
- Dataset latest sha at 2026-06-28 00:47 KST: `93e0f1b4f66fd46580b30f7982fedac88291f74a`

## 실행 중인 세션

상태 확인:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
bash lfm2_ko_cpt/scripts/status_lfm2_ko_cpt.sh
```

현재 tmux 세션:

- `lfm2_ko_cpt_full_lfmstyle_20260627`: H200 8GPU full-parameter CPT 본 학습
- `lfm2_ko_cpt_tokenized_export_watcher_20260628`: 학습 loss 확인 뒤 tokenized Parquet export/upload를 낮은 IO 우선순위로 실행하는 watcher

## 전처리 산출물

출력:

- full mix: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl`
- stats: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.stats.json`
- source shards: `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627`
- tokenized export target: `/home/work/.data/lfm2_ko_cpt/datasets/tokenized_lfm25_8k_20260628`

최종 full mix:

- file size: 약 `20G`
- rows after global deduplication: `4,622,971`
- chars: `11,581,567,658`
- estimated tokens: `6,492,697,020`
- raw estimate at effective batch 64 and seq 8192: `12,384`
- actual packed trainer steps observed in this run: `10,196`

소스 mix:

- `kowiki_raw_full_20260524`: 611,403 rows
- `bcai_finance_kor_hrm_20260524`: 1,861,531 rows
- `korean_legal_raw_full_20260523`: 227,687 rows
- `korean_legal_tasks_full_20260524`: 1,383,340 rows
- `korean_admrule_precedent_raw_full_20260524`: 203,477 rows
- `ko_legal_source_agent_sft_20260621`: 5,999 rows
- `ko_legal_rag_agent_sft_round15_v2`: 749 rows
- `current_law_bar_json_answer_sft_20260621`: 2,000 rows
- `lfm25_terminal_toolbench_hrm_turns_v1`: 326,785 rows

`lfm25_terminal_toolbench_hrm_turns_v1`는 영어 terminal/tool 데이터라 전역 한국어 비율 필터에 걸렸었다. `configs/ko_cpt_sources_full_20260627.json`에서 해당 source만 `min_hangul_ratio: 0.0`으로 낮추고, `scripts/build_ko_cpt_mix_parallel.py`에 source별 filter override를 추가해 포함시켰다.

## 현재 GPU 학습 상태

세션: `lfm2_ko_cpt_full_lfmstyle_20260627`

데이터:

```text
/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl
```

출력:

```text
/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle
```

설정:

- `torchrun --nproc_per_node=8`
- `max_seq_length=8192`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- effective batch: `64 sequences/update`
- token/update upper bound: `64 * 8192 = 524,288 tokens`
- `max_steps=-1`
- `num_train_epochs=1`
- `learning_rate=2e-5`
- `save_steps=1000`
- `save_total_limit=4`
- optimizer: `adamw_8bit`
- `gradient_checkpointing=True`

2026-06-28 10:39 KST 기준:

- full corpus tokenization/packing 완료
- actual total step: `10,196`
- completed step: `10,196 / 10,196`
- final epoch: `1.0`
- final logged train loss: `0.712`
- train runtime: 약 `9h 38m`
- train samples/sec: `18.81`
- train steps/sec: `0.294`
- 로그 위치: `lfm2_ko_cpt/logs/20260628_lfm25_8b_ko_cpt_full_lfmstyle/train.log`
- checkpoint: `checkpoint-8000`, `checkpoint-9000`, `checkpoint-10000`, `checkpoint-10196`
- `save_total_limit=4` 때문에 오래된 checkpoint는 자동 pruning됨
- `final_full`: 존재함
- `final_full` 출처: 무결성 검사를 통과한 `checkpoint-10196` inference files
- `final_full/model.safetensors`: `16,936,006,912` bytes, 2,302 tensors로 safetensors open 확인
- 주의: step 완료와 `checkpoint-10196` 저장 후 추가 `trainer.save_model(final_full)` 중 rank 5 SIGSEGV가 발생했고, 최초 `final_full/model.safetensors`는 불완전했다. 해당 파일은 `final_full.corrupt_20260628_1036`로 보존하고, 정상 checkpoint에서 `final_full`을 재구성했다.
- 현재 GPU: 학습 종료 후 비어 있음

속도/예상:

- steady-state 기준 체크포인트 저장 제외 약 `3.3-3.5 sec/step`
- 실제 완료: 2026-06-28 10:36 KST 부근
- HF model upload: 완료
- uploaded model repo: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
- uploaded sha: `0d0ff67c79be8a4acd613302146c5349beaaf44f`
- 다음 병목: lm-eval/vLLM 공식 벤치마크 환경 정리와 실행

## vLLM 평가 준비

평가는 무조건 vLLM으로 한다. 학습은 끝났고 `final_full` 무결성 확인, HF 업로드, vLLM TP=8 smoke까지 완료했다.

준비된 파일:

- `docs/LFM2_KO_VLLM_EVAL_RUNBOOK_20260628.ko.md`
- `scripts/run_lfm2_ko_vllm_lm_eval.sh`
- `scripts/run_lfm2_ko_eval_matrix.sh`
- `scripts/run_lfm2_ko_vllm_smoke.sh`
- `scripts/vllm_lfm2_ko_smoke.py`
- `scripts/summarize_lm_eval_results.py`
- `scripts/upload_full_model.py`

학습 완료 후 실행 순서:

1. 완료: `python scripts/upload_full_model.py`로 model repo에 README와 weights 업로드
2. 완료: `bash scripts/run_lfm2_ko_vllm_smoke.sh`
3. 다음: `.vllm-lfm-cu12`에 lm-eval/datasets를 추가하거나 별도 clean eval venv 구성
4. 다음: `TASK_SET=smoke LIMIT=100 bash scripts/run_lfm2_ko_eval_matrix.sh`
5. 다음: `TASK_SET=korean bash scripts/run_lfm2_ko_eval_matrix.sh`
6. 다음: `TASK_SET=regression bash scripts/run_lfm2_ko_eval_matrix.sh`
7. 결과 요약: `python scripts/summarize_lm_eval_results.py /home/work/.data/lfm2_ko_cpt/evals/<RUN_ID>_vllm_matrix`

vLLM smoke 결과:

- path: `/home/work/.data/lfm2_ko_cpt/evals/20260628_1052_smoke_clean_vllm_smoke`
- base: TP=8 load/generation 성공
- CPT: TP=8 load/generation 성공
- CPT pass: Korean legal, Korean finance, tool-call format, English instruction smoke
- CPT wiki smoke: 답변은 관련 있었지만 literal keyword `요약` 체크가 false
- 중요 환경: `.vllm-lfm-cu12`, `PYTHONPATH=`와 `PYTHONNOUSERSITE=1`로 user-site torch 오염 차단 필요

## Hugging Face Dataset Upload

완료:

```text
https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-DATA
```

업로드된 파일:

- `README.md`
- `metadata/ko_cpt_sources_full_20260627.json`
- `metadata/ko_cpt_mix_full_lfmstyle_20260627.stats.json`
- `metadata/ko_cpt_mix_full_lfmstyle_20260627.stats.json.full_report.json`
- `data/ko_cpt_mix_full_lfmstyle_20260627.jsonl`

다음 업로드:

- `tokenized/tokenized_lfm25_8k_20260628/`
- 생성 스크립트: `scripts/export_lfm25_tokenized_dataset.py`
- 업로드 스크립트: `scripts/upload_cpt_dataset.py`

중요: tokenized export는 별도 CPU/IO 작업이다. 실제 학습 loss가 나온 뒤 `lfm2_ko_cpt_tokenized_export_watcher_20260628` 세션으로 낮은 우선순위(`nice`, `ionice`)에서 실행 중이다. 학습 GPU util이 98-100%로 유지되는지 계속 확인한다.

## 중단과 재개

본 run 중단:

```bash
tmux send-keys -t lfm2_ko_cpt_full_lfmstyle_20260627 C-c
```

checkpoint에서 재개:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
RUN_ID=20260628_lfm25_8b_ko_cpt_full_lfmstyle \
MIX_PATH=/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_SEQ_LENGTH=8192 \
MAX_STEPS=-1 \
NUM_TRAIN_EPOCHS=1 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=4 \
RESUME_FROM_CHECKPOINT=auto \
bash lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_full.sh
```

## 문서와 커밋

현재 repo는 별도 git repo다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
git log --oneline -8
git status --short
```

최근 주요 커밋:

- `b0b277b Add vLLM evaluation workflow for LFM2 KO CPT`
- `3afb965 Document active LFM2 full CPT training`
- `13806b0 Wait for training loss before tokenized export`
- `93299b7 Rebuild tokenized export cleanly`
- `7976431 Tighten tokenized export watcher trigger`
- `c77eac6 Add tokenized export watcher`
- `78480a2 Add packed tokenized dataset export`
- `3f5cf84 Add full CPT dataset upload workflow`
- `ae72d2b Clarify Legalize KR citation guidance`
- `fc21873 Add Legalize KR attribution to model card`
- `fc44cf2 Update full CPT running status`

주요 문서:

- `README.md`
- `docs/LFM2_KO_CPT_RUNBOOK_20260627.ko.md`
- `docs/LFM2_KO_EVAL_PLAN_20260627.ko.md`
- `docs/LFM2_KO_VLLM_EVAL_RUNBOOK_20260628.ko.md`
- `model_cards/LFM2.5-8B-A1B-KO-CPT-FULL.md`
- `dataset_cards/LFM2.5-8B-A1B-KO-CPT-DATA.md`

## 다음 작업

1. `.vllm-lfm-cu12`에 lm-eval/datasets를 추가하거나 별도 clean eval venv를 만든다.
2. `TASK_SET=smoke LIMIT=100 bash scripts/run_lfm2_ko_eval_matrix.sh`로 공개 벤치마크 smoke를 시작한다.
3. tokenized Parquet shard export가 끝나면 `scripts/upload_cpt_dataset.py --skip-corpus`로 dataset repo에 추가 업로드한다.
4. 평가 결과를 요약해 model card benchmark table에 반영한다.
5. 공개 평가 추가 구현 우선순위는 `KMMLU`, `KMMLU-Pro`, `Ko-IFEval`, `Ko-GSM8K`, `Ko-ARC`, 법률 RAG/source QA다.
