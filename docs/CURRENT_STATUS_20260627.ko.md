# LFM2.5 KO CPT 현재 상태

최종 갱신: 2026-06-28 01:00 KST
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
- estimated training steps at effective batch 64 and seq 8192: `12,384`

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

2026-06-28 01:00 KST 기준:

- full corpus tokenization 완료
- packing 완료 후 실제 optimizer step 진입
- 로그 위치: `lfm2_ko_cpt/logs/20260628_lfm25_8b_ko_cpt_full_lfmstyle/train.log`
- loss 로그 확인: 약 37 step, 최근 loss 약 `1.92`
- GPU VRAM: 각 GPU 약 `82-84GB / 143GB`
- GPU util: 각 GPU 약 `98-100%`
- 첫 checkpoint는 `save_steps=1000` 기준 checkpoint-1000에서 생성 예정

속도/예상:

- practice 기준 체크포인트 저장 제외 약 `3.35-3.45 sec/step`
- packed dataset 기준 실제 1 epoch 총 step은 약 `10.2k`로 관측됨
- 첫 37 step 구간 기준 step 속도는 추후 100+ step에서 재측정
- full 1 epoch 예상 runtime은 실제 step 속도 안정화 후 갱신

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
- `model_cards/LFM2.5-8B-A1B-KO-CPT-FULL.md`
- `dataset_cards/LFM2.5-8B-A1B-KO-CPT-DATA.md`

## 다음 작업

1. 100+ step 구간에서 실제 sec/step과 ETA를 갱신한다.
2. checkpoint-1000 생성 여부를 확인한다.
3. tokenized Parquet shard export가 끝나면 `scripts/upload_cpt_dataset.py --skip-corpus`로 dataset repo에 추가 업로드한다.
4. checkpoint-1000 생성 시 모델 card에 intermediate checkpoint 상태를 반영한다.
5. GPU util이 떨어지거나 OOM/RuntimeError가 나오면 즉시 로그와 checkpoint 상태를 확인한다.
6. 공개 평가 우선순위는 `KMMLU`, `KMMLU-Pro`, `Ko-IFEval`, `Ko-GSM8K`, `Ko-ARC`, 법률 RAG/source QA다.
