# LFM2.5 KO CPT 현재 상태

최종 갱신: 2026-06-27 23:54 KST  
작업 repo: `/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt`  
산출물 root: `/home/work/.data/lfm2_ko_cpt`

## 현재 목표

`LiquidAI/LFM2.5-8B-A1B`를 한국어 위키/법률/금융/도구 데이터로 full-parameter CPT해서 `LFM2.5-8B-A1B-KO` 계열 모델을 만든다. 현재는 full corpus 전처리가 최우선이고, GPU에서는 seed corpus로 full FT 설정을 검증하는 practice run이 돌고 있다.

## 실행 중인 세션

상태 확인:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
bash lfm2_ko_cpt/scripts/status_lfm2_ko_cpt.sh
```

현재 tmux 세션:

- `lfm2_ko_cpt_full_lfmstyle_parallel_20260627`: CPU 병렬 full corpus 전처리
- `lfm2_ko_cpt_full_seed_practice_20260627`: H200 8GPU full FT practice run
- `lfm2_ko_cpt_full_lfmstyle_watcher_20260627`: full corpus 생성 및 practice 종료 후 본 run 자동 시작

## 전처리 상태

출력:

- full mix: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl`
- stats: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.stats.json`
- shards: `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627`

2026-06-27 23:53 KST 기준:

- shard size: 약 `19G`
- 완료 소스: `7/10`
- 예상 full merge 완료: 2026-06-28 00:20-00:35 KST

완료된 소스:

- `kowiki_raw_full_20260524`: 611,403 rows, 922,602,857 chars
- `korean_legal_raw_full_20260523`: 227,687 rows, 789,324,880 chars
- `korean_legal_tasks_full_20260524`: 1,383,340 rows, 1,813,366,700 chars
- `korean_admrule_precedent_raw_full_20260524`: 203,477 rows, 719,804,895 chars
- `ko_legal_source_agent_sft_20260621`: 5,999 rows, 60,652,338 chars
- `ko_legal_rag_agent_sft_round15_v2`: 749 rows, 7,656,157 chars
- `current_law_bar_json_answer_sft_20260621`: 2,000 rows, 4,693,801 chars

남은 큰 소스는 금융 2개와 terminal/toolbench 계열이다.

## 현재 GPU 학습 상태

세션: `lfm2_ko_cpt_full_seed_practice_20260627`

Practice run 목적:

- full-parameter CPT가 실제로 8GPU에서 도는지 확인
- batch/VRAM/step speed/checkpoint 저장 확인
- full corpus가 완성되기 전 GPU 공백을 줄이는 연습 run

데이터:

```text
/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_seed_20260627.jsonl
```

출력:

```text
/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260627_lfm25_8b_ko_cpt_full_seed_practice
```

설정:

- `torchrun --nproc_per_node=8`
- `max_seq_length=8192`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- effective batch: `64 sequences/update`
- 최대 token/update: 약 `64 * 8192 = 524,288 tokens`
- `max_steps=500`
- `learning_rate=2e-5`
- `save_steps=25`
- `save_total_limit=4`

2026-06-27 23:53 KST 기준 GPU:

- H200 8장 모두 사용 중
- VRAM: 각 GPU 약 `85-87GB / 143GB`
- utilization: 대부분 `99-100%`

속도:

- 체크포인트 저장 구간 제외: 약 `3.33-3.45 sec/step`
- 25 step마다 model shard 저장 때문에 약 37-40초 추가 지연
- checkpoint 저장 후 다시 3.3초대 step으로 복귀

## Watcher 동작

세션: `lfm2_ko_cpt_full_lfmstyle_watcher_20260627`

역할:

1. full LFM-style mix와 stats 파일이 생길 때까지 대기한다.
2. practice run이 종료될 때까지 대기한다.
3. 본 run 세션 `lfm2_ko_cpt_full_lfmstyle_20260627`을 시작한다.

본 run 기본값:

- mix: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl`
- output run id: `20260627_lfm25_8b_ko_cpt_full_lfmstyle`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=4`
- `max_seq_length=8192`
- `max_steps=8000`
- `save_steps=100`
- `learning_rate=2e-5`

수동 시작이 필요하면:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
tmux new-session -d -s lfm2_ko_cpt_full_lfmstyle_20260627 \
  'RUN_ID=20260627_lfm25_8b_ko_cpt_full_lfmstyle MIX_PATH=/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 MAX_SEQ_LENGTH=8192 MAX_STEPS=8000 SAVE_STEPS=100 LEARNING_RATE=0.00002 bash lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_full.sh'
```

## 중단과 재개

학습 중단:

```bash
tmux send-keys -t lfm2_ko_cpt_full_seed_practice_20260627 C-c
```

본 run 중단:

```bash
tmux send-keys -t lfm2_ko_cpt_full_lfmstyle_20260627 C-c
```

checkpoint에서 재개:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
RESUME_FROM_CHECKPOINT=auto bash lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_full.sh
```

## 문서와 커밋

현재 repo는 별도 git repo다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
git log --oneline -5
git status --short
```

이미 들어간 주요 커밋:

- `fa1d20c Add LFM2 Korean CPT workflow`
- `0efd4a7 Add full CPT training and parallel preprocessing workflow`

주요 문서:

- `README.md`
- `docs/LFM2_KO_CPT_RUNBOOK_20260627.ko.md`
- `docs/LFM2_KO_EVAL_PLAN_20260627.ko.md`
- `model_cards/LFM2.5-8B-A1B-KO-CPT-LoRA.md`

## 다음 작업

1. full corpus 전처리 완료를 확인한다.
2. practice run이 checkpoint 저장과 종료까지 정상 진행되는지 확인한다.
3. watcher가 본 run을 시작했는지 확인한다.
4. 본 run이 시작되면 step speed와 VRAM을 보고 batch를 조정한다. 현재 기준은 안정성 우선으로 `per_device=2`, `grad_accum=4`다.
5. full CPT checkpoint가 생기면 평가 준비를 시작한다.
6. 공개 평가 우선순위는 `KMMLU`, `KMMLU-Pro`, `KMMLU-Redux`, `Ko-IFEval`, `Ko-GSM8K`, `Ko-ARC`다.
