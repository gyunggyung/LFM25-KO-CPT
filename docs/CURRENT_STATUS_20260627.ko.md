# LFM2.5 KO CPT 현재 상태

최종 갱신: 2026-06-28 11:46 KST
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
- vLLM/lm-eval 환경 보강: 완료
- 공식 IFEval 전체 541문항 base-vs-CPT 평가: 완료
- GSM8K 5-shot `LIMIT=200` 회귀 평가: 완료, CPT 상승
- Global MMLU Korean `LIMIT=500`: 완료, CPT 전체 점수 상승
- 추가 vLLM 평가: 8개 GPU에 1개 작업씩 병렬 진행 중

## vLLM 평가 준비

평가는 무조건 vLLM으로 한다. 학습은 끝났고 `final_full` 무결성 확인, HF 업로드, vLLM TP=8 smoke까지 완료했다.

준비된 파일:

- `docs/LFM2_KO_VLLM_EVAL_RUNBOOK_20260628.ko.md`
- `scripts/run_lfm2_ko_vllm_lm_eval.sh`
- `scripts/run_lfm2_ko_eval_matrix.sh`
- `scripts/run_lfm2_ko_parallel_eval_1gpu.sh`
- `scripts/run_lfm2_ko_vllm_smoke.sh`
- `scripts/vllm_lfm2_ko_smoke.py`
- `scripts/summarize_lm_eval_results.py`
- `scripts/upload_full_model.py`

학습 완료 후 실행 순서:

1. 완료: `python scripts/upload_full_model.py`로 model repo에 README와 weights 업로드
2. 완료: `bash scripts/run_lfm2_ko_vllm_smoke.sh`
3. 완료: `.vllm-lfm-cu12`에 `lm-eval==0.4.11`, `datasets==4.3.0`, `ray`, `langdetect`, `immutabledict` 추가
4. 완료: `TASKS=ifeval LIMIT=100` 방향 확인 평가
5. 완료: `TASKS=ifeval` 전체 541문항 평가
6. 완료: `TASKS=gsm8k LIMIT=200 NUM_FEWSHOT=5` 회귀 평가
7. 완료: `TASKS=global_mmlu_full_ko LIMIT=500` 한국어 지식 평가
8. 결과 요약: `python scripts/summarize_lm_eval_results.py /home/work/.data/lfm2_ko_cpt/evals/<RUN_ID>_vllm_matrix`
9. 진행 중: `TP=1`, `CUDA_VISIBLE_DEVICES=0..7`로 8개 vLLM 평가 병렬 실행


공식 IFEval 전체 결과 (`20260628_022743_ifeval_full_vllm_vllm_matrix`, vLLM TP=8, limit 없음):

| metric | base | CPT | delta | relative |
|---|---:|---:|---:|---:|
| prompt strict | 0.2810 | 0.2976 | +0.0166 | +5.91% |
| prompt loose | 0.2921 | 0.3216 | +0.0295 | +10.10% |
| instruction strict | 0.4221 | 0.4365 | +0.0144 | +3.41% |
| instruction loose | 0.4341 | 0.4628 | +0.0287 | +6.61% |

방향 확인용 IFEval `LIMIT=100`도 전 항목 상승했다. 전체 점수는 위 표를 우선한다.


GSM8K 5-shot `LIMIT=200` 결과 (`20260628_023657_gsm8k_200_vllm_vllm_matrix`, vLLM TP=8, 방향 확인용 limited run):

| metric | base | CPT | delta | relative |
|---|---:|---:|---:|---:|
| exact_match strict | 0.2600 | 0.4250 | +0.1650 | +63.46% |
| exact_match flexible | 0.4250 | 0.4950 | +0.0700 | +16.47% |


Global MMLU Korean `LIMIT=500` 결과 (`20260628_024442_global_mmlu_ko_500_vllm_vllm_matrix`, vLLM TP=8, 방향 확인용 limited run):

| metric | base | CPT | delta | relative |
|---|---:|---:|---:|---:|
| global_mmlu_full_ko acc | 0.2803 | 0.3086 | +0.0283 | +10.10% |
| humanities acc | 0.2784 | 0.3022 | +0.0238 | +8.55% |
| other acc | 0.2914 | 0.3385 | +0.0471 | +16.16% |
| social_sciences acc | 0.2911 | 0.3404 | +0.0493 | +16.93% |
| stem acc | 0.2623 | 0.2591 | -0.0032 | -1.22% |

현재 추가 vLLM 병렬 평가 (`20260628_030217_parallel_1gpu`, GPU 0-7, 각 작업 TP=1):

실행 스크립트:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
RUN_ID_BASE=20260628_030217_parallel_1gpu bash scripts/run_lfm2_ko_parallel_eval_1gpu.sh
```

| GPU | task | limit | few-shot | status |
|---:|---|---:|---:|---|
| 0 | `mmlu_pro` | 500 | default | running |
| 1 | `mmlu_pro_law` | 500 | default | running |
| 2 | `mmlu_pro_economics` | 500 | default | running |
| 3 | `mmlu_prox_lite_ko` | 500 | default | running |
| 4 | `arc_challenge` | 500 | default | running |
| 5 | `gpqa_main_zeroshot` | 200 | default | running |
| 6 | `gsm8k` | full | 5 | running |
| 7 | `aime24` | full | default | running |

추가 완료 결과:

| task | metric | base | CPT | delta | relative | note |
|---|---|---:|---:|---:|---:|---|
| `arc_challenge` `LIMIT=500` | acc | 0.3600 | 0.4020 | +0.0420 | +11.67% | limited run |
| `arc_challenge` `LIMIT=500` | acc_norm | 0.3760 | 0.4140 | +0.0380 | +10.11% | limited run |
| `gsm8k` full 5-shot | exact_match strict | 0.2472 | 0.4617 | +0.2145 | +86.77% | full task |
| `gsm8k` full 5-shot | exact_match flexible | 0.4845 | 0.5701 | +0.0856 | +17.67% | full task |
| `mmlu_pro_economics` `LIMIT=500` | exact_match | 0.4420 | 0.4900 | +0.0480 | +10.86% | limited run |
| `mmlu_pro_law` `LIMIT=500` | exact_match | 0.1840 | 0.1240 | -0.0600 | -32.61% | limited run |
| `mmlu_prox_lite_ko` `LIMIT=500` | exact_match | 0.2585 | 0.1667 | -0.0918 | -35.51% | limited run |
| `global_mmlu_full_ko_professional_law` full | acc | 0.2581 | 0.2595 | +0.0014 | +0.54% | full subject |
| `global_mmlu_full_ko_professional_accounting` full | acc | 0.2730 | 0.2340 | -0.0390 | -14.29% | full subject |
| `global_mmlu_full_ko_high_school_macroeconomics` full | acc | 0.2436 | 0.2846 | +0.0410 | +16.83% | full subject |
| `global_mmlu_full_ko_virology` full | acc | 0.2831 | 0.3795 | +0.0964 | +34.05% | full subject |
| `global_mmlu_full_ko_world_religions` full | acc | 0.3450 | 0.4854 | +0.1404 | +40.70% | full subject |
| `leaderboard_instruction_following` / `leaderboard_ifeval` | prompt_level_loose_acc | 0.2976 | 0.3346 | +0.0370 | +12.42% | lm-eval leaderboard task |
| `global_mmlu_full_ko_business_ethics` full | acc | 0.2100 | 0.4500 | +0.2400 | +114.29% | full subject |
| `global_mmlu_full_ko_sociology` full | acc | 0.2886 | 0.4776 | +0.1891 | +65.52% | full subject |
| `global_mmlu_full_ko_computer_security` full | acc | 0.2900 | 0.4500 | +0.1600 | +55.17% | full subject |
| `global_mmlu_full_ko_marketing` full | acc | 0.3590 | 0.5000 | +0.1410 | +39.29% | full subject |
| `global_mmlu_full_ko_professional_psychology` full | acc | 0.2729 | 0.3284 | +0.0556 | +20.36% | full subject |
| `global_mmlu_full_ko_college_biology` full | acc | 0.2569 | 0.3333 | +0.0764 | +29.73% | full subject |
| `global_mmlu_full_ko_electrical_engineering` full | acc | 0.2759 | 0.3103 | +0.0345 | +12.50% | full subject |
| `global_mmlu_full_ko_high_school_world_history` full | acc | 0.2911 | 0.3376 | +0.0464 | +15.94% | full subject |
| `global_mmlu_full_ko_high_school_statistics` full | acc | 0.2870 | 0.1574 | -0.1296 | -45.16% | full subject |
| `global_mmlu_full_ko_astronomy` full | acc | 0.3421 | 0.2829 | -0.0592 | -17.31% | full subject |
| `kmmlu_hard_humss` `LIMIT=1000` | acc | 0.2533 | 0.2675 | +0.0143 | +5.63% | limited run |
| `kmmlu_hard` `LIMIT=1000` | acc | 0.2015 | 0.1720 | -0.0295 | -14.63% | limited run |
| `kmmlu_hard_stem` `LIMIT=1000` | acc | 0.1973 | 0.1564 | -0.0409 | -20.74% | limited run |

해석:

- CPT는 Global MMLU Korean 세부 과목 다수, GSM8K, BoolQ, ARC, IFEval/leaderboard instruction-following에서 확실히 오른다.
- 반대로 KMMLU hard 전체/STEM, MMLU-ProX-lite-ko, MMLU-Pro law, professional accounting, high-school statistics, astronomy, chemistry, formal logic은 하락했다.
- KMMLU direct exact-match는 base가 0에 가깝고 CPT도 0.01대라 현재 프롬프트/파서 진단용으로만 본다. 품질 판단은 `kmmlu_hard` acc와 `global_mmlu_full_ko_*` acc 중심이 맞다.
- 과목별 전체 표는 `docs/evals/GLOBAL_MMLU_KO_SUBJECT_SWEEP_20260628.md`에 둔다.

## 다음 post-training 의견

다음 단계는 broad CPT 반복보다 targeted post-training이 우선이다. 현재 CPT는 한국어 지식 과목과 일반 추론을 크게 올렸지만, KMMLU hard, MMLU-ProX-lite-ko, MMLU-Pro law, 회계 쪽에서 하락했다. 그래서 다음 학습 목표는 "한국어를 더 많이 읽히기"보다 "한국어 선택형/법률/회계/툴콜 포맷을 정확히 맞히게 하기"다.

추천 순서:

1. Korean MCQA remediation SFT
   - 데이터: KMMLU, KMMLU-hard, MMLU-ProX-lite-ko 형식, 한국 법률/회계/금융 객관식, 현재 틀린 샘플 hard negative.
   - 출력: 정답 선택지 라벨, 짧은 근거, 필요 시 최종 answer-only 필드.
   - 목표: `kmmlu_hard` 0.1720 -> 0.2300 이상, `kmmlu_hard_stem` 0.1564 -> 0.2100 이상, `mmlu_prox_lite_ko` 0.1667 -> 0.3000 목표.
2. Korean instruction-following SFT
   - 데이터: Ko-IFEval식 제약, 한국어 포맷 준수, 불확실성 표현, 거절, 다조건 지시문.
   - 목표: `leaderboard_instruction_following` prompt loose 0.3346 이상 유지, IFEval full loose 0.3216 이상 유지.
3. Tool/agent SFT
   - 데이터: 한국어 BFCL식 함수 호출, terminal/tool-call traces, JSON schema, 다중 턴 작업 완료.
   - 목표: Liquid 공식 축인 BFCL/Tau2 계열을 나중에 별도 harness로 평가 가능하게 만들기.
4. DPO/ORPO/KTO
   - 데이터: 현재 base-vs-CPT 평가 실패 샘플에서 만든 선호쌍.
   - 선호: 정확한 선택지 추출, 짧고 안정적인 한국어, 유효한 JSON/tool-call, 모르면 불확실성 표시.
5. Preservation mix
   - 보존: GSM8K flexible 0.5701 이상, BoolQ 0.7902 이상, ARC/Global MMLU KO 상승 과목을 유지하도록 일부 섞는다.

RL/GRPO는 첫 단계로 두지 않는다. 정답 검증이 명확한 tool-call, option correctness, math final-answer에 대해서만 SFT/DPO 이후 적용하는 편이 낫다.

실패/교체:

- `gpqa_main_zeroshot`: `Idavidrein/gpqa` gated dataset 접근 실패로 중단.
- `aime24`: vLLM/lm-eval 조합에서 empty decoder prompt 오류로 중단.
- 위 두 작업과 완료된 짧은 작업이 만든 빈 GPU는 Global MMLU Korean 세부 과목 full 평가로 다시 채웠다.

추가로 계속 돌릴 평가 기준:

- Liquid LFM2.5-8B-A1B 공식 축: knowledge, instruction following, math, tool use, agentic workflows.
  - 공식 링크: https://www.liquid.ai/blog/lfm2-5-8b-a1b
  - 공식 표에 있는 주요 이름: AA-Omniscience, IFEval, IFBench, Multi-IF, MATH500, AIME25, AIME26, BFCLv3, BFCLv4, Tau2 Telecom, Tau2 Retail.
  - 지금 바로 lm-eval/vLLM로 계속 가능한 것: `ifeval`, `leaderboard_instruction_following`, `leaderboard_math_hard`, `hendrycks_math`, `gsm8k`, `mmlu_pro`, `arc_challenge`, `hellaswag`, `winogrande`, `piqa`, `boolq`.
  - 별도 harness가 필요한 것: BFCLv3/v4, Tau2 Telecom/Retail, IFBench/Multi-IF, AA-Omniscience.
- Liquid LFM2.5-1.2B-JP-202606식 언어 특화 축: knowledge, instruction following, math, code, tool use, domain average.
  - 공식 링크: https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606
  - JP 표 기준 이름: JMMLU-ProX, JMMLU, JCulture, JGPQA, J-MIFEval, JFBench, J-GSM8K, J-MATH500, JHumanEval+, J-BFCLv3.
  - 한국어 대응 우선순위: `kmmlu`, `kmmlu_direct`, `kmmlu_hard`, `kmmlu_cot_hard`, Ko-IFEval, Ko-GSM8K, Korean MATH500식 세트, Korean HumanEval/MBPP식 코드 세트, Korean BFCL식 tool-call 세트.
  - 지금 바로 lm-eval/vLLM로 가능한 것: `kmmlu`, `kmmlu_direct`, `kmmlu_hard`, `kmmlu_cot_hard`, `global_mmlu_full_ko`, 세부 과목 full 평가.

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
