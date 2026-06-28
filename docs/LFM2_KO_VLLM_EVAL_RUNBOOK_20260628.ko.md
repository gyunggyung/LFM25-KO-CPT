# LFM2.5-8B-A1B-KO vLLM 평가 Runbook

작성일: 2026-06-28

## 현재 원칙

학습이 1순위다. 현재 full CPT가 끝나기 전에는 평가 GPU 작업을 시작하지 않는다. 이 문서는 학습 완료 직후 같은 조건으로 `LiquidAI/LFM2.5-8B-A1B`와 `LFM2.5-8B-A1B-KO-CPT-FULL`을 비교하기 위한 실행 지침이다.

## 비교 대상

- Base: `LiquidAI/LFM2.5-8B-A1B`
- CPT: `/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full`
- 결과 저장: `/home/work/.data/lfm2_ko_cpt/evals/`
- 실행 위치: `/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt`

## vLLM 환경

이 워크스페이스에서는 vLLM과 `lm_eval`이 한 가상환경에 깔려 있지 않다. 그래서 실행 스크립트가 다음 조합을 고정한다.

- vLLM 실행 Python: `/home/work/.projects/LLM-OS-Models/Terminal/.vllm-lfm/bin/python`
- `lm_eval`/datasets 보조 path: `/home/work/.projects/LLM-OS-Models/Terminal/.liquid-sft-env/lib/python3.12/site-packages`
- user site fallback: `/home/work/.local/lib/python3.12/site-packages`
- tensor parallel: `TP=8`
- dtype: `bfloat16`
- `max_model_len=8192`
- `gpu_memory_utilization=0.88`

## 공개/공식 벤치마크

LiquidAI 모델 카드가 원본 LFM2.5 성능을 `IFEval`, `IFBench`, `Multi-IF`, `MATH500`, `AIME25`, `BFCLv3/v4`, `Tau²`로 보여준다. 우리가 바로 돌릴 1차 vLLM 회귀 평가는 로컬 harness에 확인된 태스크 중심으로 잡는다.

### 1. 원본 성능 보존 회귀

- `ifeval`: instruction following 보존 확인
- `mmlu_pro`: 영어 전문 지식/추론 회귀 확인
- `mmlu_pro_law`, `mmlu_pro_economics`: 법률/경제 도메인 회귀 확인
- `gpqa_main_zeroshot`, `gpqa_diamond_zeroshot`: 고난도 과학 QA 회귀 확인
- `gsm8k`: 산술 reasoning 회귀 확인
- `arc_challenge`: 상식/과학 추론 회귀 확인
- `aime24`, `aime25`: 수학 고난도 회귀 확인

### 2. 한국어 성능 상승 확인

- `global_mmlu_full_ko`: 한국어 다분야 지식
- `global_mmlu_full_ko_professional_law`: 한국어 법률/전문 법 영역
- `global_mmlu_full_ko_professional_accounting`: 한국어 회계/금융 인접 영역
- `mmlu_prox_lite_ko`, `mmlu_prox_ko`: 한국어 MMLU-ProX 계열

### 3. 추가 구현 대상

아래는 모델카드 설득력을 위해 이어서 넣을 공개 데이터셋이다. 현재 설치 harness 기본 목록에는 바로 보이지 않아 custom task 또는 별도 scorer가 필요하다.

- `KMMLU`: https://huggingface.co/datasets/HAERAE-HUB/KMMLU
- `KMMLU-Pro`: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro
- `KMMLU-Redux`: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Redux
- `Ko-IFEval`: https://huggingface.co/datasets/davidkim205/ko-ifeval
- `Ko-GSM8K`, `Ko-ARC`: 한국어 추론 회귀/상승 확인용

## Smoke 평가

`scripts/run_lfm2_ko_vllm_smoke.sh`는 공식 점수가 아니다. 긴 평가 전에 base와 CPT를 vLLM으로 직접 호출해서 다음이 깨졌는지 확인한다.

- 한국어 법률 설명
- 한국어 금융 설명
- 한국어 위키/역사 요약
- 영어 지시 준수
- LFM2.5 tool-call special token 형식

LFM2.5는 ChatML-like template을 쓰고, tool call은 `<|tool_call_start|>`와 `<|tool_call_end|>` 사이에 출력하는 형식이다. smoke 스크립트는 tokenizer의 `apply_chat_template()`를 우선 사용한다.

## 실행 명령

학습 완료와 `final_full` 저장을 확인한 뒤 실행한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt

# 1차 빠른 확인: base + CPT 직접 생성 smoke
bash scripts/run_lfm2_ko_vllm_smoke.sh

# 2차 짧은 공개 벤치마크 smoke: base + CPT, limit 적용
TASK_SET=smoke LIMIT=100 bash scripts/run_lfm2_ko_eval_matrix.sh

# 3차 한국어 중심 공개 벤치마크
TASK_SET=korean bash scripts/run_lfm2_ko_eval_matrix.sh

# 4차 원본 성능 보존 회귀
TASK_SET=regression bash scripts/run_lfm2_ko_eval_matrix.sh

# 5차 전체 1차 표 생성
TASK_SET=full bash scripts/run_lfm2_ko_eval_matrix.sh
```

단일 모델/단일 태스크 디버깅은 다음처럼 한다.

```bash
MODEL_PATH=LiquidAI/LFM2.5-8B-A1B \
MODEL_NAME=base_debug \
TASKS=ifeval \
LIMIT=50 \
bash scripts/run_lfm2_ko_vllm_lm_eval.sh
```

## 예상 시간

현재 학습 로그 기준 full CPT는 2026-06-28 10:30-10:40 KST 부근에 1 epoch step이 끝날 것으로 본다. 이후 final save가 10-20분 걸릴 수 있다.

- smoke 생성 평가: 20-40분
- `TASK_SET=smoke LIMIT=100`: 20-40분
- `TASK_SET=korean`: 1-3시간
- `TASK_SET=regression`: 2-4시간
- `TASK_SET=full`: 4-8시간
- KMMLU/KMMLU-Pro/Ko-IFEval custom scorer 추가와 전체 실행: 별도 2-6시간

따라서 첫 의미 있는 before/after 결과는 final save 후 약 1시간 안에 나오고, 공개 벤치마크 표 1차본은 2026-06-28 13:00-15:00 KST, full 표는 17:00-19:00 KST 정도가 현실적인 목표다.

## 결과 반영 순서

1. 결과 JSON을 `/home/work/.data/lfm2_ko_cpt/evals/`에 보관한다.
2. base와 CPT의 같은 태스크 같은 설정만 표로 비교한다.
   ```bash
   python scripts/summarize_lm_eval_results.py /home/work/.data/lfm2_ko_cpt/evals/<RUN_ID>_vllm_matrix
   ```
3. `model_cards/LFM2.5-8B-A1B-KO-CPT-FULL.md`에 benchmark table을 갱신한다.
4. 모델 repo `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`에 결과와 모델 카드 업데이트를 올린다.
5. 필요하면 dataset repo에도 eval manifest를 추가한다.
