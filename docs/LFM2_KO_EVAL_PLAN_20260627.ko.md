# LFM2.5-8B-A1B-KO 평가 계획

작성일: 2026-06-27

## 방향

LFM2.5 JP 모델 카드가 일본어 모델 성능을 `JMMLU-ProX`, `JMMLU`, `JCulture`, `JGPQA`, `J-MIFEval`, `JFBench`, `J-GSM8K`, `J-MATH500`, `JHumanEval+`, `J-BFCLv3`로 보여준 것처럼, KO 모델도 공개 한국어 벤치마크를 중심으로 before/after 표를 만든다.

## 1차 공개 벤치마크

1. `KMMLU`  
   - 링크: https://huggingface.co/datasets/HAERAE-HUB/KMMLU  
   - 목적: 한국어 지식/시험형 다분야 이해력
   - 목표: base `LiquidAI/LFM2.5-8B-A1B` 대비 상승

2. `KMMLU-Pro`  
   - 링크: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro  
   - 목적: 한국 국가 전문 자격 시험 기반 고난도 전문 지식
   - 목표: 법률/금융 CPT 효과 확인

3. `KMMLU-Redux`  
   - 링크: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Redux  
   - 목적: 기존 KMMLU leakage/오류 리스크를 줄인 한국어 지식 평가
   - 목표: 오염 리스크가 낮은 지표에서 base 대비 상승

4. `Ko-IFEval`  
   - 링크: https://huggingface.co/datasets/davidkim205/ko-ifeval  
   - 목적: 한국어 명시 지시 준수/형식 준수
   - 목표: CPT 후 한국어 instruction-following 저하 없이 상승 또는 유지

5. Korean Benchmark Suite 계열  
   - 후보: `Ko-ARC`, `Ko-GSM8K`, `Ko-EQBench`, `Ko-WinoGrande`, `Ko-LAMBADA`, `Ko-IFEval`
   - 목적: 지식뿐 아니라 추론/상식/언어모델링 균형 확인
   - 사용 가능 데이터/평가 스크립트 확인 후 적용

## 보조 도메인 평가

공개 지표와 별도로 모델카드 설득력을 위해 도메인 holdout도 만든다.

- 법률: harness-1 한국 법률 RAG/bar exam holdout
- 금융: BCAI/Won-Instruct style holdout
- 위키: 한국어 요약/질의응답 holdout
- 도구형 동작: terminal/tool call smoke

보조 도메인은 “공식 순위”로 주장하지 않고, 도메인 동작 확인용으로만 표기한다.

## 실행 순서

1. Base `LiquidAI/LFM2.5-8B-A1B` baseline 측정
2. seed CPT checkpoint 측정
3. full CPT checkpoint 측정
4. 모델카드에 같은 prompt/template/inference 설정으로 before/after 표 작성

## 학습과 연결

- seed CPT: 약 `0.26B tokens`, GPU 즉시 가동용
- full CPT: 원천 데이터 전체 처리 후 약 `4~6B tokens` 예상
- H200 8대에서 full CPT는 checkpoint resume 방식으로 이어간다.
- VRAM 목표: 안정 step 확인 후 per-device batch를 `4 -> 6 -> 8` 순서로 올린다.

