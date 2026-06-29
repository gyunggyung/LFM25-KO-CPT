# LFM2.5 KO CPT 단계별 의미와 목표

작성일: 2026-06-29  
폴더: `/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt`

이 문서는 `LFM2.5-8B-A1B-KO-CPT-FULL`을 왜 CPT로 만들었고, 각 단계가
어떤 의미를 가지며, 후속 SFT/Agentic 단계와 어떻게 이어지는지 정리한다.

## 전체 목표

CPT의 목적은 chat model을 곧바로 더 친절하게 만드는 것이 아니라, base model의
파라미터 안에 한국어 분포와 도메인 지식을 다시 밀어 넣는 것이다.

이 프로젝트에서 CPT가 담당한 역할은 다음 네 가지다.

- 한국어 일반 지식과 문체 이식: Korean Wiki와 일반 한국어 텍스트로 한국어
  token sequence를 더 자연스럽게 예측하게 한다.
- 법률/금융 도메인 지식 이식: 법령, 판례, 행정규칙, 법률 태스크, 금융/회계
  텍스트를 넣어 한국어 전문 용어와 문서 구조를 익히게 한다.
- LFM 계열 동작 보존: instruction, legal RAG, terminal/tool-use 샘플은 LFM
  ChatML-like 형식으로 감싸서 LFM2.5의 대화/도구 형식을 완전히 잃지 않게 한다.
- 후속 SFT의 기반 만들기: CPT만으로는 선택지 추출, 지시 준수, tool-call 같은
  정렬 행동이 충분히 안정되지 않는다. 그래서 CPT는 지식 기반이고, SFT는 행동
  교정이라는 역할 분담을 둔다.

## Stage 0: 모델/토크나이저 점검

| 항목 | 결정 |
|---|---|
| base | `LiquidAI/LFM2.5-8B-A1B` |
| tokenizer vocab | 기존 `125017` 유지 |
| vocab 확장 | 하지 않음 |
| 기본 context | 8192 |

왜 이 단계를 했는가:

- LFM2.5 tokenizer가 한국어를 실제로 어느 정도 효율적으로 자르는지 확인해야 했다.
- 로컬 샘플에서 한국어가 약 `1.78 chars/token` 수준으로 확인되어, 짧은 일정에서는
  vocab 확장보다 기존 tokenizer 유지가 안전했다.
- vocab을 늘리면 embedding/head resize, checkpoint 호환성, vLLM/추론 호환성,
  후속 SFT 데이터 재토큰화 문제가 생긴다. 이번 run은 6월 30일 마감이 있으므로
  안정성이 우선이었다.

효과:

- base LFM tokenizer와 완전히 호환되는 CPT/SFT 체인을 유지했다.
- HF 업로드, vLLM 평가, GGUF 후보 변환에서 tokenizer mismatch 위험을 줄였다.

## Stage 1: Seed/Practice CPT

| 항목 | 내용 |
|---|---|
| 목적 | full fine-tuning 설정과 GPU/저장/checkpoint 경로 검증 |
| 데이터 | cap이 걸린 seed mix |
| 방식 | full-parameter CPT 우선, LoRA는 fallback 성격 |
| context | 8192 |

왜 이 단계를 했는가:

- 8B 모델 full fine-tuning은 데이터보다 먼저 런타임 안정성이 문제다.
- H200 8장 DDP에서 remote code, Unsloth/TRL 경로, checkpoint 저장, resume가
  실제로 작동하는지 확인해야 했다.
- seed run은 성능을 노리는 본 학습이 아니라, batch size, VRAM, 저장 주기,
  재개 경로를 검증하는 안전 장치다.

효과:

- full CPT가 가능한지, checkpoint가 만들어지는지, resume가 가능한지 확인했다.
- LoRA는 "빠른 어댑터 공개"나 "full FT 실패 시 fallback" 정도의 의미로 남기고,
  최종 목표는 full-parameter CPT로 확정했다.

## Stage 2: Full LFM-style Preprocessing

| 출력 | 경로 |
|---|---|
| full raw mix | `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_20260627.jsonl` |
| LFM-style full mix | `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl` |
| source shards | `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627` |

데이터 의미:

| source family | 역할 |
|---|---|
| Korean Wiki | 한국어 일반 지식, 문장 분포, 설명체 |
| Korean finance | 금융/회계/공시/재무 용어와 설명 |
| Korean legal raw | 법령/판례/행정규칙 문체와 장문 구조 |
| Korean legal tasks | 질의응답과 시험형 문제의 기본 구조 |
| legal RAG/source agent | 근거 문서 기반 답변 습관 |
| current law bar JSON | 변시/법률형 정답 JSON과 선택지 추출 형식 |
| terminal ToolBench | LFM tool-use/terminal 스타일 보존 |

왜 LFM-style 변환을 했는가:

- raw 문서는 plain completion으로 둬도 되지만, instruction/RAG/tool 샘플을 그냥
  이어 붙이면 LFM의 role과 tool-call 경계를 잃는다.
- 대화형 샘플은 `<|startoftext|>`, `<|im_start|>`, `<|im_end|>` 계열의 LFM
  chat text로 감싸야 후속 SFT와 같은 형식으로 이어진다.
- tool-use 샘플은 `<|tool_call_start|>`와 `<|tool_call_end|>`를 보존해야 나중에
  BFCL/Tau2/agent harness에서 도구 호출 형식이 무너지지 않는다.

효과:

- CPT 단계에서도 일부 instruction/tool 구조가 완전히 지워지지 않는다.
- SFT 전환 시 tokenizer, chat template, source attribution을 같은 규칙으로
  이어갈 수 있다.
- source shard를 따로 공개해 어떤 데이터가 들어갔는지 감사 가능하게 만들었다.

## Stage 3: Full CPT 학습

| 항목 | 값 |
|---|---|
| 방식 | full-parameter continued pretraining |
| epoch | 1 |
| context length | 8192 |
| per-device batch | 2 |
| gradient accumulation | 4 |
| effective batch | 64 sequences/update |
| upper token/update | `64 * 8192 = 524,288` |
| planned/final steps | `10196 / 10196` |
| final train loss | 약 `0.712` |
| save policy | 1000 steps, keep 4 checkpoints |

왜 8192 context인가:

- 법률/판례/RAG/terminal trace는 4k보다 긴 샘플이 많다.
- 8192는 H200 8장 full FT에서 안정적으로 돌릴 수 있는 현실적인 길이다.
- 16k/32k는 별도 long-context 확장 실험이 필요하다. 이번 run에서 바로 올리면
  VRAM, batch, 학습 시간, 평가 비용이 모두 커진다.

왜 1 epoch인가:

- CPT는 이미 강한 base model에 도메인 분포를 이식하는 작업이다.
- 수B token을 여러 epoch 돌리면 한국어 도메인에는 더 맞을 수 있지만, base의
  영어/수학/일반 instruction 능력이 더 흔들릴 수 있다.
- 1 epoch 후 SFT와 평가로 행동을 교정하는 쪽이 일정과 안정성 면에서 맞다.

효과:

- 여러 한국어 지식/일반 benchmark에서 상승이 확인됐다.
- IFEval, GSM8K, BoolQ, ARC 등도 일부 상승했다.
- 반대로 KMMLU hard, MMLU-ProX Lite KO, 법/회계/STEM 선택지 추출에서 하락이
  확인되어, CPT만으로는 완성 모델이 아니라는 결론이 나왔다.

## Stage 4: Final Reconstruction / Upload

| 항목 | 내용 |
|---|---|
| 최종 기준 checkpoint | `checkpoint-10196` |
| final model | `final_full` |
| HF model | `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL` |
| public data | CPT raw, LFM-style raw, source shards |

왜 재구성이 필요했는가:

- 학습 step과 checkpoint 저장은 완료됐지만, 추가 `trainer.save_model(final_full)`
  중 일부 rank에서 문제가 발생한 적이 있었다.
- 최초 `final_full`을 그대로 믿지 않고, 무결성 확인이 되는 `checkpoint-10196`
  inference files에서 final을 재구성했다.

효과:

- 업로드된 모델은 검증된 final checkpoint 기준이다.
- 후속 SFT는 이 CPT final을 base로 삼아 이어간다.

## Stage 5: CPT 평가와 후속 판단

평가 목표:

- base 대비 CPT가 실제로 한국어 지식과 instruction을 끌어올렸는지 확인한다.
- 동시에 base의 강점이 무너진 항목을 찾는다.
- 후속 SFT 데이터를 어디에 집중할지 결정한다.

확인된 큰 흐름:

- 상승: IFEval 계열, GSM8K, BoolQ, ARC, Global MMLU KO 일부 subject.
- 하락: KMMLU hard, MMLU-ProX Lite KO, 법/회계/STEM exact option extraction.

해석:

- "한국어가 전반적으로 나빠졌다"가 아니라, parser-sensitive한 객관식/정답 추출
  형식이 약해진 것이 핵심이다.
- CPT는 지식 이식에는 의미가 있었지만, instruction-following과 정답 형식은
  SFT로 다시 잡아야 한다.

## 후속 SFT와의 연결

CPT 이후 broad CPT를 계속 반복하지 않는 이유:

- 이미 한국어 지식은 들어갔다.
- 추가 CPT는 raw 분포를 더 넣는 작업이라, 객관식 정답 형식/툴콜/에이전트
  행동을 직접 교정하지 못한다.
- 후속 목표는 "한국어를 더 많이 읽은 모델"이 아니라 "한국어로 정확히 지시를
  따르고, 근거를 쓰고, 도구를 올바르게 호출하는 모델"이다.

그래서 후속 작업은 다음 순서가 맞다.

1. Stage0/0b SFT로 포맷과 학습 체인 검증
2. Stage1 SFT로 finance/Text2SQL/legal/terminal 핵심 행동 학습
3. Stage2 SFT로 한국어 일반, SWE/coding, reasoning, KoTSQA를 섞어 균형 보강
4. Stage3 Agentic/Fable SFT로 문서 읽기, 로그 진단, tool-call, terminal 행동 보강
5. vLLM/lm-eval과 agent harness로 base/CPT/SFT/Agentic 비교

## 운영 원칙

- CPT 폴더는 지식 이식과 base-vs-CPT 평가의 기록이다.
- SFT 폴더는 행동 교정, agentic 후속 학습, 최종 평가의 기록이다.
- 데이터는 public HF dataset repo로 공개하고, 각 repo README에 source와 format을
  적는다.
- 모델 카드는 성능이 나온 항목과 하락한 항목을 같이 적어야 한다. 좋은 점만
  적으면 후속 SFT의 목적이 흐려진다.
