# LFM2.5-8B-A1B-KO CPT 런북

작성일: 2026-06-27  
작업 위치: `/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt`  
산출물 위치: `/home/work/.data/lfm2_ko_cpt`

## 결론

GLM5.2 튜닝은 중단했다. 새 목표는 `LiquidAI/LFM2.5-8B-A1B`를 한국어 법률/위키/금융 중심으로 계속 사전학습(CPT)해서 `LFM2.5-8B-A1B-KO` 계열 LoRA 어댑터와 병합 모델을 만드는 것이다.

첫 실행은 tokenizer를 확장하지 않는다. LFM2.5는 공식 문서상 한국어가 지원 언어에 포함되어 있고, 로컬 `LiquidAI/LFM2.5-8B-A1B` tokenizer는 vocab size `125017`로 확인됐다. 한국어 샘플 tokenization도 평균 약 `1.78 chars/token`이라 짧은 CPT/LoRA 단계에서는 호환성을 유지하는 편이 낫다. vocab 확장은 장기 CPT에서 토큰 효율을 더 끌어올릴 때 별도 실험으로 한다.

## 근거 링크

- Liquid AI LFM2.5 발표: https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai
- LFM2.5-8B-A1B 모델 카드: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
- LFM2.5 Japanese 모델 카드: https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606
- LFM JP 컬렉션: https://huggingface.co/collections/LiquidAI/lfm-jp
- LFM2 text generation/prompting: https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting
- LFM2 chat template: https://docs.liquid.ai/lfm/key-concepts/chat-template
- LFM2 tool use: https://docs.liquid.ai/lfm/key-concepts/tool-use
- Liquid 공식 Unsloth fine-tuning 문서: https://docs.liquid.ai/lfm/fine-tuning/unsloth
- Unsloth LFM2.5 문서: https://unsloth.ai/docs/models/tutorials/lfm2.5
- 한국어 데이터셋 큐레이션: https://github.com/gyunggyung/LLM-Ko-Datasets

## JP 모델에서 배운 점

공식 자료가 JP 모델의 전체 학습 레시피를 완전히 공개하지는 않는다. 다만 공개된 모델 카드와 Liquid 문서 기준으로 보면 `LFM2.5-1.2B-JP-202606`은 단순 SFT가 아니라 일본어 언어 적응과 chat/tool post-training이 결합된 모델로 보는 게 맞다.

- JP 모델 카드는 일본어 일반 목적 chat model로 설명하며, 일본어 지식, instruction following, math, code, tool-use 성능 개선을 목표로 한다.
- 공개 벤치마크는 `JMMLU-ProX`, `JMMLU`, `JCulture`, `JGPQA`, `J-MIFEval`, `JFBench`, `J-GSM8K`, `J-MATH500`, `JHumanEval+`, `J-BFCLv3`를 사용한다.
- Liquid LFM2.5 발표 글은 Instruct 계열이 SFT, preference alignment, multi-stage RL을 사용한다고 설명한다.
- 따라서 KO 모델도 먼저 full-parameter CPT로 한국어 분포/지식을 이식하고, 이후 한국어 instruction/tool-use SFT 또는 GRPO를 붙여 공개 벤치마크 표를 만드는 순서가 맞다.

LFM formatting 원칙:

- 위키/법률/금융 원문은 CPT 목적의 plain text completion corpus로 둔다.
- QA, legal RAG, terminal/tool/agentic 데이터는 LFM2 ChatML-like 형식으로 저장한다.
- LFM chat template 핵심 토큰: `<|startoftext|>`, `<|im_start|>`, `<|im_end|>`.
- 지원 role: `system`, `user`, `assistant`, `tool`.
- tool call은 `<|tool_call_start|>`와 `<|tool_call_end|>`를 보존한다.

## 현재 로컬 상태

- GLM 캐시 삭제 완료:
  - `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2`
  - `/home/work/.data/huggingface/hub/models--zai-org--GLM-5.2-FP8`
- 디스크 여유: `/home/work/.data` 기준 약 2.8T
- LFM2.5-8B-A1B 캐시 존재:
  - `/home/work/.data/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B`
  - `/home/work/.cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B`
- 가상환경: `/home/work/.projects/LLM-OS-Models/Terminal/.liquid-sft-env`
- HF 토큰: `/home/work/.projects/LLM-OS-Models/Terminal/.env`의 `HF_TOKEN` 사용
- LFM2.5-8B-A1B tokenizer 점검:
  - vocab size: `125017`
  - 한국어 샘플 평균: 약 `1.78 chars/token`

## 1차 CPT 데이터 믹스

설정 파일: `lfm2_ko_cpt/configs/ko_cpt_sources_20260627.json`

사용 데이터:

- 위키: `/home/work/.data/huggingface/hrm_text_extra/sft/kowiki_raw_full_20260524.jsonl`
- 금융: `/home/work/.data/huggingface/hrm_text_extra/finance/bcai_finance_kor_hrm_20260524.jsonl`
- 법률 원문: `/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_raw_full_20260523.jsonl`
- 법률 태스크: `/home/work/.data/huggingface/hrm_text_extra/sft/korean_legal_tasks_full_20260524.jsonl`
- 행정규칙/판례: `/home/work/.data/huggingface/hrm_text_extra/sft/korean_admrule_precedent_raw_full_20260524.jsonl`
- 한국 법률 RAG/agentic: `/home/work/.data/harness1/ko_legal_source_rag/source_rag_balanced_20260621_v1/ko_legal_source_agent_sft.jsonl`
- 도구/agentic 보존용 소량: `/home/work/.data/hrm_text_raw_sft/lfm25_terminal_toolbench_hrm_turns_v1.jsonl`

1차 seed mix는 약 19.5만 문서 cap으로 시작한다. 학습이 안정되면 cap을 늘려 1M+ 문서/수B token 단계로 확장한다.

전체 원천 처리 설정:

- `lfm2_ko_cpt/configs/ko_cpt_sources_full_20260627.json`
- 순차 출력: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_20260627.jsonl`
- 병렬 LFM-style 출력: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl`
- 추정 규모: 원천 JSONL 약 23GB 기준 `4~6B tokens`
- 정확한 token 수는 full mix 생성 후 `stats`와 tokenizer pass로 확정한다.

병렬 LFM-style full preprocess:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
../.liquid-sft-env/bin/python scripts/build_ko_cpt_mix_parallel.py \
  --config configs/ko_cpt_sources_full_20260627.json \
  --output /home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl \
  --stats-output /home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.stats.json \
  --shard-dir /home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627 \
  --workers 8
```

이 병렬 builder는 raw corpus는 plain completion으로 두고, `messages`/`conversations`/QA/agentic 데이터는 LFM2.5 ChatML 계열 형식인 `<|startoftext|><|im_start|>role ... <|im_end|>`로 저장한다.

## 학습 방식

- 베이스: `LiquidAI/LFM2.5-8B-A1B`
- 방식: Unsloth + TRL `SFTTrainer` text completion CPT
- 기본 목표: full-parameter CPT
- LoRA: 실패 시 fallback 또는 빠른 어댑터 공개용으로만 사용
- 컨텍스트: 8192
- 8GPU DDP: `torchrun --nproc_per_node=8`
- full FT 초기 batch:
  - per-device batch 2
  - grad accumulation 4
  - effective batch sequence 64
  - 8192 기준 update당 약 52만 token
- 기본 step: 3000
- 저장: 50 step마다 checkpoint, 최대 4개 보존
- 재개: `RESUME_FROM_CHECKPOINT=auto`

현재 practice run:

- 세션: `lfm2_ko_cpt_full_seed_practice_20260627`
- 데이터: `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_seed_20260627.jsonl`
- 목적: full FT 설정, VRAM, checkpoint 저장, step 속도 확인
- 실제 full-data run은 `/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl` 생성 완료 후 watcher가 자동 시작한다.

## 실행

tmux로 시작:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
tmux new-session -d -s lfm2_ko_cpt_lora_20260627 \
  'bash lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_lora.sh'
```

로그 확인:

```bash
tmux attach -t lfm2_ko_cpt_lora_20260627
tail -f /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt/logs/20260627_lfm25_8b_ko_cpt_lora/train.log
```

중단:

```bash
tmux send-keys -t lfm2_ko_cpt_lora_20260627 C-c
```

재개:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
tmux new-session -d -s lfm2_ko_cpt_lora_20260627_resume \
  'RESUME_FROM_CHECKPOINT=auto bash lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_lora.sh'
```

## Hugging Face 업로드

학습 중 checkpoint가 생기면 최신 LoRA 어댑터만 업로드한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_cpt
source ../.liquid-sft-env/bin/activate
export HF_TOKEN="$(awk -F= '/^HF_TOKEN=/{print substr($0, index($0, "=") + 1)}' ../.env | tail -n 1)"
python scripts/upload_latest_lora.py \
  --output-dir /home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-LoRA-20260627_lfm25_8b_ko_cpt_lora \
  --repo-id LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-LoRA
```

## 예상 시간

H200 8대 기준 rough estimate:

- 데이터 seed mix 생성: 5~30분
- 첫 100 step checkpoint: 학습 시작 후 20~60분
- 3000 step: 약 8~18시간
- 병합/업로드/모델카드 정리: 30~90분

실제 시간은 packing/tokenization 처리량과 LFM2.5 remote code/Unsloth kernel 경로에 따라 달라진다.

## 다음 작업

1. seed CPT 안정화 후 batch를 올려 VRAM 사용률을 확인한다.
2. `LLM-Ko-Datasets`에서 `Won-Instruct`, `Yi-Sang/KOREAson`, `NuminaMath-CoT-Ko`, `Magpie-Pro-MT-300K-ko`를 후속 SFT/Reasoning mix 후보로 검토한다.
3. CPT LoRA를 병합해 `LFM2.5-8B-A1B-KO` merged checkpoint를 만든다.
4. vLLM으로 한국어 법률 QA, 금융 설명, Ko-MMLU/KMMLU 계열, tool-call smoke를 평가한다.
5. 결과가 유의미하면 GGUF/quant 모델과 모델카드를 공개한다.
