---
license: other
base_model: LiquidAI/LFM2.5-8B-A1B
library_name: transformers
pipeline_tag: text-generation
language:
  - ko
  - en
tags:
  - liquid-ai
  - lfm2.5
  - korean
  - continued-pretraining
  - full-finetuning
  - legal
  - finance
  - wiki
  - tool-use
datasets:
  - HAERAE-HUB/KMMLU
  - LGAI-EXAONE/KMMLU-Pro
  - LGAI-EXAONE/KMMLU-Redux
---

# LFM2.5-8B-A1B-KO-CPT-FULL

Full-parameter Korean continued-pretraining project for [`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B).

This model is intended to make LFM2.5 stronger at Korean legal, finance, wiki-style knowledge, and terminal/tool-use behavior while preserving the base model's general English and instruction-following ability.

> Status: full CPT completed on 2026-06-28. Weights are prepared from the verified `checkpoint-10196` final-step checkpoint and uploaded to Hugging Face. vLLM smoke passed, and IFEval, GSM8K limited, and Global MMLU Korean limited evaluations show gains over the base model.

## Contents

- [English](#english)
- [Quick Start](#quick-start)
- [Colab Example](#colab-example)
- [Training Configuration](#training-configuration)
- [Data Mix](#data-mix)
- [Legal Data Attribution](#legal-data-attribution)
- [Korean](#korean)
- [한국어 사용법](#한국어-사용법)
- [한국어 학습 설정](#한국어-학습-설정)
- [Evaluation Plan](#evaluation-plan)

## English

`LFM2.5-8B-A1B-KO-CPT-FULL` is a full fine-tuned Korean CPT checkpoint, not a LoRA adapter. The training objective is text completion over a Korean-heavy corpus, with LFM chat-template formatting applied to instruction, RAG, and tool-use examples.

Target strengths:

- Korean legal document understanding and legal RAG-style answering
- Korean finance explanations and finance-domain terminology
- Korean wiki/general knowledge prose
- Korean instruction-following
- Terminal/tool-use style structured assistant behavior

## Quick Start

The examples below use the full model repository.

### Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a precise Korean assistant."},
    {"role": "user", "content": "대한민국 민법상 계약 해제와 해지의 차이를 간단히 설명해줘."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### vLLM

```bash
vllm serve LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size 8
```

OpenAI-compatible request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL",
    messages=[
        {"role": "system", "content": "You are a precise Korean assistant."},
        {"role": "user", "content": "한국 기준금리 인상이 은행 순이자마진에 미치는 영향을 설명해줘."},
    ],
    temperature=0.5,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

## Colab Example

Use this after model weights are uploaded. For typical Colab GPUs, start with 4-bit loading to avoid OOM.

```python
!pip install -U "transformers>=4.44" accelerate bitsandbytes sentencepiece huggingface_hub
```

```python
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Optional for gated/private models:
# login("hf_xxx")

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "너는 한국어로 정확하고 간결하게 답하는 어시스턴트다."},
    {"role": "user", "content": "한국어로 주택임대차보호법의 대항력 요건을 설명해줘."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

If you have an A100/H100/H200 runtime, bf16 loading can be used instead of 4-bit:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

### Prompt Format

The model follows the LFM2 chat-template style. Use the tokenizer chat template when possible. The CPT corpus preserves these special tokens for chat and tool-use records:

- `<|startoftext|>`
- `<|im_start|>`
- `<|im_end|>`
- roles: `system`, `user`, `assistant`, `tool`

References:

- LFM text generation: https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting
- LFM chat template: https://docs.liquid.ai/lfm/key-concepts/chat-template
- LFM tool use: https://docs.liquid.ai/lfm/key-concepts/tool-use

## Training Configuration

- Base model: [`LiquidAI/LFM2.5-8B-A1B`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- Method: full-parameter continued pretraining, not LoRA
- Framework: Unsloth + TRL `SFTTrainer`
- Hardware target: 8x NVIDIA H200
- Context length: 8192
- Precision: bf16 when supported
- Optimizer: `adamw_8bit`
- GPUs: 8
- Per-device batch size: 2
- Gradient accumulation steps: 4
- Effective batch: 64 sequences/update
- Maximum tokens/update: 524,288
- Learning rate: 2e-5
- Schedule: 1 epoch over the prepared full corpus
- `max_steps`: -1
- Checkpoint interval: 1,000 steps
- Checkpoint retention: 4 latest checkpoints plus final model

Completed run:

- Estimated tokens: 6,492,697,020
- Raw estimated steps before packing: 12,384
- Actual packed trainer steps: 10,196
- Train runtime: about 9h 38m
- Train samples/sec: 18.81
- Train steps/sec: 0.294
- Final logged train loss: 0.712
- Final checkpoint source: `checkpoint-10196`
- Final model integrity check: `model.safetensors` opens successfully with 2,302 tensors

Note: the distributed `torchrun` process reached step `10196/10196` and wrote `checkpoint-10196`. A SIGSEGV occurred during the extra post-train `trainer.save_model(final_full)` write, leaving the initial `final_full/model.safetensors` incomplete. The published `final_full` was rebuilt from the verified `checkpoint-10196` inference files.

## Data Mix

Prepared full mix:

`/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl`

Statistics:

- Rows after global deduplication: 4,622,971
- Characters: 11,581,567,658
- Estimated tokens: 6,492,697,020
- Raw estimated training steps: 12,384 at effective batch 64 and sequence length 8192
- Actual packed trainer steps: 10,196

Per-source rows:

| Source | Rows |
| --- | ---: |
| `kowiki_raw_full_20260524` | 611,403 |
| `bcai_finance_kor_hrm_20260524` | 1,861,531 |
| `korean_legal_raw_full_20260523` | 227,687 |
| `korean_legal_tasks_full_20260524` | 1,383,340 |
| `korean_admrule_precedent_raw_full_20260524` | 203,477 |
| `ko_legal_source_agent_sft_20260621` | 5,999 |
| `ko_legal_rag_agent_sft_round15_v2` | 749 |
| `current_law_bar_json_answer_sft_20260621` | 2,000 |
| `lfm25_terminal_toolbench_hrm_turns_v1` | 326,785 |

Raw Korean wiki/legal/finance documents are kept as plain completion text for CPT. Instruction, legal RAG, and terminal/tool-use examples are converted to LFM ChatML-style text.

## Legal Data Attribution

Legal-domain data is attributed to the public Legalize-KR ecosystem and related Korean legal source corpora used in the local CPT mix.

Legalize-KR links:

- Organization: https://github.com/legalize-kr
- Korean statutes repository: https://github.com/legalize-kr/legalize-kr
- Korean court precedent repository: https://github.com/legalize-kr/precedent-kr
- Korean administrative rules repository: https://github.com/legalize-kr/admrule-kr
- Korean local ordinance repository: https://github.com/legalize-kr/ordinance-kr
- Data collection/conversion pipeline: https://github.com/legalize-kr/legalize-pipeline
- Legalize-KR website: https://legalize.kr
- Original public legal source: https://www.law.go.kr

The Legalize-KR organization describes its project as converting Korean statutes, precedents, administrative rules, and local ordinances into Markdown and Git history. Its README states that source data is obtained from the National Law Information Center OpenAPI and transformed into Git repositories. Long-term reproducibility should pin a snapshot or release where possible because Legalize-KR notes that Git history can be reconstructed when parsing and normalization rules improve.

Recommended attribution format:

- Statutes: cite `legalize-kr/legalize-kr`, the Markdown path such as `kr/{statute-name}/{statute-type}.md`, and stable metadata fields such as `법령ID`, `법령MST`, promulgation date, effective date, and the `출처` URL from `law.go.kr`.
- Precedents: cite `legalize-kr/precedent-kr`, the Markdown path such as `{case-type}/{court-level}/{court}_{decision-date}_{case-number}.md`, and stable identifiers such as `판례일련번호`, court name, decision date, and case number.
- Administrative rules: cite `legalize-kr/admrule-kr`, the Markdown path such as `{agency-path}/{rule-type}/{rule-name}/본문.md`, plus rule serial number or issuing number when available.
- Local ordinances: cite `legalize-kr/ordinance-kr`, the Markdown path such as `{province}/{city-or-office}/{ordinance-type}/{ordinance-name}/본문.md`, plus `자치법규ID`, `자치법규일련번호`, promulgation date, promulgation number, and the `출처` URL.
- Avoid using only Git commit hashes as long-term identifiers because Legalize-KR warns that repository history may be reconstructed after parser or normalization improvements.
- License note from the Legalize-KR READMEs: original legal text is Korean government public work; repository structure and metadata are MIT where specified by the repository.

Local legal sources included in this CPT run:

- `korean_legal_raw_full_20260523`
- `korean_legal_tasks_full_20260524`
- `korean_admrule_precedent_raw_full_20260524`
- `ko_legal_source_agent_sft_20260621`
- `ko_legal_rag_agent_sft_round15_v2`
- `current_law_bar_json_answer_sft_20260621`

## Korean

`LFM2.5-8B-A1B-KO-CPT-FULL`은 LoRA 어댑터가 아니라 full-parameter CPT 모델입니다. 목표는 LFM2.5-8B-A1B에 한국어 법률, 금융, 위키 지식과 터미널/도구 사용 스타일을 계속 사전학습으로 이식하는 것입니다.

목표 성능:

- 한국어 법률 문서 이해와 법률 RAG 답변
- 한국어 금융 설명과 금융 용어 처리
- 한국어 위키/일반 지식 문체
- 한국어 instruction following
- 터미널/도구 호출형 assistant 동작 보존

## 한국어 사용법

가중치 업로드 후 아래처럼 사용할 수 있습니다.

### Transformers 사용

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "너는 한국어로 정확하고 간결하게 답하는 어시스턴트다."},
    {"role": "user", "content": "상법상 이사의 충실의무를 실무 관점에서 설명해줘."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### vLLM 사용

```bash
vllm serve LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size 8
```

요청 예시:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL",
    messages=[
        {"role": "system", "content": "너는 한국어로 정확하고 간결하게 답하는 어시스턴트다."},
        {"role": "user", "content": "부동산 임대차 계약에서 보증금 반환 분쟁의 핵심 쟁점을 정리해줘."},
    ],
    temperature=0.5,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

### Colab 사용 예시

일반 Colab GPU에서는 VRAM 부족을 피하려고 4-bit 로딩부터 쓰는 것이 좋습니다.

```python
!pip install -U "transformers>=4.44" accelerate bitsandbytes sentencepiece huggingface_hub
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "너는 한국어로 정확하고 간결하게 답하는 어시스턴트다."},
    {"role": "user", "content": "한국 금융시장에서 기준금리와 채권 가격의 관계를 설명해줘."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### 권장 생성 설정

- 법률/금융 설명: `temperature=0.3-0.6`, `top_p=0.8-0.95`
- 일반 한국어 답변: `temperature=0.5-0.8`, `top_p=0.9`
- 긴 문서 요약: `max_new_tokens=1024` 이상
- 도구 사용/구조화 출력: 낮은 temperature 권장

## 한국어 학습 설정

- 베이스 모델: `LiquidAI/LFM2.5-8B-A1B`
- 방식: full-parameter CPT, LoRA 아님
- 하드웨어: NVIDIA H200 8장
- 컨텍스트 길이: 8192
- GPU당 batch size: 2
- gradient accumulation: 4
- effective batch: 64 sequences/update
- update당 최대 token: 524,288
- learning rate: 2e-5
- epoch: 1
- `max_steps`: -1
- 저장 간격: 1,000 steps
- checkpoint 보존: 최신 4개와 final model

학습 규모:

- 전체 row: 4,622,971
- 추정 token: 6.49B
- raw 예상 step: 12,384
- 실제 packed trainer step: 10,196
- 실제 train runtime: 약 9시간 38분
- 최종 train loss: 0.712
- 최종 weight 출처: 무결성 검사를 통과한 `checkpoint-10196`

### 한국어 법률 데이터 출처

법률 도메인 데이터 출처는 Legalize-KR 생태계와 로컬 한국 법률 corpus를 명시한다.

- Legalize-KR 조직: https://github.com/legalize-kr
- 대한민국 법령: https://github.com/legalize-kr/legalize-kr
- 대한민국 판례: https://github.com/legalize-kr/precedent-kr
- 대한민국 행정규칙: https://github.com/legalize-kr/admrule-kr
- 대한민국 자치법규: https://github.com/legalize-kr/ordinance-kr
- 수집/변환 파이프라인: https://github.com/legalize-kr/legalize-pipeline
- Legalize-KR 웹사이트: https://legalize.kr
- 원천 공공 법령 출처: https://www.law.go.kr

Legalize-KR은 법령/판례/행정규칙/자치법규를 Markdown과 Git 이력으로 관리하는 공개 프로젝트다. 조직 README 기준 원천 데이터는 국가법령정보센터 OpenAPI에서 가져오며, 파싱과 정규화 규칙이 개선되면 Git 이력이 재구성될 수 있으므로 장기 재현에는 snapshot 또는 release 고정이 필요하다.

출처 표기 방식:

- 법령: `legalize-kr/legalize-kr` 저장소, `kr/{법령명}/{법령구분}.md` 경로, `법령ID`, `법령MST`, `공포일자`, `시행일자`, `출처` URL을 함께 적는다.
- 판례: `legalize-kr/precedent-kr` 저장소, `{사건종류}/{법원등급}/{법원명}_{선고일자}_{사건번호}.md` 경로, `판례일련번호`, 법원명, 선고일자, 사건번호를 함께 적는다.
- 행정규칙: `legalize-kr/admrule-kr` 저장소, `{기관경로}/{행정규칙종류}/{행정규칙명}/본문.md` 경로, 행정규칙일련번호 또는 발령번호를 함께 적는다.
- 자치법규: `legalize-kr/ordinance-kr` 저장소, `{광역}/{기초 또는 _본청 또는 _교육청}/{자치법규종류}/{자치법규명}/본문.md` 경로, `자치법규ID`, `자치법규일련번호`, `공포일자`, `공포번호`, `출처` URL을 함께 적는다.
- commit hash만 장기 출처로 쓰지 않는다. Legalize-KR README는 파서/정규화 개선 시 저장소 history가 재구성될 수 있다고 안내한다.
- Legalize-KR README 기준 원문은 대한민국 정부 공공저작물이고, 저장소 구조와 메타데이터는 저장소별 MIT 표기를 따른다.

## Evaluation Plan

## Current vLLM Smoke Check

This is not a benchmark score. It verifies that both the base model and the CPT model load and generate with vLLM tensor parallelism.

- Date: 2026-06-28
- vLLM environment: local `.vllm-lfm-cu12`, vLLM `0.19.1`, Torch `2.10.0+cu128`
- Tensor parallel size: 8
- Max model length: 8192
- Base model smoke: passed model load and generation
- CPT model smoke: passed model load and generation
- Smoke result path: `/home/work/.data/lfm2_ko_cpt/evals/20260628_1052_smoke_clean_vllm_smoke`
- CPT checks passed: Korean legal, Korean finance, tool-call format, English instruction smoke
- CPT wiki smoke note: the answer was relevant, but the simple keyword check expected the literal word `요약`, so that specific automatic check is false.


## Current vLLM Benchmark Results

Evaluation uses EleutherAI lm-evaluation-harness with vLLM tensor parallelism. The IFEval run below is the full 541-prompt public task, not a limited smoke sample.

- Date: 2026-06-28
- Task: `ifeval`
- Runner: `lm_eval==0.4.11`, `vllm==0.19.1`, Torch `2.10.0+cu128`
- Tensor parallel size: 8
- Max model length: 8192
- Result path: `/home/work/.data/lfm2_ko_cpt/evals/20260628_022743_ifeval_full_vllm_vllm_matrix`

| Metric | LiquidAI/LFM2.5-8B-A1B | LFM2.5-8B-A1B-KO-CPT-FULL | Delta | Relative |
|---|---:|---:|---:|---:|
| prompt_level_strict_acc | 0.2810 | 0.2976 | +0.0166 | +5.91% |
| prompt_level_loose_acc | 0.2921 | 0.3216 | +0.0295 | +10.10% |
| inst_level_strict_acc | 0.4221 | 0.4365 | +0.0144 | +3.41% |
| inst_level_loose_acc | 0.4341 | 0.4628 | +0.0287 | +6.61% |

GSM8K 5-shot `LIMIT=200` limited regression check:

| Metric | LiquidAI/LFM2.5-8B-A1B | LFM2.5-8B-A1B-KO-CPT-FULL | Delta | Relative |
|---|---:|---:|---:|---:|
| exact_match strict-match | 0.2600 | 0.4250 | +0.1650 | +63.46% |
| exact_match flexible-extract | 0.4250 | 0.4950 | +0.0700 | +16.47% |

Global MMLU Korean `LIMIT=500` limited check:

| Metric | LiquidAI/LFM2.5-8B-A1B | LFM2.5-8B-A1B-KO-CPT-FULL | Delta | Relative |
|---|---:|---:|---:|---:|
| global_mmlu_full_ko acc | 0.2803 | 0.3086 | +0.0283 | +10.10% |
| humanities acc | 0.2784 | 0.3022 | +0.0238 | +8.55% |
| other acc | 0.2914 | 0.3385 | +0.0471 | +16.16% |
| social_sciences acc | 0.2911 | 0.3404 | +0.0493 | +16.93% |
| stem acc | 0.2623 | 0.2591 | -0.0032 | -1.22% |

Note: GSM8K and Global MMLU Korean above are limited runs and should be treated as early regression checks, not final public benchmark scores. Additional vLLM evaluations are running with one task per GPU.

Additional vLLM checks:

| Task | Metric | LiquidAI/LFM2.5-8B-A1B | LFM2.5-8B-A1B-KO-CPT-FULL | Delta | Relative | Note |
|---|---|---:|---:|---:|---:|---|
| `arc_challenge` `LIMIT=500` | acc | 0.3600 | 0.4020 | +0.0420 | +11.67% | limited |
| `arc_challenge` `LIMIT=500` | acc_norm | 0.3760 | 0.4140 | +0.0380 | +10.11% | limited |
| `gsm8k` full 5-shot | exact_match strict | 0.2472 | 0.4617 | +0.2145 | +86.77% | full task |
| `gsm8k` full 5-shot | exact_match flexible | 0.4845 | 0.5701 | +0.0856 | +17.67% | full task |
| `mmlu_pro_economics` `LIMIT=500` | exact_match | 0.4420 | 0.4900 | +0.0480 | +10.86% | limited |
| `mmlu_pro_law` `LIMIT=500` | exact_match | 0.1840 | 0.1240 | -0.0600 | -32.61% | limited |
| `mmlu_prox_lite_ko` `LIMIT=500` | exact_match | 0.2585 | 0.1667 | -0.0918 | -35.51% | limited |
| `global_mmlu_full_ko_professional_law` full | acc | 0.2581 | 0.2595 | +0.0014 | +0.54% | full subject |
| `global_mmlu_full_ko_professional_accounting` full | acc | 0.2730 | 0.2340 | -0.0390 | -14.29% | full subject |
| `global_mmlu_full_ko_high_school_macroeconomics` full | acc | 0.2436 | 0.2846 | +0.0410 | +16.83% | full subject |

The limited checks are useful for regression tracking, but they should not be read as final leaderboard-quality numbers. The model improves strongly on several reasoning and instruction-following checks, while law-focused MMLU-Pro and MMLU-ProX-lite-ko need targeted remediation.

## Public Benchmark Plan

Primary public Korean benchmarks:

- KMMLU: https://huggingface.co/datasets/HAERAE-HUB/KMMLU
- KMMLU-Pro: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro
- KMMLU-Redux: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Redux
- Ko-IFEval: https://huggingface.co/datasets/davidkim205/ko-ifeval

Secondary checks:

- Korean legal RAG holdout
- Korean finance explanation holdout
- Korean wiki QA/summarization holdout
- Terminal/tool-use smoke tests

Benchmark results will be added after vLLM base-vs-CPT evaluation.
