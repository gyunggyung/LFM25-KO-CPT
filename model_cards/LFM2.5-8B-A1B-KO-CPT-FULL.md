---
license: other
base_model: LiquidAI/LFM2.5-8B-A1B
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

Korean continued-pretraining project for `LiquidAI/LFM2.5-8B-A1B`.

This repository is being prepared as the full-parameter CPT checkpoint for a Korean-specialized LFM2.5 model. The immediate target is Korean legal, finance, wiki, and terminal/tool-use behavior while preserving the base model's general chat and instruction-following behavior.

## Base Model

- Base model: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
- LFM2.5 release notes: https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai
- LFM text generation docs: https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting
- LFM chat template docs: https://docs.liquid.ai/lfm/key-concepts/chat-template
- LFM tool-use docs: https://docs.liquid.ai/lfm/key-concepts/tool-use
- Liquid Unsloth fine-tuning docs: https://docs.liquid.ai/lfm/fine-tuning/unsloth

## Training Plan

- Method: full-parameter continued pretraining, not LoRA
- Framework: Unsloth + TRL `SFTTrainer`
- Hardware target: 8x NVIDIA H200
- Context length: 8192
- Precision: bf16 when supported
- Optimizer: `adamw_8bit`
- Batch: 8 GPUs x per-device batch 2 x grad accumulation 4
- Effective batch: 64 sequences/update
- Maximum tokens/update: 524,288
- Schedule: 1 epoch over the prepared full corpus
- Checkpointing: every 1,000 steps
- Checkpoint retention: 4 latest checkpoints plus final model

## Prepared Corpus

Prepared full mix:

`/home/work/.data/lfm2_ko_cpt/datasets/ko_cpt_mix_full_lfmstyle_20260627.jsonl`

Statistics:

- Rows after global deduplication: 4,622,971
- Characters: 11,581,567,658
- Estimated tokens: 6,492,697,020
- Estimated training steps: 12,384 at effective batch 64 and sequence length 8192

Per-source rows:

- `kowiki_raw_full_20260524`: 611,403
- `bcai_finance_kor_hrm_20260524`: 1,861,531
- `korean_legal_raw_full_20260523`: 227,687
- `korean_legal_tasks_full_20260524`: 1,383,340
- `korean_admrule_precedent_raw_full_20260524`: 203,477
- `ko_legal_source_agent_sft_20260621`: 5,999
- `ko_legal_rag_agent_sft_round15_v2`: 749
- `current_law_bar_json_answer_sft_20260621`: 2,000
- `lfm25_terminal_toolbench_hrm_turns_v1`: 326,785

## Formatting

Raw Korean wiki/legal/finance documents are kept as plain completion text for CPT. Instruction, legal RAG, and terminal/tool-use examples are converted to LFM ChatML-style text with `<|startoftext|>`, `<|im_start|>`, and `<|im_end|>` role blocks.

## Expected Runtime

At the observed practice speed of roughly 3.35-3.45 seconds per step, 12,384 steps would take about 11.5-12 hours. A safer estimate including full-corpus packing/tokenization overhead is 12-15.5 hours.

## Evaluation Plan

Primary public Korean evaluations:

- KMMLU: https://huggingface.co/datasets/HAERAE-HUB/KMMLU
- KMMLU-Pro: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro
- KMMLU-Redux: https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Redux
- Ko-IFEval: https://huggingface.co/datasets/davidkim205/ko-ifeval

Secondary checks:

- Korean legal RAG holdout
- Korean finance explanation holdout
- Korean wiki QA/summarization holdout
- Terminal/tool-use smoke tests

## Status

The full corpus is prepared. Training artifacts and benchmark results will be uploaded after the first completed full CPT run.
