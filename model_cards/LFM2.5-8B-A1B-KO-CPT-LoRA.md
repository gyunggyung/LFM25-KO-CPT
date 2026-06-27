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
  - lora
  - legal
  - finance
  - wiki
---

# LFM2.5-8B-A1B-KO-CPT-LoRA

Korean continued-pretraining LoRA adapter for `LiquidAI/LFM2.5-8B-A1B`.

## Goal

This adapter targets Korean-language fluency and domain behavior for:

- Korean legal text and legal RAG style responses
- Korean finance explanations
- Korean wiki/general knowledge prose
- A small amount of tool/agentic text to preserve structured assistant behavior

## Base Model

- Base: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
- LFM2.5 family: https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai
- Fine-tuning reference: https://docs.liquid.ai/lfm/fine-tuning/unsloth

## Training

- Method: Continued pre-training as text completion
- Framework: Unsloth + TRL
- Hardware target: 8x NVIDIA H200
- Context length: 8192
- LoRA rank: 64
- LoRA alpha: 128
- Target modules: all-linear
- Optimizer: adamw_8bit

## Data Mix

Local prepared corpora:

- Korean Wikipedia raw text
- Korean legal raw text
- Korean legal task data
- Korean administrative rule and precedent text
- Korean finance corpus
- Korean legal source/RAG agent data
- Small terminal/tool-agentic preservation slice

Data preparation script:

`lfm2_ko_cpt/scripts/build_ko_cpt_mix.py`

## Status

Initial training run is configured in:

`lfm2_ko_cpt/scripts/run_lfm25_8b_ko_cpt_lora.sh`

Evaluation results and merged checkpoint details will be added after the first completed run.

