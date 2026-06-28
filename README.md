# LFM2.5 Korean CPT

This folder contains the Korean CPT workflow for `LiquidAI/LFM2.5-8B-A1B`.

- Runbook: `docs/LFM2_KO_CPT_RUNBOOK_20260627.ko.md`
- Current status / handoff: `docs/CURRENT_STATUS_20260627.ko.md`
- Evaluation plan: `docs/LFM2_KO_EVAL_PLAN_20260627.ko.md`
- vLLM evaluation runbook: `docs/LFM2_KO_VLLM_EVAL_RUNBOOK_20260628.ko.md`
- Data mix config: `configs/ko_cpt_sources_20260627.json`
- Data builder: `scripts/build_ko_cpt_mix.py`
- Parallel LFM-style full data builder: `scripts/build_ko_cpt_mix_parallel.py`
- Public dataset card: `dataset_cards/LFM2.5-8B-A1B-KO-CPT-DATA.md`
- 8GPU LoRA CPT trainer: `scripts/train_lfm25_ko_cpt_lora.py`
- 8GPU full-parameter CPT trainer: `scripts/train_lfm25_ko_cpt_full.py`
- Runner: `scripts/run_lfm25_8b_ko_cpt_lora.sh`
- Full FT runner: `scripts/run_lfm25_8b_ko_cpt_full.sh`
- Watch full preprocess then train: `scripts/watch_full_preprocess_then_train.sh`
- Status: `scripts/status_lfm2_ko_cpt.sh`
- HF upload helper: `scripts/upload_latest_lora.py`
- HF dataset upload helper: `scripts/upload_cpt_dataset.py`
- Packed tokenized export helper: `scripts/export_lfm25_tokenized_dataset.py`
- vLLM lm-eval runner: `scripts/run_lfm2_ko_vllm_lm_eval.sh`
- vLLM base-vs-CPT eval matrix: `scripts/run_lfm2_ko_eval_matrix.sh`
- vLLM smoke runner: `scripts/run_lfm2_ko_vllm_smoke.sh`
- lm-eval result summarizer: `scripts/summarize_lm_eval_results.py`

Hugging Face:

- Model repo: `https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`
- Dataset repo: `https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-DATA`

Default artifact root:

`/home/work/.data/lfm2_ko_cpt`

Quick status:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal
bash lfm2_ko_cpt/scripts/status_lfm2_ko_cpt.sh
```
