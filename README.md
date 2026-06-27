# LFM2.5 Korean CPT

This folder contains the Korean CPT workflow for `LiquidAI/LFM2.5-8B-A1B`.

- Runbook: `docs/LFM2_KO_CPT_RUNBOOK_20260627.ko.md`
- Evaluation plan: `docs/LFM2_KO_EVAL_PLAN_20260627.ko.md`
- Data mix config: `configs/ko_cpt_sources_20260627.json`
- Data builder: `scripts/build_ko_cpt_mix.py`
- 8GPU LoRA CPT trainer: `scripts/train_lfm25_ko_cpt_lora.py`
- Runner: `scripts/run_lfm25_8b_ko_cpt_lora.sh`
- HF upload helper: `scripts/upload_latest_lora.py`

Default artifact root:

`/home/work/.data/lfm2_ko_cpt`
