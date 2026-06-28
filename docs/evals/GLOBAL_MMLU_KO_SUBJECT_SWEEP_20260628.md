# Global MMLU Korean Subject Sweep

This file tracks vLLM/lm-eval base-vs-CPT comparisons for completed
`global_mmlu_full_ko_*` subjects.

- Date: 2026-06-28
- Base: `LiquidAI/LFM2.5-8B-A1B`
- CPT: `/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full`
- Runner: vLLM, tensor parallel size 1 per GPU, one task per GPU
- Result roots:
  - `/home/work/.data/lfm2_ko_cpt/evals/20260628_034500_parallel_1gpu_refill6_*_vllm_matrix`
  - `/home/work/.data/lfm2_ko_cpt/evals/20260628_034800_parallel_1gpu_queue_manual1_*_vllm_matrix`
- CSV copy: `/home/work/.data/lfm2_ko_cpt/evals/20260628_latest_global_mmlu_ko_subjects_034500_034800.csv`

Interpretation: CPT improves several Korean knowledge/history/economics slices,
but it regresses multiple STEM/logical/legal option-choice slices. The next
post-training pass should prioritize Korean MCQA remediation, option-label
stability, STEM multiple-choice, and law/accounting style hard negatives.

| task | metric | base | CPT | delta | relative |
|---|---|---:|---:|---:|---:|
| `global_mmlu_full_ko_astronomy` | acc,none | 0.3421 | 0.2829 | -0.0592 | -17.31% |
| `global_mmlu_full_ko_conceptual_physics` | acc,none | 0.3149 | 0.2936 | -0.0213 | -6.76% |
| `global_mmlu_full_ko_econometrics` | acc,none | 0.2632 | 0.2807 | +0.0175 | +6.67% |
| `global_mmlu_full_ko_electrical_engineering` | acc,none | 0.2759 | 0.3103 | +0.0345 | +12.50% |
| `global_mmlu_full_ko_formal_logic` | acc,none | 0.3254 | 0.2778 | -0.0476 | -14.63% |
| `global_mmlu_full_ko_high_school_biology` | acc,none | 0.2710 | 0.2871 | +0.0161 | +5.95% |
| `global_mmlu_full_ko_high_school_chemistry` | acc,none | 0.2315 | 0.1921 | -0.0394 | -17.02% |
| `global_mmlu_full_ko_high_school_physics` | acc,none | 0.2185 | 0.2185 | +0.0000 | +0.00% |
| `global_mmlu_full_ko_high_school_mathematics` | acc,none | 0.2259 | 0.2148 | -0.0111 | -4.92% |
| `global_mmlu_full_ko_high_school_statistics` | acc,none | 0.2870 | 0.1574 | -0.1296 | -45.16% |
| `global_mmlu_full_ko_high_school_microeconomics` | acc,none | 0.3025 | 0.2689 | -0.0336 | -11.11% |
| `global_mmlu_full_ko_high_school_european_history` | acc,none | 0.2788 | 0.3152 | +0.0364 | +13.04% |
| `global_mmlu_full_ko_high_school_us_history` | acc,none | 0.2892 | 0.2892 | +0.0000 | +0.00% |
| `global_mmlu_full_ko_high_school_world_history` | acc,none | 0.2911 | 0.3376 | +0.0464 | +15.94% |
| `global_mmlu_full_ko_jurisprudence` | acc,none | 0.2870 | 0.2685 | -0.0185 | -6.45% |
| `global_mmlu_full_ko_logical_fallacies` | acc,none | 0.3067 | 0.2945 | -0.0123 | -4.00% |
