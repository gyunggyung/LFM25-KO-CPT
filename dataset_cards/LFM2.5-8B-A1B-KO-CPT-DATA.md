---
license: other
language:
  - ko
  - en
tags:
  - korean
  - continued-pretraining
  - legal
  - finance
  - wiki
  - tool-use
  - lfm2.5
pretty_name: LFM2.5-8B-A1B Korean CPT Data
task_categories:
  - text-generation
---

# LFM2.5-8B-A1B Korean CPT Data

Prepared Korean continued-pretraining data for `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`.

## Files

- `data/ko_cpt_mix_full_lfmstyle_20260627.jsonl`: prepared full CPT corpus with one JSON object per line and a `text` field
- `metadata/ko_cpt_mix_full_lfmstyle_20260627.stats.json`: corpus statistics
- `metadata/ko_cpt_mix_full_lfmstyle_20260627.stats.json.full_report.json`: per-source preprocessing report
- `metadata/ko_cpt_sources_full_20260627.json`: source configuration
- `tokenized/tokenized_lfm25_8k_20260628/`: packed 8192-token Parquet blocks will be uploaded after generation

## Corpus Statistics

- Rows after global deduplication: 4,622,971
- Characters: 11,581,567,658
- Estimated tokens: 6,492,697,020
- Target tokenizer/model: `LiquidAI/LFM2.5-8B-A1B`
- Target sequence length: 8192
- Estimated packed training steps at effective batch 64: 12,384
- Tokenized export format: Parquet shards with `input_ids` and `length`

## Format

Raw Korean wiki, legal, and finance documents are kept as plain completion text. Instruction, legal RAG, and terminal/tool-use examples are converted to LFM chat-style text using:

- `<|startoftext|>`
- `<|im_start|>`
- `<|im_end|>`
- roles: `system`, `user`, `assistant`, `tool`

Each row:

```json
{"text": "...", "source": "...", "category": "..."}
```

## Source Mix

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

## Legal Data Attribution

Legal-domain data is attributed to Legalize-KR and related Korean legal corpora.

Legalize-KR links:

- Organization: https://github.com/legalize-kr
- Korean statutes: https://github.com/legalize-kr/legalize-kr
- Korean court precedents: https://github.com/legalize-kr/precedent-kr
- Korean administrative rules: https://github.com/legalize-kr/admrule-kr
- Korean local ordinances: https://github.com/legalize-kr/ordinance-kr
- Collection/conversion pipeline: https://github.com/legalize-kr/legalize-pipeline
- Website: https://legalize.kr
- Public source: https://www.law.go.kr

Recommended citation:

- Statutes: include repository, Markdown path, `법령ID`, `법령MST`, promulgation date, effective date, and `law.go.kr` source URL.
- Precedents: include repository, Markdown path, `판례일련번호`, court name, decision date, and case number.
- Administrative rules: include repository, Markdown path, rule serial number or issuing number when available.
- Local ordinances: include repository, Markdown path, `자치법규ID`, `자치법규일련번호`, promulgation date, promulgation number, and source URL.
- Do not use only Git commit hashes as long-term identifiers because Legalize-KR notes that history may be reconstructed after parser or normalization improvements.

## Korean Notes

이 데이터셋은 `LiquidAI/LFM2.5-8B-A1B`의 한국어 CPT를 위해 만든 공개용 준비 corpus입니다. 법률 데이터 출처는 Legalize-KR 및 관련 로컬 한국 법률 corpus를 명시합니다. Legalize-KR README 기준 법령/판례/행정규칙/자치법규 원문은 국가법령정보센터 OpenAPI에서 가져오며, 원문은 대한민국 정부 공공저작물로 안내되어 있습니다. 재현성을 위해서는 commit hash만이 아니라 법령ID, 판례일련번호, 자치법규ID, 날짜, law.go.kr URL을 함께 보존하는 것이 좋습니다.

## Intended Use

This dataset is intended for continued pretraining and controlled evaluation of Korean language model behavior. It should not be used as legal advice, financial advice, or a substitute for authoritative legal texts.
