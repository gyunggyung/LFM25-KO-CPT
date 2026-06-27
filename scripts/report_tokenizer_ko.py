#!/usr/bin/env python
"""Report rough Korean tokenization density for an LFM tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


SAMPLES = [
    "대한민국 민법상 계약의 성립 요건과 의사표시의 효력 발생 시점을 설명하라.",
    "금융시장에서 기준금리 인상은 채권 가격, 환율, 기업 조달비용에 어떤 영향을 미치는가?",
    "한국어 위키 문서의 문장 구조와 전문 용어를 유지하면서 핵심 내용을 요약한다.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--sample-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    texts = list(SAMPLES)
    if args.sample_jsonl:
        with Path(args.sample_jsonl).open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if len(texts) >= args.limit:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text[:2000])
    total_chars = 0
    total_tokens = 0
    rows = []
    for text in texts:
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        char_count = len(text)
        total_chars += char_count
        total_tokens += token_count
        rows.append({"chars": char_count, "tokens": token_count, "chars_per_token": char_count / max(token_count, 1)})
    report = {
        "model_path": args.model_path,
        "vocab_size": len(tokenizer),
        "samples": len(texts),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "avg_chars_per_token": total_chars / max(total_tokens, 1),
        "first_rows": rows[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

