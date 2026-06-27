#!/usr/bin/env python
"""Build a Korean continued-pretraining JSONL mix from local corpora.

The output schema is one JSON object per line:
{"text": "...", "source": "...", "category": "..."}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


HANGUL_RE = re.compile(r"[가-힣]")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
PARA_RE = re.compile(r"\n{3,}")

TEXT_FIELDS = (
    "text",
    "content",
    "body",
    "document",
    "passage",
    "article",
    "summary",
    "answer",
    "response",
    "output",
    "completion",
)
QUESTION_FIELDS = ("question", "query", "prompt", "instruction", "input")
TITLE_FIELDS = ("title", "name", "case_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--min-hangul-ratio", type=float, default=0.08)
    parser.add_argument("--max-docs-total", type=int, default=0)
    parser.add_argument("--dedupe", action="store_true", default=True)
    return parser.parse_args()


def clean_text(text: str, max_chars: int) -> str:
    text = text.replace("\u0000", " ").replace("\ufeff", "")
    text = text.replace("\\n", "\n") if "\\n" in text and "\n" not in text else text
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = PARA_RE.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0].strip() or text[:max_chars].strip()
    return text


def hangul_ratio(text: str) -> float:
    useful = sum(ch.isalnum() or ("\uac00" <= ch <= "\ud7a3") for ch in text)
    if useful == 0:
        return 0.0
    return len(HANGUL_RE.findall(text)) / useful


def get_first_string(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for key in fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_messages(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from") or item.get("speaker") or "turn"
        content = item.get("content") or item.get("value") or item.get("text")
        if isinstance(content, list):
            content = "\n".join(
                str(piece.get("text", "")) if isinstance(piece, dict) else str(piece)
                for piece in content
            )
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content.strip()}")
    return "\n\n".join(parts)


def extract_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""

    for key in ("messages", "conversations", "dialogue", "turns"):
        text = extract_messages(row.get(key))
        if text:
            return text

    title = get_first_string(row, TITLE_FIELDS)
    question = get_first_string(row, QUESTION_FIELDS)
    answer = get_first_string(row, TEXT_FIELDS)

    if question and answer and question != answer:
        prefix = f"{title}\n\n" if title else ""
        return f"{prefix}질문:\n{question}\n\n답변:\n{answer}"

    if answer:
        return f"{title}\n\n{answer}" if title and title not in answer[:200] else answer

    strings: list[str] = []
    for key, value in row.items():
        if key.startswith("_") or key in {"id", "url", "source", "meta", "metadata"}:
            continue
        if isinstance(value, str) and len(value.strip()) >= 80:
            strings.append(value.strip())
    return "\n\n".join(strings[:4])


def source_iter(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, line


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output) if args.stats_output else output_path.with_suffix(".stats.json")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    totals = Counter()
    per_source: dict[str, Counter] = {}
    written_total = 0

    with output_path.open("w", encoding="utf-8") as out:
        for source in config["sources"]:
            source_id = source["id"]
            category = source.get("category", "unknown")
            path = Path(source["path"])
            max_docs = int(source.get("max_docs") or 0)
            stats = Counter()
            per_source[source_id] = stats

            if not path.exists():
                stats["missing"] += 1
                continue

            for _, row in source_iter(path):
                stats["read"] += 1
                text = clean_text(extract_text(row), args.max_chars)
                if len(text) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                ratio = hangul_ratio(text)
                if ratio < args.min_hangul_ratio:
                    stats["low_hangul_ratio"] += 1
                    continue
                digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                if args.dedupe and digest in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(digest)

                record = {"text": text, "source": source_id, "category": category}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats["chars"] += len(text)
                totals["written"] += 1
                totals["chars"] += len(text)
                written_total += 1

                if max_docs and stats["written"] >= max_docs:
                    break
                if args.max_docs_total and written_total >= args.max_docs_total:
                    break

            if args.max_docs_total and written_total >= args.max_docs_total:
                break

    stats_doc = {
        "config": str(config_path),
        "output": str(output_path),
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "min_hangul_ratio": args.min_hangul_ratio,
        "totals": dict(totals),
        "per_source": {key: dict(value) for key, value in per_source.items()},
    }
    stats_path.write_text(json.dumps(stats_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats_doc, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

