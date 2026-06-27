#!/usr/bin/env python
"""Parallel Korean CPT preprocessing with LFM2.5-style chat formatting."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
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
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--min-hangul-ratio", type=float, default=0.06)
    parser.add_argument("--max-docs-total", type=int, default=0)
    parser.add_argument("--force", action="store_true")
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


def first_string(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for key in fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, dict):
                text = piece.get("text") or piece.get("content") or piece.get("value")
                if isinstance(text, str):
                    parts.append(text.strip())
            elif isinstance(piece, str):
                parts.append(piece.strip())
        return "\n".join(part for part in parts if part)
    return ""


def normalize_role(role: Any) -> str:
    role_text = str(role or "user").lower()
    if role_text in {"human", "user", "instruction", "prompt"}:
        return "user"
    if role_text in {"gpt", "assistant", "bot", "model", "response", "output"}:
        return "assistant"
    if role_text in {"system", "developer", "tool"}:
        return role_text
    return "user"


def lfm_chat(messages: list[dict[str, str]]) -> str:
    parts = ["<|startoftext|>"]
    for message in messages:
        role = normalize_role(message.get("role"))
        content = clean_text(message.get("content", ""), 10_000_000)
        if not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts).strip()


def extract_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("messages", "conversations", "dialogue", "turns"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            content = content_to_text(item.get("content") or item.get("value") or item.get("text"))
            if content:
                messages.append({"role": normalize_role(item.get("role") or item.get("from") or item.get("speaker")), "content": content})
        if messages:
            return messages
    return []


def extract_record_text(row: Any, source_kind: str, max_chars: int) -> str:
    if isinstance(row, str):
        return clean_text(row, max_chars)
    if not isinstance(row, dict):
        return ""

    messages = extract_messages(row)
    if messages:
        return clean_text(lfm_chat(messages), max_chars)

    title = first_string(row, TITLE_FIELDS)
    question = first_string(row, QUESTION_FIELDS)
    answer = first_string(row, TEXT_FIELDS)

    if question and answer and question != answer:
        system = ""
        if source_kind in {"agent_text", "instruction_as_text"}:
            system = "너는 한국어로 정확하고 간결하게 답하는 LFM2.5 한국어 어시스턴트다."
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user = f"{title}\n\n{question}" if title else question
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": answer})
        return clean_text(lfm_chat(messages), max_chars)

    if answer:
        text = f"{title}\n\n{answer}" if title and title not in answer[:200] else answer
        if source_kind in {"agent_text", "instruction_as_text"}:
            return clean_text(lfm_chat([{"role": "assistant", "content": text}]), max_chars)
        return clean_text(text, max_chars)

    strings: list[str] = []
    for key, value in row.items():
        if key.startswith("_") or key in {"id", "url", "source", "meta", "metadata"}:
            continue
        if isinstance(value, str) and len(value.strip()) >= 80:
            strings.append(value.strip())
    return clean_text("\n\n".join(strings[:4]), max_chars)


def source_iter(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield line


def process_source(source: dict[str, Any], index: int, shard_dir: str, args_dict: dict[str, Any]) -> dict[str, Any]:
    source_id = source["id"]
    category = source.get("category", "unknown")
    source_kind = source.get("kind", "raw_text")
    path = Path(source["path"])
    min_chars = int(source.get("min_chars", args_dict["min_chars"]))
    max_chars = int(source.get("max_chars", args_dict["max_chars"]))
    min_hangul_ratio = float(source.get("min_hangul_ratio", args_dict["min_hangul_ratio"]))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)
    shard_path = Path(shard_dir) / f"{index:03d}_{safe_id}.jsonl"
    stats_path = shard_path.with_suffix(".stats.json")

    if shard_path.exists() and stats_path.exists() and not args_dict["force"]:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["shard_path"] = str(shard_path)
        return stats

    stats = Counter(source_id=source_id, category=category)
    seen: set[str] = set()
    tmp_path = shard_path.with_suffix(".jsonl.tmp")

    if not path.exists():
        stats["missing"] += 1
        stats_doc = dict(stats)
        stats_doc["shard_path"] = str(shard_path)
        stats_doc["filter_min_chars"] = min_chars
        stats_doc["filter_max_chars"] = max_chars
        stats_doc["filter_min_hangul_ratio"] = min_hangul_ratio
        stats_path.write_text(json.dumps(stats_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return stats_doc

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as out:
        for row in source_iter(path):
            stats["read"] += 1
            text = extract_record_text(row, source_kind, max_chars)
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            if min_hangul_ratio > 0 and hangul_ratio(text) < min_hangul_ratio:
                stats["low_hangul_ratio"] += 1
                continue
            digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            if digest in seen:
                stats["duplicate_local"] += 1
                continue
            seen.add(digest)
            out.write(json.dumps({"text": text, "source": source_id, "category": category}, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["chars"] += len(text)

    tmp_path.replace(shard_path)
    stats_doc = dict(stats)
    stats_doc["shard_path"] = str(shard_path)
    stats_doc["filter_min_chars"] = min_chars
    stats_doc["filter_max_chars"] = max_chars
    stats_doc["filter_min_hangul_ratio"] = min_hangul_ratio
    stats_path.write_text(json.dumps(stats_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats_doc


def merge_shards(shard_paths: list[Path], output: Path, stats_output: Path, max_docs_total: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(".jsonl.tmp")
    seen: set[str] = set()
    totals = Counter()
    per_source = Counter()

    with tmp_output.open("w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        totals["bad_lines"] += 1
                        continue
                    text = row.get("text", "")
                    if not isinstance(text, str):
                        continue
                    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                    if digest in seen:
                        totals["duplicate_global"] += 1
                        continue
                    seen.add(digest)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    totals["written"] += 1
                    totals["chars"] += len(text)
                    per_source[row.get("source", "unknown")] += 1
                    if max_docs_total and totals["written"] >= max_docs_total:
                        break
            if max_docs_total and totals["written"] >= max_docs_total:
                break

    tmp_output.replace(output)
    stats = {
        "output": str(output),
        "totals": dict(totals),
        "per_source_written": dict(per_source),
        "estimated_tokens_by_1_78_chars_per_token": int(totals["chars"] / 1.7837837837837838),
        "estimated_billion_tokens": totals["chars"] / 1.7837837837837838 / 1e9,
    }
    stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    stats_output = Path(args.stats_output) if args.stats_output else Path(args.output).with_suffix(".stats.json")

    args_dict = {
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "min_hangul_ratio": args.min_hangul_ratio,
        "force": args.force,
    }

    results: list[dict[str, Any]] = []
    sources = list(config["sources"])
    with futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        submitted = [
            pool.submit(process_source, source, index, str(shard_dir), args_dict)
            for index, source in enumerate(sources)
        ]
        for future in futures.as_completed(submitted):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed_source": result.get("source_id"), "written": result.get("written", 0), "chars": result.get("chars", 0)}, ensure_ascii=False), flush=True)

    shard_paths = [
        Path(result["shard_path"])
        for result in sorted(results, key=lambda item: item.get("shard_path", ""))
        if result.get("shard_path")
    ]
    merged = merge_shards(shard_paths, Path(args.output), stats_output, args.max_docs_total)
    report = {
        "config": args.config,
        "shard_dir": str(shard_dir),
        "source_stats": results,
        "merged": merged,
    }
    Path(str(stats_output) + ".full_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
