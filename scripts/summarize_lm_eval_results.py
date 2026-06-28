#!/usr/bin/env python
"""Flatten lm-eval result JSON files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iter_result_files(root: Path) -> list[Path]:
    candidates = []
    for pattern in ("**/results*.json", "**/*results*.json"):
        candidates.extend(root.glob(pattern))
    return sorted({path.resolve() for path in candidates if path.is_file()})


def metric_value(metrics: dict[str, Any]) -> tuple[str, Any] | None:
    preferred = (
        "exact_match,strict-match",
        "exact_match",
        "acc,none",
        "acc",
        "acc_norm,none",
        "acc_norm",
        "f1,none",
        "f1",
    )
    for key in preferred:
        if key in metrics and not key.endswith("_stderr"):
            return key, metrics[key]
    for key, value in metrics.items():
        if key.endswith("_stderr") or key == "alias":
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_file in iter_result_files(root):
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        metadata = payload.get("metadata", {})
        model_name = (
            metadata.get("model_name")
            or config.get("model_name")
            or config.get("model")
            or result_file.parent.name
        )
        for task, metrics in payload.get("results", {}).items():
            selected = metric_value(metrics)
            if selected is None:
                continue
            metric, value = selected
            rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "result_file": str(result_file),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "task", "metric", "value", "result_file"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# lm-eval summary",
        "",
        "| model | task | metric | value |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        value = row["value"]
        if isinstance(value, float):
            value_text = f"{value:.4f}"
        else:
            value_text = str(value)
        lines.append(
            f"| {row['model']} | {row['task']} | {row['metric']} | {value_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    rows = collect(args.result_root)
    if not rows:
        raise SystemExit(f"no lm-eval result rows found under {args.result_root}")

    csv_path = args.csv or args.result_root / "summary.csv"
    md_path = args.markdown or args.result_root / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
