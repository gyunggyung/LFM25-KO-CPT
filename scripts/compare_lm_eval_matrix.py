#!/usr/bin/env python
"""Build base-vs-CPT comparison tables from lm-eval matrix result dirs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASE_HINTS = ("LiquidAI_LFM2.5-8B-A1B_base", "LiquidAI__LFM2.5-8B-A1B")
CPT_HINTS = ("LLM-OS-Models_LFM2.5-8B-A1B-KO-CPT-FULL", "final_full")


def classify_model(path: Path) -> str | None:
    text = str(path)
    if any(hint in text for hint in CPT_HINTS):
        return "cpt"
    if any(hint in text for hint in BASE_HINTS):
        return "base"
    return None


def metric_values(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    preferred = (
        "acc,none",
        "acc",
        "acc_norm,none",
        "acc_norm",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,get-answer",
        "exact_match",
        "prompt_level_loose_acc,none",
        "prompt_level_loose_acc",
        "inst_level_loose_acc,none",
        "inst_level_loose_acc",
    )
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and key not in seen:
            out.append((key, float(value)))
            seen.add(key)
    for key, value in metrics.items():
        if key in seen or key == "alias" or "_stderr" in key:
            continue
        if isinstance(value, (int, float)):
            out.append((key, float(value)))
            seen.add(key)
    return out


def collect(root: Path) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for result_file in sorted(root.glob("**/results*.json")):
        model = classify_model(result_file)
        if model is None:
            continue
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        for task, metrics in payload.get("results", {}).items():
            for metric, value in metric_values(metrics):
                key = (task, metric)
                row = rows.setdefault(
                    key,
                    {
                        "task": task,
                        "metric": metric,
                        "base": None,
                        "cpt": None,
                        "base_result_file": "",
                        "cpt_result_file": "",
                    },
                )
                row[model] = value
                row[f"{model}_result_file"] = str(result_file)

    completed = []
    for row in rows.values():
        base = row["base"]
        cpt = row["cpt"]
        if base is None or cpt is None:
            continue
        delta = cpt - base
        relative = (delta / base * 100.0) if base else 0.0
        row["delta"] = delta
        row["relative_pct"] = relative
        completed.append(row)
    return sorted(completed, key=lambda item: (item["task"], item["metric"]))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# lm-eval Base vs CPT Comparison",
        "",
        "| task | metric | base | CPT | delta | relative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['task']}` | {row['metric']} | {fmt(row['base'])} | "
            f"{fmt(row['cpt'])} | {row['delta']:+.4f} | {row['relative_pct']:+.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "metric",
        "base",
        "cpt",
        "delta",
        "relative_pct",
        "base_result_file",
        "cpt_result_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_roots", type=Path, nargs="+")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for root in args.result_roots:
        rows.extend(collect(root))
    if not rows:
        raise SystemExit("no complete base-vs-CPT rows found")

    if args.csv:
        write_csv(rows, args.csv)
    if args.markdown:
        write_markdown(rows, args.markdown)

    for row in rows:
        print(
            f"| `{row['task']}` | {row['metric']} | {fmt(row['base'])} | "
            f"{fmt(row['cpt'])} | {row['delta']:+.4f} | {row['relative_pct']:+.2f}% |"
        )


if __name__ == "__main__":
    main()
