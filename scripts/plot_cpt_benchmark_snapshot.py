#!/usr/bin/env python3
"""Create the Base vs KO-CPT benchmark visualization used in the model card."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets"
OUT_PATH = OUT_DIR / "cpt_benchmark_snapshot.png"


@dataclass(frozen=True)
class Score:
    name: str
    group: str
    base: float
    cpt: float

    @property
    def delta(self) -> float:
        return self.cpt - self.base

    @property
    def relative(self) -> float:
        if self.base == 0:
            return 0.0
        return self.delta / self.base * 100.0


SCORES = [
    Score("GSM8K\nflex", "General reasoning", 0.4845, 0.5701),
    Score("BoolQ", "General reasoning", 0.6544, 0.7902),
    Score("ARC-Challenge", "General reasoning", 0.3771, 0.4241),
    Score("PIQA", "General reasoning", 0.7203, 0.7476),
    Score("IFEval\nprompt", "Instruction", 0.2921, 0.3216),
    Score("Leaderboard\nIFEval", "Instruction", 0.2902, 0.3457),
    Score("G-MMLU KO\nbusiness ethics", "Korean knowledge", 0.2100, 0.4500),
    Score("G-MMLU KO\nsociology", "Korean knowledge", 0.2886, 0.4776),
    Score("G-MMLU KO\ncomputer security", "Korean knowledge", 0.2900, 0.4500),
    Score("G-MMLU KO\nworld religions", "Korean knowledge", 0.3450, 0.4854),
    Score("G-MMLU KO\nmanagement", "Korean knowledge", 0.3107, 0.4369),
    Score("G-MMLU KO\nmedical genetics", "Korean knowledge", 0.2900, 0.3800),
    Score("MMLU-Pro\neconomics", "Reasoning", 0.4277, 0.4704),
    Score("KMMLU hard\nHUMSS", "Korean MCQA", 0.2533, 0.2675),
    Score("MMLU-ProX\nLite KO", "Regression", 0.2585, 0.1667),
    Score("KMMLU hard", "Regression", 0.2015, 0.1720),
    Score("G-MMLU KO\nstatistics", "Regression", 0.2870, 0.1574),
    Score("MMLU-Pro\nlaw", "Regression", 0.1840, 0.1240),
]


GROUP_COLORS = {
    "General reasoning": "#49B6FF",
    "Instruction": "#7C5CFF",
    "Korean knowledge": "#11C5A5",
    "Reasoning": "#F5B84B",
    "Korean MCQA": "#8FD17F",
    "Regression": "#FF6B6B",
}

BG = "#0B1020"
PANEL = "#111A2E"
TEXT = "#EAF0FF"
MUTED = "#9AA8C7"
GRID = "#2A375A"
BASE = "#6F7D9B"
CPT = "#15D1B5"
NEG = "#FF6B6B"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.9)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ordered = sorted(SCORES, key=lambda row: row.relative)
    labels = [row.name for row in ordered]
    base = np.array([row.base for row in ordered])
    cpt = np.array([row.cpt for row in ordered])
    rel = np.array([row.relative for row in ordered])
    colors = [GROUP_COLORS[row.group] for row in ordered]
    y = np.arange(len(ordered))

    fig = plt.figure(figsize=(18, 11), dpi=180)
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(
        ncols=2,
        nrows=1,
        width_ratios=[1.35, 1.0],
        left=0.06,
        right=0.98,
        top=0.76,
        bottom=0.13,
        wspace=0.14,
    )

    ax_scores = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])
    _style_axis(ax_scores)
    _style_axis(ax_delta)

    bar_h = 0.36
    ax_scores.barh(y - bar_h / 2, base, height=bar_h, color=BASE, alpha=0.85, label="Base")
    ax_scores.barh(y + bar_h / 2, cpt, height=bar_h, color=CPT, alpha=0.95, label="KO-CPT")
    ax_scores.set_yticks(y)
    ax_scores.set_yticklabels(labels, color=TEXT, fontsize=9)
    ax_scores.set_xlim(0, 0.84)
    ax_scores.set_xlabel("Score, higher is better", color=MUTED, fontsize=10)
    ax_scores.set_title("Base vs KO-CPT scores", loc="left", color=TEXT, fontsize=16, weight="bold", pad=12)
    ax_scores.legend(
        loc="lower right",
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        framealpha=0.95,
    )

    for idx, row in enumerate(ordered):
        ax_scores.text(
            max(row.base, row.cpt) + 0.011,
            idx,
            f"{row.cpt:.3f}",
            color=TEXT,
            fontsize=8,
            va="center",
        )

    delta_colors = [GROUP_COLORS[row.group] if row.delta >= 0 else NEG for row in ordered]
    ax_delta.barh(y, rel, color=delta_colors, alpha=0.95)
    ax_delta.axvline(0, color=TEXT, linewidth=1.2, alpha=0.8)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels([""] * len(y))
    ax_delta.set_xlim(-55, 125)
    ax_delta.set_xlabel("Relative change vs Base (%)", color=MUTED, fontsize=10)
    ax_delta.set_title("CPT delta", loc="left", color=TEXT, fontsize=16, weight="bold", pad=12)

    for idx, row in enumerate(ordered):
        value = row.relative
        x = value + 2.0 if value >= 0 else value - 2.0
        ha = "left" if value >= 0 else "right"
        ax_delta.text(
            x,
            idx,
            f"{value:+.1f}%",
            color=TEXT,
            fontsize=8,
            va="center",
            ha=ha,
        )

    fig.text(
        0.06,
        0.94,
        "LFM2.5-8B-A1B-KO-CPT-FULL",
        color=TEXT,
        fontsize=28,
        weight="bold",
    )
    fig.text(
        0.06,
        0.895,
        "Korean continued pretraining: broad gains in instruction, reasoning, and Korean knowledge; targeted MCQA repair remains open.",
        color=MUTED,
        fontsize=13,
    )

    summary = [
        ("+20.8%", "BoolQ"),
        ("+17.7%", "GSM8K"),
        ("+114.3%", "G-MMLU KO business ethics"),
        ("-35.5%", "MMLU-ProX Lite KO"),
    ]
    x0 = 0.06
    for i, (value, label) in enumerate(summary):
        fig.text(
            x0 + i * 0.225,
            0.845,
            value,
            color=CPT if value.startswith("+") else NEG,
            fontsize=19,
            weight="bold",
        )
        fig.text(x0 + i * 0.225, 0.822, label, color=MUTED, fontsize=9)

    legend_items = [
        Patch(facecolor=color, edgecolor="none", label=group)
        for group, color in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.51, 0.045),
        frameon=False,
        labelcolor=MUTED,
        fontsize=9,
    )

    fig.text(
        0.06,
        0.025,
        "Source: vLLM/lm-eval base-vs-CPT runs in LFM25-KO-CPT. Base is LiquidAI/LFM2.5-8B-A1B.",
        color="#7180A4",
        fontsize=8,
    )

    fig.savefig(OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    print(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
