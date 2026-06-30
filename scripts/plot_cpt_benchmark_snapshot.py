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
SCORES_OUT_PATH = OUT_DIR / "cpt_benchmark_scores.png"
DELTA_OUT_PATH = OUT_DIR / "cpt_benchmark_delta.png"
LINKEDIN_SQUARE_OUT_PATH = OUT_DIR / "cpt_benchmark_linkedin_square.png"
LINKEDIN_PORTRAIT_OUT_PATH = OUT_DIR / "cpt_benchmark_linkedin_portrait.png"
LINKEDIN_WIDE_OUT_PATH = OUT_DIR / "cpt_benchmark_linkedin_wide.png"
LINKEDIN_HIGHLIGHTS_OUT_PATH = OUT_DIR / "cpt_benchmark_linkedin_highlights.png"
LINKEDIN_LIGHT_GRID_OUT_PATH = OUT_DIR / "cpt_benchmark_linkedin_light_grid.png"
LINKEDIN_DARK_GRID_OUT_PATH = OUT_DIR / "cpt_benchmark_linkedin_dark_grid.png"


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


def _ordered_scores() -> list[Score]:
    return sorted(SCORES, key=lambda row: row.relative)


def _add_header(fig: plt.Figure, subtitle: str) -> None:
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
        subtitle,
        color=MUTED,
        fontsize=13,
    )


def _add_summary(fig: plt.Figure) -> None:
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


def _add_group_legend(fig: plt.Figure, y: float = 0.045) -> None:
    legend_items = [
        Patch(facecolor=color, edgecolor="none", label=group)
        for group, color in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.51, y),
        frameon=False,
        labelcolor=MUTED,
        fontsize=9,
    )


def _add_source_note(fig: plt.Figure) -> None:
    fig.text(
        0.06,
        0.025,
        "Source: vLLM/lm-eval base-vs-CPT runs in LFM25-KO-CPT. Base is LiquidAI/LFM2.5-8B-A1B.",
        color="#7180A4",
        fontsize=8,
    )


def _short_name(name: str) -> str:
    return name.replace("\n", " ")


def _panel_name(name: str) -> str:
    return {
        "G-MMLU KO\nbusiness ethics": "Business ethics",
        "G-MMLU KO\nsociology": "Sociology",
        "G-MMLU KO\ncomputer security": "Comp. security",
        "G-MMLU KO\nworld religions": "World religions",
        "G-MMLU KO\nmanagement": "Management",
        "G-MMLU KO\nmedical genetics": "Medical genetics",
        "GSM8K\nflex": "GSM8K",
        "Leaderboard\nIFEval": "IFEval",
        "G-MMLU KO\nstatistics": "Statistics",
        "MMLU-ProX\nLite KO": "MMLU-ProX KO",
        "MMLU-Pro\nlaw": "MMLU-Pro law",
        "KMMLU hard\nHUMSS": "KMMLU HUMSS",
        "IFEval\nprompt": "IFEval",
    }.get(name, name.replace("\n", " "))


def _add_metric_card(
    fig: plt.Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    label: str,
    color: str,
) -> None:
    fig.text(
        x,
        y + h * 0.52,
        value,
        color=color,
        fontsize=24,
        weight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        x,
        y + h * 0.23,
        label,
        color=MUTED,
        fontsize=9,
        ha="left",
        va="bottom",
    )


def _plot_compact_delta(ax: plt.Axes, rows: list[Score], title: str, xlim: tuple[float, float]) -> None:
    y = np.arange(len(rows))
    colors = [GROUP_COLORS[row.group] if row.delta >= 0 else NEG for row in rows]
    ax.barh(y, [row.relative for row in rows], color=colors, alpha=0.96)
    ax.axvline(0, color=TEXT, linewidth=1.1, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(row.name) for row in rows], color=TEXT, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Relative change vs Base (%)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=14, weight="bold", pad=10)
    for idx, row in enumerate(rows):
        value = row.relative
        x = value + 2.0 if value >= 0 else value - 2.0
        ha = "left" if value >= 0 else "right"
        ax.text(x, idx, f"{value:+.1f}%", color=TEXT, fontsize=8, va="center", ha=ha)


def _plot_compact_scores(ax: plt.Axes, rows: list[Score], title: str) -> None:
    y = np.arange(len(rows))
    bar_h = 0.35
    ax.barh(y - bar_h / 2, [row.base for row in rows], height=bar_h, color=BASE, alpha=0.82, label="Base")
    ax.barh(y + bar_h / 2, [row.cpt for row in rows], height=bar_h, color=CPT, alpha=0.96, label="KO-CPT")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(row.name) for row in rows], color=TEXT, fontsize=9)
    ax.set_xlim(0, 0.84)
    ax.set_xlabel("Score", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=14, weight="bold", pad=10)
    ax.legend(
        loc="lower right",
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        framealpha=0.95,
        fontsize=8,
    )


def _plot_linkedin_square() -> None:
    rows = sorted(SCORES, key=lambda row: row.relative)
    focus = rows[:4] + rows[-8:]
    fig = plt.figure(figsize=(10, 10), dpi=180)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.24, 0.14, 0.70, 0.57])
    _style_axis(ax)
    _plot_compact_delta(ax, focus, "Base → KO-CPT: strongest moves", (-55, 125))
    fig.text(0.07, 0.93, "KO-CPT Benchmark Snapshot", color=TEXT, fontsize=25, weight="bold")
    fig.text(
        0.07,
        0.885,
        "Korean CPT lifted knowledge/reasoning, but MCQA repair remains open.",
        color=MUTED,
        fontsize=11,
    )
    _add_metric_card(fig, 0.07, 0.76, 0.20, 0.08, "+20.8%", "BoolQ", CPT)
    _add_metric_card(fig, 0.30, 0.76, 0.20, 0.08, "+17.7%", "GSM8K", CPT)
    _add_metric_card(fig, 0.53, 0.76, 0.20, 0.08, "+114.3%", "Business ethics", CPT)
    _add_metric_card(fig, 0.78, 0.76, 0.18, 0.08, "-35.5%", "MMLU-ProX KO", NEG)
    fig.text(0.07, 0.055, "Model: LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL", color="#7180A4", fontsize=8)
    fig.savefig(LINKEDIN_SQUARE_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)


def _plot_linkedin_portrait() -> None:
    rows = sorted(SCORES, key=lambda row: row.relative)
    focus = rows[:5] + rows[-9:]
    fig = plt.figure(figsize=(10, 12.5), dpi=180)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.25, 0.13, 0.69, 0.58])
    _style_axis(ax)
    _plot_compact_delta(ax, focus, "Relative change vs Base", (-55, 125))
    fig.text(0.07, 0.94, "Korean CPT worked.", color=TEXT, fontsize=27, weight="bold")
    fig.text(0.07, 0.905, "But exact-choice Korean MCQA needs targeted repair.", color=MUTED, fontsize=13)
    fig.text(0.07, 0.84, "6.493B", color=CPT, fontsize=34, weight="bold")
    fig.text(0.07, 0.815, "estimated CPT tokens", color=MUTED, fontsize=10)
    fig.text(0.42, 0.84, "10,196", color=CPT, fontsize=34, weight="bold")
    fig.text(0.42, 0.815, "full-parameter CPT steps", color=MUTED, fontsize=10)
    fig.text(0.07, 0.765, "+20.8% BoolQ  |  +17.7% GSM8K  |  +114.3% G-MMLU KO business ethics", color=TEXT, fontsize=11)
    fig.text(0.07, 0.735, "-35.5% MMLU-ProX Lite KO shows the remaining MCQA weakness", color=NEG, fontsize=11, weight="bold")
    fig.text(0.07, 0.055, "Base: LiquidAI/LFM2.5-8B-A1B   CPT: LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL", color="#7180A4", fontsize=8)
    fig.savefig(LINKEDIN_PORTRAIT_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)


def _plot_linkedin_wide() -> None:
    ordered = _ordered_scores()
    y = np.arange(len(ordered))
    fig = plt.figure(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.18, 0.16, 0.77, 0.60])
    _style_axis(ax)
    _plot_delta_axis(ax, ordered, y, show_labels=True)
    fig.text(0.05, 0.92, "LFM2.5 KO-CPT: clear gains, clear repair target", color=TEXT, fontsize=27, weight="bold")
    fig.text(0.05, 0.87, "A transparent Korean continued-pretraining run: publish the gains and the regressions.", color=MUTED, fontsize=13)
    _add_metric_card(fig, 0.05, 0.77, 0.18, 0.08, "+20.8%", "BoolQ", CPT)
    _add_metric_card(fig, 0.25, 0.77, 0.18, 0.08, "+17.7%", "GSM8K", CPT)
    _add_metric_card(fig, 0.45, 0.77, 0.18, 0.08, "+114.3%", "Business ethics", CPT)
    _add_metric_card(fig, 0.70, 0.77, 0.18, 0.08, "-35.5%", "MMLU-ProX KO", NEG)
    fig.text(0.05, 0.045, "Source: vLLM/lm-eval base-vs-CPT runs. Higher is better.", color="#7180A4", fontsize=8)
    fig.savefig(LINKEDIN_WIDE_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)


def _plot_linkedin_highlights() -> None:
    fig = plt.figure(figsize=(10, 10), dpi=180)
    fig.patch.set_facecolor(BG)
    fig.text(0.08, 0.91, "LFM2.5 KO-CPT", color=TEXT, fontsize=31, weight="bold")
    fig.text(0.08, 0.865, "Korean continued pretraining benchmark card", color=MUTED, fontsize=13)
    cards = [
        ("6.493B", "estimated Korean CPT tokens", CPT),
        ("10,196", "full-parameter CPT steps", CPT),
        ("+20.8%", "BoolQ vs Base", CPT),
        ("+17.7%", "GSM8K vs Base", CPT),
        ("+114.3%", "Global MMLU KO business ethics", CPT),
        ("-35.5%", "MMLU-ProX Lite KO vs Base", NEG),
    ]
    for i, (value, label, color) in enumerate(cards):
        col = i % 2
        row = i // 2
        x = 0.10 + col * 0.43
        y = 0.67 - row * 0.20
        fig.text(x, y, value, color=color, fontsize=33, weight="bold", ha="left")
        fig.text(x, y - 0.045, label, color=MUTED, fontsize=11, ha="left")
    fig.text(
        0.08,
        0.13,
        "Conclusion: CPT injected Korean/domain knowledge, but exact-choice MCQA needs a small targeted repair stage.",
        color=TEXT,
        fontsize=13,
        wrap=True,
    )
    fig.text(0.08, 0.055, "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL", color="#7180A4", fontsize=9)
    fig.savefig(LINKEDIN_HIGHLIGHTS_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)


def _lookup(name: str) -> Score:
    for row in SCORES:
        if row.name == name:
            return row
    raise KeyError(name)


def _light_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="x", color="#E5E7EF", alpha=1.0, linewidth=0.9)
    ax.tick_params(colors="#535B6B", labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#B8BECC")
        ax.spines[side].set_linewidth(1.0)


def _plot_light_delta_panel(ax: plt.Axes, rows: list[Score], title: str) -> None:
    y = np.arange(len(rows))
    colors = ["#12BFA5" if row.delta >= 0 else "#F0646A" for row in rows]
    ax.bar(y, [row.relative for row in rows], color=colors, width=0.62)
    ax.axhline(0, color="#6B7280", linewidth=1.0)
    ax.set_ylim(-55, 125)
    ax.set_xticks(y)
    ax.set_xticklabels([_panel_name(row.name) for row in rows], rotation=0, ha="center", fontsize=7)
    ax.set_title(title, color="#111827", fontsize=12, weight="bold", pad=9)
    for idx, row in enumerate(rows):
        value = row.relative
        if value >= 0:
            y_text = value + 4
            va = "bottom"
            color = "#111827"
        else:
            y_text = value + 5
            va = "bottom"
            color = "#FFFFFF"
        ax.text(idx, y_text, f"{value:+.1f}%", color=color, fontsize=7, weight="bold", ha="center", va=va)


def _plot_dark_delta_panel(ax: plt.Axes, rows: list[Score], title: str) -> None:
    _style_axis(ax)
    y = np.arange(len(rows))
    colors = [GROUP_COLORS[row.group] if row.delta >= 0 else NEG for row in rows]
    ax.bar(y, [row.relative for row in rows], color=colors, width=0.62)
    ax.axhline(0, color=TEXT, linewidth=1.0, alpha=0.8)
    ax.set_ylim(-55, 125)
    ax.set_xticks(y)
    ax.set_xticklabels([_panel_name(row.name) for row in rows], rotation=0, ha="center", color=TEXT, fontsize=7)
    ax.set_title(title, color=TEXT, fontsize=12, weight="bold", pad=9)
    for idx, row in enumerate(rows):
        value = row.relative
        y_text = value + 4 if value >= 0 else value + 5
        va = "bottom"
        ax.text(idx, y_text, f"{value:+.1f}%", color=TEXT, fontsize=7, weight="bold", ha="center", va=va)


def _plot_linkedin_light_grid() -> None:
    knowledge = [
        _lookup("G-MMLU KO\nbusiness ethics"),
        _lookup("G-MMLU KO\nsociology"),
        _lookup("G-MMLU KO\ncomputer security"),
        _lookup("G-MMLU KO\nworld religions"),
    ]
    general = [
        _lookup("BoolQ"),
        _lookup("GSM8K\nflex"),
        _lookup("ARC-Challenge"),
        _lookup("Leaderboard\nIFEval"),
    ]
    repair = [
        _lookup("G-MMLU KO\nstatistics"),
        _lookup("MMLU-ProX\nLite KO"),
        _lookup("MMLU-Pro\nlaw"),
        _lookup("KMMLU hard"),
    ]

    fig = plt.figure(figsize=(12.25, 10), dpi=180)
    fig.patch.set_facecolor("#FBFBFD")
    border = plt.Rectangle(
        (0.02, 0.02),
        0.96,
        0.96,
        transform=fig.transFigure,
        fill=False,
        edgecolor="#D8B4FE",
        linewidth=2.0,
        joinstyle="round",
    )
    fig.add_artist(border)
    fig.text(0.50, 0.92, "LFM2.5-8B-A1B-KO-CPT-FULL", color="#111827", fontsize=26, weight="bold", ha="center")
    fig.text(0.50, 0.875, "Korean Continued Pretraining: gains, tradeoffs, and next repair target", color="#4B5563", fontsize=12, ha="center")

    chips = [
        ("6.493B", "estimated tokens", "#12BFA5"),
        ("10,196", "CPT steps", "#12BFA5"),
        ("+20.8%", "BoolQ", "#12BFA5"),
        ("-35.5%", "MMLU-ProX KO", "#F0646A"),
    ]
    for idx, (value, label, color) in enumerate(chips):
        x = 0.12 + idx * 0.22
        fig.text(x, 0.805, value, color=color, fontsize=20, weight="bold", ha="center")
        fig.text(x, 0.777, label, color="#6B7280", fontsize=8, ha="center")

    panels = [
        ([0.07, 0.49, 0.40, 0.20], knowledge, "Korean knowledge gains"),
        ([0.53, 0.49, 0.40, 0.20], general, "General / instruction gains"),
        ([0.07, 0.20, 0.40, 0.20], repair, "MCQA repair targets"),
    ]
    for rect, rows, title in panels:
        ax = fig.add_axes(rect)
        _light_axis(ax)
        _plot_light_delta_panel(ax, rows, title)

    ax_note = fig.add_axes([0.53, 0.20, 0.40, 0.20])
    ax_note.axis("off")
    ax_note.text(0.0, 0.86, "What this means", color="#111827", fontsize=13, weight="bold")
    ax_note.text(0.0, 0.62, "CPT injected Korean/domain knowledge.", color="#374151", fontsize=10)
    ax_note.text(0.0, 0.44, "It also exposed a clear exact-choice MCQA weakness.", color="#374151", fontsize=10)
    ax_note.text(0.0, 0.26, "Next stage should be small, targeted, and gate-driven.", color="#374151", fontsize=10)
    ax_note.text(0.0, 0.04, "Representative model: KO-CPT", color="#111827", fontsize=10, weight="bold")
    fig.text(0.07, 0.07, "Source: vLLM/lm-eval Base vs KO-CPT runs. Higher is better.", color="#6B7280", fontsize=8)
    fig.savefig(LINKEDIN_LIGHT_GRID_OUT_PATH, facecolor="#FBFBFD", bbox_inches="tight", pad_inches=0.2)


def _plot_linkedin_dark_grid() -> None:
    knowledge = [
        _lookup("G-MMLU KO\nbusiness ethics"),
        _lookup("G-MMLU KO\nsociology"),
        _lookup("G-MMLU KO\ncomputer security"),
        _lookup("G-MMLU KO\nworld religions"),
    ]
    general = [
        _lookup("BoolQ"),
        _lookup("GSM8K\nflex"),
        _lookup("ARC-Challenge"),
        _lookup("Leaderboard\nIFEval"),
    ]
    repair = [
        _lookup("G-MMLU KO\nstatistics"),
        _lookup("MMLU-ProX\nLite KO"),
        _lookup("MMLU-Pro\nlaw"),
        _lookup("KMMLU hard"),
    ]

    fig = plt.figure(figsize=(12.25, 10), dpi=180)
    fig.patch.set_facecolor(BG)
    fig.text(0.06, 0.92, "KO-CPT: Korean knowledge up, MCQA target found", color=TEXT, fontsize=25, weight="bold")
    fig.text(0.06, 0.875, "A transparent before/after card for LFM2.5-8B-A1B Korean continued pretraining.", color=MUTED, fontsize=12)
    chips = [
        ("6.493B", "estimated tokens", CPT),
        ("10,196", "CPT steps", CPT),
        ("+114.3%", "best knowledge gain", CPT),
        ("-35.5%", "MCQA repair signal", NEG),
    ]
    for idx, (value, label, color) in enumerate(chips):
        x = 0.08 + idx * 0.23
        fig.text(x, 0.80, value, color=color, fontsize=22, weight="bold")
        fig.text(x, 0.772, label, color=MUTED, fontsize=8)

    panels = [
        ([0.08, 0.49, 0.38, 0.20], knowledge, "Korean knowledge"),
        ([0.54, 0.49, 0.38, 0.20], general, "General / instruction"),
        ([0.08, 0.20, 0.38, 0.20], repair, "Repair targets"),
    ]
    for rect, rows, title in panels:
        ax = fig.add_axes(rect)
        _plot_dark_delta_panel(ax, rows, title)

    ax_note = fig.add_axes([0.54, 0.20, 0.38, 0.20])
    ax_note.axis("off")
    ax_note.text(0.0, 0.86, "Conclusion", color=TEXT, fontsize=14, weight="bold")
    ax_note.text(0.0, 0.62, "Use KO-CPT as the representative checkpoint.", color=MUTED, fontsize=10)
    ax_note.text(0.0, 0.44, "Do not hide MCQA regressions.", color=MUTED, fontsize=10)
    ax_note.text(0.0, 0.26, "Repair should be small, exact-answer focused, and gated.", color=MUTED, fontsize=10)
    ax_note.text(0.0, 0.04, "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL", color=TEXT, fontsize=9, weight="bold")
    fig.text(0.08, 0.07, "Source: vLLM/lm-eval Base vs KO-CPT runs. Higher is better.", color="#7180A4", fontsize=8)
    fig.savefig(LINKEDIN_DARK_GRID_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)


def _plot_score_axis(ax_scores: plt.Axes, ordered: list[Score], y: np.ndarray) -> None:
    base = np.array([row.base for row in ordered])
    cpt = np.array([row.cpt for row in ordered])
    labels = [row.name for row in ordered]
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


def _plot_delta_axis(ax_delta: plt.Axes, ordered: list[Score], y: np.ndarray, show_labels: bool) -> None:
    rel = np.array([row.relative for row in ordered])
    labels = [row.name for row in ordered]
    delta_colors = [GROUP_COLORS[row.group] if row.delta >= 0 else NEG for row in ordered]
    ax_delta.barh(y, rel, color=delta_colors, alpha=0.95)
    ax_delta.axvline(0, color=TEXT, linewidth=1.2, alpha=0.8)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(labels if show_labels else [""] * len(y), color=TEXT, fontsize=9)
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ordered = _ordered_scores()
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

    _plot_score_axis(ax_scores, ordered, y)
    _plot_delta_axis(ax_delta, ordered, y, show_labels=False)
    _add_header(
        fig,
        "Korean continued pretraining: broad gains in instruction, reasoning, and Korean knowledge; targeted MCQA repair remains open.",
    )
    _add_summary(fig)
    _add_group_legend(fig)
    _add_source_note(fig)

    fig.savefig(OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)

    score_fig = plt.figure(figsize=(13.5, 11), dpi=180)
    score_fig.patch.set_facecolor(BG)
    score_ax = score_fig.add_axes([0.16, 0.13, 0.80, 0.65])
    _style_axis(score_ax)
    _plot_score_axis(score_ax, ordered, y)
    _add_header(score_fig, "Absolute benchmark scores: Base vs Korean CPT.")
    _add_summary(score_fig)
    _add_source_note(score_fig)
    score_fig.savefig(SCORES_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)

    delta_fig = plt.figure(figsize=(13.5, 11), dpi=180)
    delta_fig.patch.set_facecolor(BG)
    delta_ax = delta_fig.add_axes([0.20, 0.13, 0.76, 0.65])
    _style_axis(delta_ax)
    _plot_delta_axis(delta_ax, ordered, y, show_labels=True)
    _add_header(delta_fig, "Relative change vs Base: Korean knowledge gains and MCQA regressions at a glance.")
    _add_summary(delta_fig)
    _add_group_legend(delta_fig)
    _add_source_note(delta_fig)
    delta_fig.savefig(DELTA_OUT_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.2)

    _plot_linkedin_square()
    _plot_linkedin_portrait()
    _plot_linkedin_wide()
    _plot_linkedin_highlights()
    _plot_linkedin_light_grid()
    _plot_linkedin_dark_grid()

    print(OUT_PATH)
    print(SCORES_OUT_PATH)
    print(DELTA_OUT_PATH)
    print(LINKEDIN_SQUARE_OUT_PATH)
    print(LINKEDIN_PORTRAIT_OUT_PATH)
    print(LINKEDIN_WIDE_OUT_PATH)
    print(LINKEDIN_HIGHLIGHTS_OUT_PATH)
    print(LINKEDIN_LIGHT_GRID_OUT_PATH)
    print(LINKEDIN_DARK_GRID_OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
