"""Publication-quality figure of the 5-agent auditable framework (Figure 5.1).
matplotlib only; writes outputs/figures_v2/fig_architecture.png (300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "outputs/figures_v2/fig_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

BLUE, GREY, RED, GREEN, DARK = "#2c5f8a", "#e8eef4", "#b5341d", "#2e6b4f", "#1b2a3a"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(11, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, body, fc=GREY, ec=BLUE, tc=DARK, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top", fontweight="bold",
            fontsize=fs + 1, color=tc, zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=fs - 0.5,
                color="#33424f", zorder=3)


def arrow(x1, y1, x2, y2, color=BLUE, lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, zorder=1,
                                 connectionstyle="arc3,rad=0.0"))


# ── Inputs column ───────────────────────────────────────────────────────────
ax.text(13, 99.5, "Point-in-time inputs\n(filing-date enforced)", ha="center", va="top",
        fontsize=9, fontweight="bold", color=DARK)
inputs = [
    (78, "SEC 10-K Item 1A /\n10-Q / 8-K  (text)"),
    (61, "Macro block\n(6 lagged signals)"),
    (44, "Price / risk\n(6 momentum & vol)"),
    (27, "Fundamentals\n(9: FFO, leverage, NAV…)"),
]
for y, lab in inputs:
    ax.add_patch(FancyBboxPatch((1, y), 24, 11, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#fbf7ee", ec=GREEN, lw=1.4, zorder=2))
    ax.text(1 + 12, y + 5.5, lab, ha="center", va="center", fontsize=7.5,
            color="#33424f", zorder=3)

# ── Specialist agents ───────────────────────────────────────────────────────
ax.text(46, 99.5, "Specialist LLM agents\n(temperature 0, strict JSON)", ha="center",
        va="top", fontsize=9, fontweight="bold", color=DARK)
agents = [
    (78, "Disclosure Agent", "risk events, sentiment"),
    (61, "Macro Agent", "rate regime"),
    (44, "Price Agent", "momentum state"),
    (27, "Fundamentals Agent", "financial health"),
]
for (y, t, b), (iy, _) in zip(agents, inputs):
    box(33, y, 26, 11, t, b, fc="#eaf1f8", ec=BLUE)
    arrow(25.5, iy + 5.5, 33, y + 5.5)
    # each emits structured output; white bbox keeps it legible above the arrows
    ax.text(60.5, y + 5.5, "P(↑/=/↓)\n+ rationale\n+ facts_cited", ha="left", va="center",
            fontsize=6.6, color=RED, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))

# ── Aggregator ──────────────────────────────────────────────────────────────
box(70, 48, 24, 20, "Aggregator Agent",
    "consensus distribution\n+ agreement score\n\n(≈ deterministic mean,\nverified 77% agree)",
    fc="#f3e9e7", ec=RED)
for y, *_ in agents:
    arrow(59, y + 5.5, 70, 58, color="#9c8")

# ── Decision + audit ────────────────────────────────────────────────────────
box(70, 27, 24, 14, "Decision", "final_probabilities\n→ argmax  (or threshold)",
    fc="#eaf1f8", ec=BLUE)
arrow(82, 48, 82, 41, color=RED)

box(70, 7, 24, 14, "Audit log (append-only)",
    "prompt SHA · inputs hash · outputs\ntokens · latency · determin. cache",
    fc="#eef3ef", ec=GREEN)
arrow(82, 27, 82, 21, color=GREEN)

# ── Structured baseline (comparison) ────────────────────────────────────────
box(1, 7, 60, 13, "Tuned structured-ML baseline  (same structured inputs)",
    "multinomial / ordinal logistic · OLS · Fama–MacBeth   |   elastic-net, time-series CV",
    fc="#f4f4f4", ec=GREY, tc="#444", fs=8)
arrow(11, 27, 11, 20, color=GREY, style="-|>")
ax.text(31, 2.5, "Comparison: does the framework go beyond the baseline — in prediction, or in auditable diagnosis?",
        ha="center", fontsize=7.6, style="italic", color="#555")

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("written ->", OUT)
