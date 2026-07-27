"""Generate publication-quality thesis figures from the real results.

Writes every figure twice to outputs/figures_v2/:
  - .png at 300 dpi (markdown preview)
  - .pdf vector (embedded by the LaTeX build)
No network; pure plotting from existing CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
FR = REPO / "outputs/fundamentals_robustness"
LL = REPO / "outputs/llm_deepseek_test"
OUT = REPO / "outputs/figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- global style
def _serif_family() -> str:
    """Match the thesis body font (Arial, per departmental format); sans fallback."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in names else "sans-serif"


plt.rcParams.update({
    # typography
    "font.family": _serif_family(),
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    # axes / spines / ticks
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    # grid (subtle)
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    # layout / output
    "figure.constrained_layout.use": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colorblind-safe muted palette (Okabe-Ito).
BLUE = "#0072B2"   # dark reference / structured baseline
RED = "#D55E00"    # accent: LLM / key result (vermillion)
GOLD = "#E69F00"   # random-at-budget benchmark
GREY = "#8C8C8C"   # neutral baselines
GREEN = "#009E73"  # spare

FULL_W = 6.3    # full-width figures (inches)
SINGLE_W = 4.5  # single-panel figures (inches)


def save(fig: plt.Figure, stem: str) -> None:
    """Save one figure as both 300-dpi PNG and vector PDF, then close it."""
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Bold (a)/(b) marker in the top-left corner of an axes."""
    ax.text(0.02, 0.97, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold")


# ------------------------------------------------------------------- figures
def fig1_decomposition():
    """Month vs firm explanatory power — the bedrock result."""
    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.4))
    vals = [0.433, 0.007]
    bars = ax.bar(["Which MONTH\n(systematic)", "Firm FEATURES\n(idiosyncratic)"],
                  vals, color=[RED, BLUE], width=0.55)
    ax.set_ylabel(r"$R^2$ explaining which REITs reduce")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 0.5)
    save(fig, "fig1_systematic_vs_idiosyncratic")


def fig2_reduce_timeseries():
    """Monthly reduce-rate over time — clustering / co-movement."""
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["label"])
    rr = df.assign(r=(df["label"] == "reduce")).groupby("date")["r"].mean()
    base = (df["label"] == "reduce").mean()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
    ax.fill_between(rr.index, rr.values, color=BLUE, alpha=0.30, lw=0)
    ax.plot(rr.index, rr.values, color=BLUE, lw=1.0)
    ax.axhline(2 * base, color=RED, ls="--", lw=1,
               label=f"'bad month' threshold (2× base = {2*base:.2f})")
    ax.set_ylabel("Fraction of REITs reducing")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig2_reduce_rate_timeseries")


def fig3_headline():
    """Reduce recall across methods with bootstrap CI where available."""
    h = pd.read_csv(LL / "headline_comparison.csv").set_index("method")
    boot = pd.read_csv(LL / "significance_bootstrap.csv").set_index("quantity")
    order = ["Logistic-argmax", "Threshold-logistic", "Random-at-budget", "LLM-DeepSeek"]
    vals = [h.loc[m, "reduce_recall"] for m in order]
    fig, ax = plt.subplots(figsize=(FULL_W, 3.6))
    colors = [GREY, GREY, GOLD, RED]
    bars = ax.bar(range(len(order)), vals, color=colors, width=0.6)
    # CI for LLM and random
    for i, m, q in [(3, "LLM-DeepSeek", "LLM"), (2, "Random-at-budget", "Random-at-budget")]:
        lo, hi = boot.loc[q, "ci_lo"], boot.loc[q, "ci_hi"]
        ax.errorbar(i, vals[i], yerr=[[vals[i] - lo], [hi - vals[i]]],
                    color="black", capsize=4, lw=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Logistic\n(argmax)", "Threshold\nlogistic",
                        "Random\nat-budget", "LLM\n5-agent"])
    ax.set_ylabel("Reduce recall (test, 575)")
    for i, (b, v) in enumerate(zip(bars, vals)):
        if i in (2, 3):  # bars carry CI whiskers at the bar centre: offset the label
            ax.text(b.get_x() + b.get_width() / 2 + 0.08, v + 0.008, f"{v:.3f}",
                    ha="left", fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_ylim(0, 0.37)
    save(fig, "fig3_headline_reduce_recall")


def fig4_grounding():
    """Rationale grounding v1 (XBRL) vs v2 (fixed) — the auditability contribution."""
    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.4))
    vals = [0.57, 0.88]
    bars = ax.bar(["v1\n(XBRL input)", "v2\n(fixed Item 1A\n+ Fundamentals)"],
                  vals, color=[GREY, BLUE], width=0.55)
    ax.set_ylabel("% of reduce decisions\nciting specific filing facts")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 1.0)
    save(fig, "fig4_grounding")


def fig5_ablation_pr():
    """Channel precision-recall profiles — distinct, non-redundant diagnoses."""
    a = pd.read_csv(LL / "ablation_channels.csv")
    fig, ax = plt.subplots(figsize=(FULL_W, 4.2))
    special = {"Fundamentals-only": (0, 9, "center"),
               "Full 5-agent (LLM-agg)": (6, 7, "left")}
    for _, r in a.iterrows():
        ax.scatter(r["reduce_recall"], r["reduce_precision"], s=55, color=BLUE, zorder=3)
        dx, dy, ha = special.get(r["config"], (4, 4, "left"))
        ax.annotate(r["config"].replace(" (", "\n("),
                    (r["reduce_recall"], r["reduce_precision"]),
                    fontsize=7.5, xytext=(dx, dy), textcoords="offset points", ha=ha)
    base = 0.287
    ax.axhline(base, color=RED, ls="--", lw=1, label=f"reduce base rate ({base:.2f})")
    ax.margins(x=0.22, y=0.15)  # headroom so point annotations stay inside the axes
    ax.set_xlabel("Reduce recall")
    ax.set_ylabel("Reduce precision")
    ax.legend(frameon=False)
    save(fig, "fig5_ablation_pr")


def fig6_market_r2():
    """Per-REIT market-model R² — REITs co-move systematically."""
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["future_ret_1m"])
    mkt = df.groupby("date")["future_ret_1m"].mean().rename("mkt")
    d = df.merge(mkt, on="date")
    r2s = []
    for tk, g in d.groupby("ticker"):
        if len(g) < 24:
            continue
        m = LinearRegression().fit(g[["mkt"]].values, g["future_ret_1m"].values)
        r2s.append(m.score(g[["mkt"]].values, g["future_ret_1m"].values))
    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.4))
    ax.hist(r2s, bins=12, color=BLUE, alpha=0.85, edgecolor="white", lw=0.6)
    ax.axvline(np.median(r2s), color=RED, ls="--", lw=1.2,
               label=f"median $R^2$ = {np.median(r2s):.2f}")
    ax.set_xlabel(r"Market-model $R^2$ (per REIT)")
    ax.set_ylabel("Number of REITs")
    ax.legend(frameon=False)
    save(fig, "fig6_market_model_r2")


def fig8_budget_sweep():
    """Recall vs screening budget: LLM / threshold-logistic / random, plus the
    LLM-minus-logistic difference with month-block bootstrap band."""
    d = pd.read_csv(FR / "budget_sweep.csv").sort_values("budget")
    x = d["budget"] * 100  # budgets shown in %
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_W, 5.2), sharex=True)

    # (a) recall vs budget
    ax1.plot(x, d["llm_recall"], color=RED, lw=1.6, marker="o", ms=3.5,
             label="LLM 5-agent")
    ax1.plot(x, d["logit_recall"], color=BLUE, lw=1.4, ls="--", marker="s", ms=3,
             label="Threshold logistic")
    ax1.plot(x, d["random_expected_recall"], color=GREY, lw=1.2, ls=":",
             label="Random at budget")
    ax1.set_ylabel("Reduce recall")
    ax1.legend(frameon=False, loc="lower right", handlelength=2.4)
    panel_label(ax1, "(a)")

    # (b) LLM − logistic difference with bootstrap band
    ax2.axhline(0, color="black", lw=0.8)
    ax2.fill_between(x, d["boot_ci_lo"], d["boot_ci_hi"], color=RED, alpha=0.15,
                     lw=0, label="95% month-block bootstrap CI")
    ax2.plot(x, d["recall_diff_llm_minus_logit"], color=RED, lw=1.6,
             marker="o", ms=3.5, label="LLM − logistic")
    ax2.set_ylabel("Recall difference\n(LLM − logistic)")
    ax2.set_xlabel("Screening budget (% of firm-months flagged)")
    leg2 = ax2.legend(frameon=True, loc="lower right", framealpha=1.0,
                      facecolor="white", edgecolor="none")
    leg2.set_zorder(5)
    panel_label(ax2, "(b)")

    save(fig, "fig8_budget_sweep")


def fig9_sector():
    """Sector dumbbell: where the LLM concentrates reduce flags vs where the
    true reduces actually are."""
    d = pd.read_csv(FR / "sector_decomposition.csv")
    d = d[d["row_type"] == "sector"].copy()
    d = d.sort_values("llm_reduce_fire_rate")  # ascending -> highest at top
    fire = d["llm_reduce_fire_rate"].to_numpy() * 100
    true = d["true_reduce_rate"].to_numpy() * 100
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(FULL_W, 4.4))
    ax.hlines(y, true, fire, color=GREY, lw=1.4, alpha=0.7, zorder=2)
    ax.scatter(true, y, s=58, facecolor="white", edgecolor=GREY, lw=1.2,
               zorder=3, label="True reduce rate")
    ax.scatter(fire, y, s=38, color=RED, zorder=4, label="LLM reduce-fire rate")
    ax.set_yticks(y)
    ax.set_yticklabels(d["sector"])
    ax.set_xlabel("Share of firm-months (%)")
    ax.set_xlim(0, None)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig9_sector")


def fig10_calibration():
    """Reliability diagram for the reduce probability: LLM vs logistic.
    Complements the Brier/Murphy decomposition; data from econ_value_eval.py."""
    d = pd.read_csv(FR / "calibration_curve.csv")
    ece = {m: d.loc[d["ece"].notna() & (d["model"] == m), "ece"].iloc[0]
           for m in ("llm", "logit")} if "ece" in d.columns else {}
    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.6))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.0, label="Perfect calibration")
    for model, color, marker, label in [
            ("logit", BLUE, "s", "Logistic"), ("llm", RED, "o", "LLM 5-agent")]:
        b = d[(d["model"] == model) & d["mean_forecast"].notna()]
        lab = f"{label} (ECE = {ece[model]:.2f})" if model in ece else label
        ax.plot(b["mean_forecast"], b["observed_freq"], color=color, lw=1.3,
                marker=marker, ms=4, label=lab)
    ax.set_xlabel("Mean forecast probability of reduce (bin)")
    ax.set_ylabel("Observed reduce frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig10_calibration")


def fig11_ceiling():
    """Three-panel systematic-ceiling figure: (a) the universe declines together,
    (b) the common factor explains most variation, (c) month FE >> firm features.
    Merges the former standalone figures 6-8 into one exhibit."""
    fig = plt.figure(figsize=(FULL_W, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    # (a) reduce-rate time series (full width)
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["label"])
    rr = df.assign(r=(df["label"] == "reduce")).groupby("date")["r"].mean()
    base = (df["label"] == "reduce").mean()
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(rr.index, rr.values, color=BLUE, alpha=0.30, lw=0)
    ax.plot(rr.index, rr.values, color=BLUE, lw=1.0)
    ax.axhline(2 * base, color=RED, ls="--", lw=1,
               label=f"'bad month' threshold (2× base = {2*base:.2f})")
    ax.set_ylabel("Fraction reducing")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.055, 1.0), frameon=True,
                    framealpha=1.0, edgecolor="none", facecolor="white")
    leg.set_zorder(5)
    panel_label(ax, "(a)")

    # (b) per-REIT market-model R² histogram
    dfr = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["future_ret_1m"])
    mkt = dfr.groupby("date")["future_ret_1m"].mean().rename("mkt")
    d = dfr.merge(mkt, on="date")
    r2s = []
    for tk, g in d.groupby("ticker"):
        if len(g) < 24:
            continue
        m = LinearRegression().fit(g[["mkt"]].values, g["future_ret_1m"].values)
        r2s.append(m.score(g[["mkt"]].values, g["future_ret_1m"].values))
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(r2s, bins=12, color=BLUE, alpha=0.85, edgecolor="white", lw=0.6)
    ax.axvline(np.median(r2s), color=RED, ls="--", lw=1.2,
               label=f"median $R^2$ = {np.median(r2s):.2f}")
    ax.set_xlabel(r"Market-model $R^2$ (per REIT)")
    ax.set_ylabel("Number of REITs")
    ax.legend(loc="upper left", bbox_to_anchor=(0.14, 1.0), frameon=False,
              fontsize=8)
    panel_label(ax, "(b)")

    # (c) month FE vs firm features
    ax = fig.add_subplot(gs[1, 1])
    vals = [0.433, 0.007]
    bars = ax.bar(["Month fixed\neffects", "Firm's own\nfeatures"], vals,
                  color=[RED, BLUE], width=0.55)
    ax.set_ylabel(r"$R^2$, reduce indicator")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 0.5)
    panel_label(ax, "(c)")

    save(fig, "fig11_systematic_ceiling")


def main():
    fig1_decomposition(); fig2_reduce_timeseries(); fig3_headline()
    fig4_grounding(); fig5_ablation_pr(); fig6_market_r2()
    fig8_budget_sweep(); fig9_sector(); fig10_calibration(); fig11_ceiling()
    figs = sorted(p.name for p in OUT.iterdir()
                  if p.suffix in (".png", ".pdf") and p.stem != "fig_architecture")
    print(f"Generated {len(figs)} files -> {OUT}")
    for f in figs:
        print("  ", f)


if __name__ == "__main__":
    main()
