import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt

PANEL_SLIM_PATH = ROOT / "data" / "processed" / "backtest_ready_panel_slim.csv"
PANEL_FULL_PATH = ROOT / "data" / "processed" / "backtest_ready_panel.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_LABEL_COUNTS = OUT_TABLE_DIR / "oracle_monthly_label_counts.csv"
OUT_RETURNS_CHECK = OUT_TABLE_DIR / "oracle_monthly_returns_check.csv"
OUT_SUMMARY = OUT_TABLE_DIR / "oracle_diagnostic_verification_summary.csv"
OUT_LABEL_DIST_FIG = OUT_FIG_DIR / "oracle_monthly_label_distribution.png"
OUT_RET_HIST_FIG = OUT_FIG_DIR / "oracle_monthly_returns_histogram.png"

THRESHOLD = 0.02


def require_columns(df, required, source_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def pick_input_panel():
    if PANEL_SLIM_PATH.exists():
        return PANEL_SLIM_PATH
    if PANEL_FULL_PATH.exists():
        return PANEL_FULL_PATH
    raise FileNotFoundError("Neither backtest_ready_panel_slim.csv nor backtest_ready_panel.csv exists.")


def reconstruct_label(future_ret):
    return np.where(future_ret > THRESHOLD, "increase", np.where(future_ret < -THRESHOLD, "reduce", "hold"))


def compute_monthly_label_counts(df):
    counts = (
        df.groupby(["date", "label_reconstructed"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    for col in ["increase", "hold", "reduce"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(
        columns={
            "increase": "n_increase",
            "hold": "n_hold",
            "reduce": "n_reduce",
        }
    )
    counts["total_reits"] = counts["n_increase"] + counts["n_hold"] + counts["n_reduce"]
    counts["pct_increase"] = np.where(counts["total_reits"] > 0, counts["n_increase"] / counts["total_reits"], np.nan)
    counts["pct_hold"] = np.where(counts["total_reits"] > 0, counts["n_hold"] / counts["total_reits"], np.nan)
    counts["pct_reduce"] = np.where(counts["total_reits"] > 0, counts["n_reduce"] / counts["total_reits"], np.nan)
    counts = counts[
        ["date", "n_increase", "n_hold", "n_reduce", "total_reits", "pct_increase", "pct_hold", "pct_reduce"]
    ].sort_values("date")
    return counts


def compute_oracle_increase_returns(df):
    rows = []
    for dt, g in df.groupby("date", sort=True):
        selected = g[g["label_reconstructed"] == "increase"]
        n_holdings = int(len(selected))

        if n_holdings == 0:
            monthly_return = 0.0
        else:
            monthly_return = float(selected["future_ret_1m"].mean())

        rows.append({"date": dt, "n_holdings": n_holdings, "monthly_return": monthly_return})

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["nav"] = (1.0 + out["monthly_return"]).cumprod()
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0
    out["is_negative_month"] = out["monthly_return"] < 0
    out["is_zero_month"] = np.isclose(out["monthly_return"], 0.0)
    return out


def build_summary(returns_df):
    n_months = int(len(returns_df))
    min_ret = float(returns_df["monthly_return"].min()) if n_months else np.nan
    mean_ret = float(returns_df["monthly_return"].mean()) if n_months else np.nan
    max_ret = float(returns_df["monthly_return"].max()) if n_months else np.nan
    neg_months = int((returns_df["monthly_return"] < 0).sum()) if n_months else 0
    max_dd = float(returns_df["drawdown"].min()) if n_months else np.nan
    avg_holdings = float(returns_df["n_holdings"].mean()) if n_months else np.nan
    zero_months = int(np.isclose(returns_df["monthly_return"], 0.0).sum()) if n_months else 0

    explanation = (
        "Under the ±2% rule, Oracle Increase-Only holds only REITs with future_ret_1m > +2%. "
        "When holdings exist, the equal-weight realized return is mechanically > +2%, so those months cannot be negative. "
        "Any drawdown can only come from cash months (0% return) or implementation frictions not in this gross check."
    )

    return pd.DataFrame(
        [
            {
                "label_threshold": THRESHOLD,
                "min_monthly_return": min_ret,
                "mean_monthly_return": mean_ret,
                "max_monthly_return": max_ret,
                "n_negative_months": neg_months,
                "n_zero_months": zero_months,
                "max_drawdown": max_dd,
                "avg_n_holdings": avg_holdings,
                "n_months": n_months,
                "explanation": explanation,
            }
        ]
    )


def plot_label_distribution(label_counts):
    plt.figure(figsize=(12, 7))
    plt.plot(label_counts["date"], label_counts["pct_increase"], label="pct_increase", linewidth=1.8)
    plt.plot(label_counts["date"], label_counts["pct_hold"], label="pct_hold", linewidth=1.8)
    plt.plot(label_counts["date"], label_counts["pct_reduce"], label="pct_reduce", linewidth=1.8)
    plt.title("Monthly Label Distribution (Reconstructed, ±2% Threshold)")
    plt.xlabel("Formation Date")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_LABEL_DIST_FIG, dpi=180)
    plt.close()


def plot_returns_histogram(returns_df):
    plt.figure(figsize=(10, 6))
    plt.hist(returns_df["monthly_return"], bins=30, alpha=0.8, edgecolor="black")
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="0%")
    plt.title("Oracle Increase-Only Monthly Returns Histogram (Gross)")
    plt.xlabel("Monthly Return")
    plt.ylabel("Count")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_RET_HIST_FIG, dpi=180)
    plt.close()


def main():
    panel_path = pick_input_panel()
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    require_columns(panel, ["ticker", "date", "future_ret_1m"], panel_path.name)

    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel["future_ret_1m"] = pd.to_numeric(panel["future_ret_1m"], errors="coerce")
    panel = panel.dropna(subset=["ticker", "date", "future_ret_1m"]).sort_values(["date", "ticker"]).copy()
    panel["label_reconstructed"] = reconstruct_label(panel["future_ret_1m"])

    monthly_label_counts = compute_monthly_label_counts(panel)
    oracle_returns = compute_oracle_increase_returns(panel)
    summary = build_summary(oracle_returns)

    monthly_label_counts.to_csv(OUT_LABEL_COUNTS, index=False)
    oracle_returns.to_csv(OUT_RETURNS_CHECK, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    plot_label_distribution(monthly_label_counts)
    plot_returns_histogram(oracle_returns)

    print(f"Input panel: {panel_path}")
    print("Saved:", OUT_LABEL_COUNTS)
    print("Saved:", OUT_RETURNS_CHECK)
    print("Saved:", OUT_SUMMARY)
    print("Saved:", OUT_LABEL_DIST_FIG)
    print("Saved:", OUT_RET_HIST_FIG)
    print("\nOracle verification summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
