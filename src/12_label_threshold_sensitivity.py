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

PANEL_PATH = ROOT / "data" / "processed" / "reit_monthly_panel.csv"
MACRO_PATH = ROOT / "data" / "processed" / "monthly_macro_signals.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_LABEL_DIST = OUT_TABLE_DIR / "label_threshold_sensitivity_label_distribution.csv"
OUT_ORACLE_PERF = OUT_TABLE_DIR / "label_threshold_sensitivity_oracle_performance.csv"
OUT_DIST_FIG = OUT_FIG_DIR / "label_threshold_sensitivity_distribution.png"
OUT_NAV_FIG = OUT_FIG_DIR / "label_threshold_sensitivity_oracle_nav.png"

THRESHOLDS = [0.01, 0.02, 0.03, 0.05]
TC_RATE = 0.001
STRATEGY_INC = "Oracle Increase-Only"
STRATEGY_NOT_REDUCE = "Oracle Reduce-Avoidance"


def require_columns(df, required, source_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def load_monthly_rf(panel):
    if "monthly_rf" in panel.columns:
        rf = panel[["date", "monthly_rf"]].drop_duplicates(subset=["date"]).copy()
        rf["monthly_rf"] = pd.to_numeric(rf["monthly_rf"], errors="coerce")
        rf = rf.sort_values("date")
        rf["monthly_rf"] = rf["monthly_rf"].ffill().bfill().fillna(0.0)
        return rf

    if MACRO_PATH.exists():
        macro = pd.read_csv(MACRO_PATH, parse_dates=["date"])
        require_columns(macro, ["date"], MACRO_PATH.name)
        if "monthly_rf" not in macro.columns:
            if "FEDFUNDS" in macro.columns:
                macro["monthly_rf"] = pd.to_numeric(macro["FEDFUNDS"], errors="coerce") / 100.0 / 12.0
            else:
                macro["monthly_rf"] = 0.0
        rf = macro[["date", "monthly_rf"]].drop_duplicates(subset=["date"]).sort_values("date").copy()
        rf["monthly_rf"] = pd.to_numeric(rf["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
        return rf

    if "FEDFUNDS" in panel.columns:
        rf = panel[["date", "FEDFUNDS"]].drop_duplicates(subset=["date"]).copy()
        rf["monthly_rf"] = pd.to_numeric(rf["FEDFUNDS"], errors="coerce") / 100.0 / 12.0
        rf = rf[["date", "monthly_rf"]].sort_values("date")
        rf["monthly_rf"] = rf["monthly_rf"].ffill().bfill().fillna(0.0)
        return rf

    rf = panel[["date"]].drop_duplicates(subset=["date"]).copy()
    rf["monthly_rf"] = 0.0
    return rf.sort_values("date")


def relabel_future_ret(future_ret, threshold):
    return np.where(future_ret > threshold, "increase", np.where(future_ret < -threshold, "reduce", "hold"))


def label_distribution(label_series, threshold):
    counts = label_series.value_counts()
    n_total = int(len(label_series))
    count_increase = int(counts.get("increase", 0))
    count_hold = int(counts.get("hold", 0))
    count_reduce = int(counts.get("reduce", 0))
    return {
        "threshold": threshold,
        "n_total": n_total,
        "count_increase": count_increase,
        "count_hold": count_hold,
        "count_reduce": count_reduce,
        "proportion_increase": count_increase / n_total if n_total else np.nan,
        "proportion_hold": count_hold / n_total if n_total else np.nan,
        "proportion_reduce": count_reduce / n_total if n_total else np.nan,
    }


def build_strategy_monthly(df, strategy_name, selector):
    all_tickers = sorted(df["ticker"].unique().tolist())
    prev_w = pd.Series(0.0, index=all_tickers, dtype=float)
    rows = []

    for dt, g in df.groupby("date", sort=True):
        available = sorted(g["ticker"].unique().tolist())
        selected = sorted(set(selector(g)).intersection(available))

        w = pd.Series(0.0, index=all_tickers, dtype=float)
        if selected:
            w.loc[selected] = 1.0 / len(selected)

        ret_map = g.set_index("ticker")["future_ret_1m"]
        gross_return = float((w.loc[ret_map.index] * ret_map).sum())
        turnover = float(0.5 * np.abs(w - prev_w).sum())
        n_holdings = int((w > 0).sum())

        rows.append(
            {
                "date": dt,
                "strategy": strategy_name,
                "gross_return": gross_return,
                "turnover": turnover,
                "n_holdings": n_holdings,
            }
        )
        prev_w = w

    return pd.DataFrame(rows)


def finalize_monthly(monthly_df, rf_df):
    out = monthly_df.merge(rf_df, on="date", how="left").sort_values("date").reset_index(drop=True)
    out["monthly_rf"] = pd.to_numeric(out["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
    out["net_return"] = out["gross_return"] - TC_RATE * out["turnover"]
    out["excess_return_gross"] = out["gross_return"] - out["monthly_rf"]
    out["excess_return_net"] = out["net_return"] - out["monthly_rf"]
    out["nav_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["nav_net"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown_gross"] = out["nav_gross"] / out["nav_gross"].cummax() - 1.0
    out["drawdown_net"] = out["nav_net"] / out["nav_net"].cummax() - 1.0
    return out


def compute_metrics(df):
    n_months = int(len(df))
    if n_months == 0:
        return {
            "final_nav_gross": np.nan,
            "final_nav_net": np.nan,
            "annualized_return_gross": np.nan,
            "annualized_return_net": np.nan,
            "annualized_volatility_gross": np.nan,
            "annualized_volatility_net": np.nan,
            "annualized_excess_return_gross": np.nan,
            "annualized_excess_return_net": np.nan,
            "excess_sharpe_gross": np.nan,
            "excess_sharpe_net": np.nan,
            "max_drawdown_gross": np.nan,
            "max_drawdown_net": np.nan,
            "n_months": 0,
            "avg_n_holdings": np.nan,
            "avg_monthly_turnover": np.nan,
        }

    final_nav_gross = float(df["nav_gross"].iloc[-1])
    final_nav_net = float(df["nav_net"].iloc[-1])
    annualized_return_gross = final_nav_gross ** (12.0 / n_months) - 1.0
    annualized_return_net = final_nav_net ** (12.0 / n_months) - 1.0

    vol_gross = df["gross_return"].std(ddof=1)
    vol_net = df["net_return"].std(ddof=1)
    annualized_volatility_gross = float(vol_gross * np.sqrt(12.0)) if pd.notna(vol_gross) else np.nan
    annualized_volatility_net = float(vol_net * np.sqrt(12.0)) if pd.notna(vol_net) else np.nan

    annualized_excess_return_gross = float(df["excess_return_gross"].mean() * 12.0)
    annualized_excess_return_net = float(df["excess_return_net"].mean() * 12.0)

    excess_sharpe_gross = (
        annualized_excess_return_gross / annualized_volatility_gross
        if pd.notna(annualized_volatility_gross) and annualized_volatility_gross > 0
        else np.nan
    )
    excess_sharpe_net = (
        annualized_excess_return_net / annualized_volatility_net
        if pd.notna(annualized_volatility_net) and annualized_volatility_net > 0
        else np.nan
    )

    return {
        "final_nav_gross": final_nav_gross,
        "final_nav_net": final_nav_net,
        "annualized_return_gross": annualized_return_gross,
        "annualized_return_net": annualized_return_net,
        "annualized_volatility_gross": annualized_volatility_gross,
        "annualized_volatility_net": annualized_volatility_net,
        "annualized_excess_return_gross": annualized_excess_return_gross,
        "annualized_excess_return_net": annualized_excess_return_net,
        "excess_sharpe_gross": excess_sharpe_gross,
        "excess_sharpe_net": excess_sharpe_net,
        "max_drawdown_gross": float(df["drawdown_gross"].min()),
        "max_drawdown_net": float(df["drawdown_net"].min()),
        "n_months": n_months,
        "avg_n_holdings": float(df["n_holdings"].mean()),
        "avg_monthly_turnover": float(df["turnover"].mean()),
    }


def plot_distribution(dist_df):
    plt.figure(figsize=(10, 6))
    plt.plot(dist_df["threshold"], dist_df["proportion_increase"], marker="o", linewidth=2, label="increase")
    plt.plot(dist_df["threshold"], dist_df["proportion_hold"], marker="o", linewidth=2, label="hold")
    plt.plot(dist_df["threshold"], dist_df["proportion_reduce"], marker="o", linewidth=2, label="reduce")
    plt.xlabel("Label Threshold")
    plt.ylabel("Proportion")
    plt.title("Label Distribution Sensitivity to Threshold")
    plt.xticks(THRESHOLDS, [f"{x:.2f}" for x in THRESHOLDS])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIST_FIG, dpi=180)
    plt.close()


def plot_oracle_nav(nav_df):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for threshold in THRESHOLDS:
        tdf = nav_df[nav_df["threshold"] == threshold]
        for strategy, s in tdf.groupby("strategy", sort=True):
            short_name = "IncOnly" if strategy == STRATEGY_INC else "NotReduce"
            label = f"{short_name} thr={threshold:.2f}"
            axes[0].plot(s["date"], s["nav_gross"], linewidth=1.7, label=label)
            axes[1].plot(s["date"], s["nav_net"], linewidth=1.7, label=label)

    axes[0].set_title("Oracle Diagnostic NAV by Threshold (Gross)")
    axes[1].set_title("Oracle Diagnostic NAV by Threshold (Net, TC-adjusted)")
    axes[0].set_ylabel("NAV")
    axes[1].set_ylabel("NAV")
    axes[1].set_xlabel("Formation Date")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_NAV_FIG, dpi=180)
    plt.close()


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    require_columns(panel, ["ticker", "date", "future_ret_1m"], PANEL_PATH.name)

    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel["future_ret_1m"] = pd.to_numeric(panel["future_ret_1m"], errors="coerce")
    panel = panel.dropna(subset=["ticker", "date", "future_ret_1m"]).sort_values(["date", "ticker"]).copy()

    monthly_rf = load_monthly_rf(panel)

    label_dist_rows = []
    perf_rows = []
    nav_rows = []

    for threshold in THRESHOLDS:
        tdf = panel.copy()
        tdf["label_reconstructed"] = relabel_future_ret(tdf["future_ret_1m"], threshold)

        label_dist_rows.append(label_distribution(tdf["label_reconstructed"], threshold))

        monthly_inc = build_strategy_monthly(
            tdf,
            STRATEGY_INC,
            selector=lambda g: g.loc[g["label_reconstructed"] == "increase", "ticker"].tolist(),
        )
        monthly_not_reduce = build_strategy_monthly(
            tdf,
            STRATEGY_NOT_REDUCE,
            selector=lambda g: g.loc[g["label_reconstructed"] != "reduce", "ticker"].tolist(),
        )

        for monthly in [monthly_inc, monthly_not_reduce]:
            monthly_final = finalize_monthly(monthly, monthly_rf)
            monthly_final["threshold"] = threshold
            nav_rows.append(monthly_final.copy())

            metrics = compute_metrics(monthly_final)
            perf_rows.append(
                {
                    "threshold": threshold,
                    "strategy": monthly_final["strategy"].iloc[0],
                    **metrics,
                }
            )

    dist_df = pd.DataFrame(label_dist_rows).sort_values("threshold").reset_index(drop=True)
    perf_df = pd.DataFrame(perf_rows).sort_values(["threshold", "strategy"]).reset_index(drop=True)
    nav_df = pd.concat(nav_rows, ignore_index=True).sort_values(["threshold", "strategy", "date"]).reset_index(drop=True)

    dist_df.to_csv(OUT_LABEL_DIST, index=False)
    perf_df.to_csv(OUT_ORACLE_PERF, index=False)

    plot_distribution(dist_df)
    plot_oracle_nav(nav_df)

    print("Saved:", OUT_LABEL_DIST)
    print("Saved:", OUT_ORACLE_PERF)
    print("Saved:", OUT_DIST_FIG)
    print("Saved:", OUT_NAV_FIG)
    print("\nLabel distribution summary:")
    print(dist_df.to_string(index=False))
    print("\nOracle performance summary:")
    print(perf_df.to_string(index=False))


if __name__ == "__main__":
    main()
