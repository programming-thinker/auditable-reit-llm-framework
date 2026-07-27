import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt

PANEL_PATH = ROOT / "data" / "processed" / "backtest_ready_panel_slim.csv"
PRED_PATH = ROOT / "outputs" / "tables" / "quant_only_test_predictions.csv"
SELECTED_PERF_PATH = ROOT / "outputs" / "tables" / "quant_only_selected_strategy_test_performance_gross_net.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_MONTHLY = OUT_TABLE_DIR / "quant_only_selected_strategy_sector_composition_test.csv"
OUT_AVG = OUT_TABLE_DIR / "quant_only_selected_strategy_average_sector_weights_test.csv"
OUT_FIG = OUT_FIG_DIR / "quant_only_selected_strategy_average_sector_weights_test.png"


def require_columns(df, required, source_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def top_by_score(group, n):
    return (
        group.sort_values(["score", "ticker"], ascending=[False, True])
        .head(min(n, len(group)))["ticker"]
        .tolist()
    )


def top_pct(group, pct, min_holdings=1):
    n = int(len(group) * pct)
    if len(group) * pct > n:
        n += 1
    n = max(min_holdings, n)
    return top_by_score(group, min(n, len(group)))


def selected_tickers(group, strategy):
    if strategy == "Quant Hard Increase-Only":
        return group.loc[group["pred_label"] == "increase", "ticker"].tolist()
    if strategy == "Quant Hard Reduce-Avoidance":
        return group.loc[group["pred_label"] != "reduce", "ticker"].tolist()
    if strategy == "Quant Score Positive":
        return group.loc[group["score"] > 0, "ticker"].tolist()
    if strategy in {"Quant Score Top-10", "Quant Score Top-10 Diversified"}:
        return top_by_score(group, 10)
    if strategy == "Quant Score Top-5":
        return top_by_score(group, 5)
    if strategy == "Quant Score Top-30pct":
        return top_pct(group, 0.30, min_holdings=1)
    if strategy == "Quant Score Top-30pct Diversified":
        return top_pct(group, 0.30, min_holdings=10)
    raise ValueError(f"Unsupported selected strategy: {strategy}")


def main():
    pred = pd.read_csv(PRED_PATH, parse_dates=["date"])
    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    perf = pd.read_csv(SELECTED_PERF_PATH)

    require_columns(pred, ["date", "ticker", "pred_label", "score"], PRED_PATH.name)
    require_columns(panel, ["date", "ticker", "sector"], PANEL_PATH.name)
    require_columns(perf, ["strategy", "strategy_group"], SELECTED_PERF_PATH.name)

    selected_rows = perf[perf["strategy_group"] == "quant_only_test"]
    if selected_rows.empty:
        raise ValueError("No selected quant_only_test strategy found.")
    strategy = selected_rows.iloc[0]["strategy"]

    pred["ticker"] = pred["ticker"].astype(str).str.upper()
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    sectors = panel[["date", "ticker", "sector"]].drop_duplicates(subset=["date", "ticker"])
    data = pred.merge(sectors, on=["date", "ticker"], how="left", suffixes=("", "_panel"))
    if "sector" not in data.columns and "sector_panel" in data.columns:
        data["sector"] = data["sector_panel"]
    elif "sector_panel" in data.columns:
        data["sector"] = data["sector"].fillna(data["sector_panel"])
    data["sector"] = data["sector"].fillna("Unknown")

    rows = []
    for dt, group in data.groupby("date", sort=True):
        tickers = selected_tickers(group, strategy)
        if not tickers:
            continue
        selected = group[group["ticker"].isin(tickers)].copy()
        sector_counts = selected.groupby("sector")["ticker"].nunique()
        for sector, count in sector_counts.items():
            rows.append(
                {
                    "strategy": strategy,
                    "date": dt,
                    "sector": sector,
                    "weight": float(count / len(tickers)),
                }
            )

    monthly = pd.DataFrame(rows).sort_values(["strategy", "date", "sector"])
    monthly.to_csv(OUT_MONTHLY, index=False)

    all_dates = pd.Series(sorted(data["date"].dropna().unique()), name="date")
    all_sectors = pd.Series(sorted(data["sector"].dropna().unique()), name="sector")
    full_index = (
        pd.MultiIndex.from_product([[strategy], all_dates, all_sectors], names=["strategy", "date", "sector"])
        .to_frame(index=False)
    )
    monthly_for_average = full_index.merge(monthly, on=["strategy", "date", "sector"], how="left")
    monthly_for_average["weight"] = monthly_for_average["weight"].fillna(0.0)

    avg = (
        monthly_for_average.groupby(["strategy", "sector"], as_index=False)
        .agg(average_weight=("weight", "mean"), max_monthly_weight=("weight", "max"), min_monthly_weight=("weight", "min"))
        .sort_values(["strategy", "average_weight"], ascending=[True, False])
    )
    avg.to_csv(OUT_AVG, index=False)

    plot_data = avg.copy()
    plt.figure(figsize=(11, 7))
    plt.barh(plot_data["sector"], plot_data["average_weight"])
    plt.xlabel("Average portfolio weight")
    plt.ylabel("Sector")
    plt.title(f"Average Sector Weights: {strategy} (Test Period)")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=180)
    plt.close()

    print("Selected strategy:", strategy)
    print("Saved:", OUT_MONTHLY)
    print("Saved:", OUT_AVG)
    print("Saved:", OUT_FIG)
    print(avg.to_string(index=False))


if __name__ == "__main__":
    main()
