import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

PANEL_PATH = ROOT / "data" / "processed" / "backtest_ready_panel_slim.csv"
DAILY_PRICES_PATH = ROOT / "data" / "raw" / "prices" / "daily_prices.csv"
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"
MACRO_SIGNALS_PATH = ROOT / "data" / "processed" / "monthly_macro_signals.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMPL_COMMON = OUT_TABLE_DIR / "baseline_performance_table_implementable_common_period_gross_net.csv"
OUT_IMPL_TEST = OUT_TABLE_DIR / "baseline_performance_table_implementable_test_period_gross_net.csv"
OUT_ORACLE = OUT_TABLE_DIR / "baseline_performance_table_oracle_diagnostic_gross_net.csv"
OUT_MONTHLY = OUT_TABLE_DIR / "baseline_monthly_returns_gross_net.csv"

OUT_NAV_IMPL_COMMON = OUT_FIG_DIR / "baseline_cumulative_nav_implementable_common_period_gross_net.png"
OUT_DD_IMPL_COMMON = OUT_FIG_DIR / "baseline_drawdown_implementable_common_period_gross_net.png"
OUT_NAV_ORACLE = OUT_FIG_DIR / "baseline_cumulative_nav_oracle_diagnostic_gross_net.png"
OUT_DD_ORACLE = OUT_FIG_DIR / "baseline_drawdown_oracle_diagnostic_gross_net.png"

TEST_START = pd.Timestamp("2024-01-31")
TEST_END = pd.Timestamp("2025-11-30")
BENCHMARK_TICKERS = ["SPY", "VNQ", "XLRE"]
TC_RATE = 0.001



def get_month_end_freq():
    try:
        to_offset("ME")
        return "ME"
    except ValueError:
        return "M"



def require_columns(df, required, source_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")



def prepare_monthly_rf():
    macro = pd.read_csv(MACRO_SIGNALS_PATH, parse_dates=["date"])
    require_columns(macro, ["date"], MACRO_SIGNALS_PATH.name)

    if "monthly_rf" not in macro.columns:
        if "FEDFUNDS" not in macro.columns:
            raise ValueError("monthly_macro_signals.csv missing both monthly_rf and FEDFUNDS")
        macro["monthly_rf"] = macro["FEDFUNDS"] / 100.0 / 12.0

    rf = macro[["date", "monthly_rf"]].copy()
    rf = rf.drop_duplicates(subset=["date"]).sort_values("date")
    rf["monthly_rf"] = pd.to_numeric(rf["monthly_rf"], errors="coerce")
    rf["monthly_rf"] = rf["monthly_rf"].ffill().bfill()
    return rf



def finalize_strategy_series(df, monthly_rf, strategy_group, diagnostic_note):
    out = df.copy()
    out = out.merge(monthly_rf, on="date", how="left")
    out["monthly_rf"] = out["monthly_rf"].ffill().bfill().fillna(0.0)

    out["net_return"] = out["gross_return"] - TC_RATE * out["turnover"]
    out["excess_return_gross"] = out["gross_return"] - out["monthly_rf"]
    out["excess_return_net"] = out["net_return"] - out["monthly_rf"]

    out = out.sort_values("date").reset_index(drop=True)
    out["nav_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["nav_net"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown_gross"] = out["nav_gross"] / out["nav_gross"].cummax() - 1.0
    out["drawdown_net"] = out["nav_net"] / out["nav_net"].cummax() - 1.0

    out["strategy_group"] = strategy_group
    out["diagnostic_note"] = diagnostic_note
    return out



def rebase_series(df):
    core_cols = [
        "date",
        "strategy",
        "strategy_group",
        "diagnostic_note",
        "gross_return",
        "net_return",
        "monthly_rf",
        "n_holdings",
        "turnover",
    ]
    out = df[core_cols].sort_values("date").reset_index(drop=True).copy()
    out["excess_return_gross"] = out["gross_return"] - out["monthly_rf"]
    out["excess_return_net"] = out["net_return"] - out["monthly_rf"]
    out["nav_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["nav_net"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown_gross"] = out["nav_gross"] / out["nav_gross"].cummax() - 1.0
    out["drawdown_net"] = out["nav_net"] / out["nav_net"].cummax() - 1.0
    return out



def build_reit_gross_series(panel, all_universe_tickers, strategy_name, selector):
    rows = []
    prev_drifted_w = pd.Series(0.0, index=all_universe_tickers, dtype=float)

    for dt, g in panel.groupby("date", sort=True):
        available_tickers = sorted(g["ticker"].unique().tolist())
        selected = sorted(set(selector(g)).intersection(available_tickers))

        w = pd.Series(0.0, index=all_universe_tickers, dtype=float)
        if selected:
            w.loc[selected] = 1.0 / len(selected)

        returns = g.set_index("ticker")["future_ret_1m"].reindex(all_universe_tickers).fillna(0.0)
        gross_return = float((w * returns).sum())

        turnover = float(0.5 * np.abs(w - prev_drifted_w).sum())
        n_holdings = int((w > 0).sum())

        rows.append(
            {
                "date": dt,
                "strategy": strategy_name,
                "gross_return": gross_return,
                "n_holdings": n_holdings,
                "turnover": turnover,
            }
        )

        if n_holdings > 0 and gross_return > -1.0:
            prev_drifted_w = (w * (1.0 + returns)) / (1.0 + gross_return)
        else:
            prev_drifted_w = pd.Series(0.0, index=all_universe_tickers, dtype=float)

    return pd.DataFrame(rows)



def build_benchmark_gross_series(daily_prices, ticker, month_end_freq):
    px = daily_prices[daily_prices["ticker"] == ticker].copy()
    if px.empty:
        raise ValueError(f"No daily price data found for benchmark ticker {ticker}")

    px = px.sort_values("date")
    monthly = (
        px.set_index("date")
        .resample(month_end_freq)["adj_close"]
        .last()
        .dropna()
        .to_frame()
        .reset_index()
    )

    monthly["gross_return"] = monthly["adj_close"].shift(-1) / monthly["adj_close"] - 1.0
    monthly = monthly.dropna(subset=["gross_return"]).copy()

    out = monthly[["date", "gross_return"]].copy()
    out["strategy"] = f"Benchmark {ticker}"
    out["n_holdings"] = 1
    out["turnover"] = 0.0
    return out



def align_common_period(strategy_frames):
    date_sets = [set(df["date"].tolist()) for df in strategy_frames.values()]
    common_dates = sorted(set.intersection(*date_sets))

    aligned = {}
    for strategy, sdf in strategy_frames.items():
        subset = sdf[sdf["date"].isin(common_dates)].copy()
        aligned[strategy] = rebase_series(subset)
    return aligned



def compute_metrics(strategy_df):
    n_months = int(len(strategy_df))
    if n_months == 0:
        return {
            "final_nav_gross": np.nan,
            "final_nav_net": np.nan,
            "annualized_return_gross": np.nan,
            "annualized_return_net": np.nan,
            "annualized_excess_return_gross": np.nan,
            "annualized_excess_return_net": np.nan,
            "annualized_volatility_gross": np.nan,
            "annualized_volatility_net": np.nan,
            "excess_sharpe_gross": np.nan,
            "excess_sharpe_net": np.nan,
            "max_drawdown_gross": np.nan,
            "max_drawdown_net": np.nan,
            "n_months": 0,
            "avg_n_holdings": np.nan,
            "avg_monthly_turnover": np.nan,
        }

    final_nav_gross = float(strategy_df["nav_gross"].iloc[-1])
    final_nav_net = float(strategy_df["nav_net"].iloc[-1])

    annualized_return_gross = final_nav_gross ** (12.0 / n_months) - 1.0
    annualized_return_net = final_nav_net ** (12.0 / n_months) - 1.0

    annualized_excess_return_gross = float(strategy_df["excess_return_gross"].mean() * 12.0)
    annualized_excess_return_net = float(strategy_df["excess_return_net"].mean() * 12.0)

    vol_gross = strategy_df["gross_return"].std(ddof=1)
    vol_net = strategy_df["net_return"].std(ddof=1)
    annualized_volatility_gross = float(vol_gross * np.sqrt(12.0)) if pd.notna(vol_gross) else np.nan
    annualized_volatility_net = float(vol_net * np.sqrt(12.0)) if pd.notna(vol_net) else np.nan

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
        "annualized_excess_return_gross": annualized_excess_return_gross,
        "annualized_excess_return_net": annualized_excess_return_net,
        "annualized_volatility_gross": annualized_volatility_gross,
        "annualized_volatility_net": annualized_volatility_net,
        "excess_sharpe_gross": excess_sharpe_gross,
        "excess_sharpe_net": excess_sharpe_net,
        "max_drawdown_gross": float(strategy_df["drawdown_gross"].min()),
        "max_drawdown_net": float(strategy_df["drawdown_net"].min()),
        "n_months": n_months,
        "avg_n_holdings": float(strategy_df["n_holdings"].mean()),
        "avg_monthly_turnover": float(strategy_df["turnover"].mean()),
    }



def build_performance_table(strategy_frames, diagnostic_note_map):
    rows = []
    for strategy, sdf in strategy_frames.items():
        metrics = compute_metrics(sdf)
        rows.append(
            {
                "strategy": strategy,
                "diagnostic_note": diagnostic_note_map.get(strategy, ""),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    return out[
        [
            "strategy",
            "diagnostic_note",
            "final_nav_gross",
            "final_nav_net",
            "annualized_return_gross",
            "annualized_return_net",
            "annualized_excess_return_gross",
            "annualized_excess_return_net",
            "annualized_volatility_gross",
            "annualized_volatility_net",
            "excess_sharpe_gross",
            "excess_sharpe_net",
            "max_drawdown_gross",
            "max_drawdown_net",
            "n_months",
            "avg_n_holdings",
            "avg_monthly_turnover",
        ]
    ]



def plot_gross_net(strategy_frames, value_col_gross, value_col_net, title, y_label, out_path):
    plt.figure(figsize=(12, 7))
    for strategy, sdf in strategy_frames.items():
        plt.plot(sdf["date"], sdf[value_col_gross], label=f"{strategy} gross", linewidth=1.8)
        plt.plot(
            sdf["date"],
            sdf[value_col_net],
            label=f"{strategy} net",
            linewidth=1.6,
            linestyle="--",
        )
    plt.title(title)
    plt.xlabel("Formation Date")
    plt.ylabel(y_label)
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def main():
    month_end_freq = get_month_end_freq()
    monthly_rf = prepare_monthly_rf()

    panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    require_columns(panel, ["ticker", "date", "future_ret_1m", "label"], PANEL_PATH.name)

    universe = pd.read_csv(UNIVERSE_PATH)
    require_columns(universe, ["ticker"], UNIVERSE_PATH.name)
    universe_tickers = sorted(universe["ticker"].astype(str).str.upper().unique().tolist())

    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel = panel[panel["ticker"].isin(universe_tickers)].copy()
    panel = panel[~panel["ticker"].isin(BENCHMARK_TICKERS)].copy()
    panel = panel.dropna(subset=["future_ret_1m", "label"]).sort_values(["date", "ticker"])

    strategy_equal = "Equal-Weight REIT Portfolio"
    strategy_oracle_increase = "Oracle Increase-Only"
    strategy_oracle_reduce_avoid = "Oracle Reduce-Avoidance"

    gross_reit_equal = build_reit_gross_series(
        panel=panel,
        all_universe_tickers=universe_tickers,
        strategy_name=strategy_equal,
        selector=lambda g: g["ticker"].tolist(),
    )
    gross_oracle_inc = build_reit_gross_series(
        panel=panel,
        all_universe_tickers=universe_tickers,
        strategy_name=strategy_oracle_increase,
        selector=lambda g: g.loc[g["label"] == "increase", "ticker"].tolist(),
    )
    gross_oracle_not_red = build_reit_gross_series(
        panel=panel,
        all_universe_tickers=universe_tickers,
        strategy_name=strategy_oracle_reduce_avoid,
        selector=lambda g: g.loc[g["label"] != "reduce", "ticker"].tolist(),
    )

    daily = pd.read_csv(DAILY_PRICES_PATH)
    date_col = "Date" if "Date" in daily.columns else "date"
    if "Adj Close" in daily.columns:
        price_col = "Adj Close"
    elif "adj_close" in daily.columns:
        price_col = "adj_close"
    elif "Close" in daily.columns:
        price_col = "Close"
    else:
        raise ValueError("daily_prices.csv missing adjusted/close price column")

    require_columns(daily, [date_col, price_col, "ticker"], DAILY_PRICES_PATH.name)
    daily = daily.rename(columns={date_col: "date", price_col: "adj_close"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ticker"] = daily["ticker"].astype(str).str.upper()

    gross_bench_spy = build_benchmark_gross_series(daily, "SPY", month_end_freq)
    gross_bench_vnq = build_benchmark_gross_series(daily, "VNQ", month_end_freq)
    gross_bench_xlre = build_benchmark_gross_series(daily, "XLRE", month_end_freq)

    diagnostic_note_oracle = "Diagnostic only; non-implementable (uses labels constructed from future_ret_1m)."

    implementable_full = {
        "Benchmark SPY": finalize_strategy_series(gross_bench_spy, monthly_rf, "implementable", ""),
        "Benchmark VNQ": finalize_strategy_series(gross_bench_vnq, monthly_rf, "implementable", ""),
        "Benchmark XLRE": finalize_strategy_series(gross_bench_xlre, monthly_rf, "implementable", ""),
        strategy_equal: finalize_strategy_series(gross_reit_equal, monthly_rf, "implementable", ""),
    }

    oracle_full = {
        strategy_oracle_increase: finalize_strategy_series(
            gross_oracle_inc,
            monthly_rf,
            "oracle_diagnostic_non_implementable",
            diagnostic_note_oracle,
        ),
        strategy_oracle_reduce_avoid: finalize_strategy_series(
            gross_oracle_not_red,
            monthly_rf,
            "oracle_diagnostic_non_implementable",
            diagnostic_note_oracle,
        ),
    }

    implementable_common = align_common_period(implementable_full)

    implementable_test = {}
    for strategy, sdf in implementable_common.items():
        subset = sdf[(sdf["date"] >= TEST_START) & (sdf["date"] <= TEST_END)].copy()
        implementable_test[strategy] = rebase_series(subset)

    note_map_impl = {k: "" for k in implementable_common.keys()}
    note_map_oracle = {k: diagnostic_note_oracle for k in oracle_full.keys()}

    table_impl_common = build_performance_table(implementable_common, note_map_impl)
    table_impl_test = build_performance_table(implementable_test, note_map_impl)
    table_oracle = build_performance_table(oracle_full, note_map_oracle)

    table_impl_common.to_csv(OUT_IMPL_COMMON, index=False)
    table_impl_test.to_csv(OUT_IMPL_TEST, index=False)
    table_oracle.to_csv(OUT_ORACLE, index=False)

    monthly_rows = []
    for frames in [implementable_full, oracle_full]:
        for _, sdf in frames.items():
            monthly_rows.append(sdf.copy())

    monthly_out = pd.concat(monthly_rows, ignore_index=True)
    monthly_out = monthly_out[
        [
            "date",
            "strategy",
            "strategy_group",
            "diagnostic_note",
            "gross_return",
            "net_return",
            "monthly_rf",
            "excess_return_gross",
            "excess_return_net",
            "nav_gross",
            "nav_net",
            "drawdown_gross",
            "drawdown_net",
            "n_holdings",
            "turnover",
        ]
    ].sort_values(["date", "strategy"])
    monthly_out.to_csv(OUT_MONTHLY, index=False)

    plot_gross_net(
        implementable_common,
        value_col_gross="nav_gross",
        value_col_net="nav_net",
        title="Implementable Baselines: Cumulative NAV (Common Period, Gross vs Net)",
        y_label="NAV",
        out_path=OUT_NAV_IMPL_COMMON,
    )
    plot_gross_net(
        implementable_common,
        value_col_gross="drawdown_gross",
        value_col_net="drawdown_net",
        title="Implementable Baselines: Drawdown (Common Period, Gross vs Net)",
        y_label="Drawdown",
        out_path=OUT_DD_IMPL_COMMON,
    )
    plot_gross_net(
        oracle_full,
        value_col_gross="nav_gross",
        value_col_net="nav_net",
        title="Oracle Diagnostic Strategies: Cumulative NAV (Gross vs Net)",
        y_label="NAV",
        out_path=OUT_NAV_ORACLE,
    )
    plot_gross_net(
        oracle_full,
        value_col_gross="drawdown_gross",
        value_col_net="drawdown_net",
        title="Oracle Diagnostic Strategies: Drawdown (Gross vs Net)",
        y_label="Drawdown",
        out_path=OUT_DD_ORACLE,
    )

    print("Saved:", OUT_IMPL_COMMON)
    print("Saved:", OUT_IMPL_TEST)
    print("Saved:", OUT_ORACLE)
    print("Saved:", OUT_MONTHLY)
    print("Saved:", OUT_NAV_IMPL_COMMON)
    print("Saved:", OUT_DD_IMPL_COMMON)
    print("Saved:", OUT_NAV_ORACLE)
    print("Saved:", OUT_DD_ORACLE)

    print("\nImplementable Performance (Common Period, Gross/Net)")
    print(table_impl_common.to_string(index=False))
    print("\nImplementable Performance (Test Period, Gross/Net)")
    print(table_impl_test.to_string(index=False))
    print("\nOracle Diagnostic Performance (Gross/Net)")
    print(table_oracle.to_string(index=False))


if __name__ == "__main__":
    main()
