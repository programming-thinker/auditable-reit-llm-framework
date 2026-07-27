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

DAILY_PRICES_PATH = ROOT / "data" / "raw" / "prices" / "daily_prices.csv"
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_BY_TICKER = OUT_TABLE_DIR / "liquidity_diagnostic_by_ticker.csv"
OUT_BY_MONTH = OUT_TABLE_DIR / "liquidity_diagnostic_by_month.csv"
OUT_SUMMARY = OUT_TABLE_DIR / "liquidity_diagnostic_summary.csv"
OUT_FIG = OUT_FIG_DIR / "liquidity_diagnostic_pct_liquid_over_time.png"

LIQUIDITY_THRESHOLD = 5_000_000.0


def require_columns(df, cols, source_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def month_end_dates(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp("M")


def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    require_columns(universe, ["ticker"], UNIVERSE_PATH.name)
    universe_tickers = sorted(universe["ticker"].astype(str).str.upper().unique().tolist())

    daily = pd.read_csv(DAILY_PRICES_PATH)
    date_col = "Date" if "Date" in daily.columns else "date"
    if "Adj Close" in daily.columns:
        price_col = "Adj Close"
    elif "adj_close" in daily.columns:
        price_col = "adj_close"
    elif "Close" in daily.columns:
        price_col = "Close"
    elif "close" in daily.columns:
        price_col = "close"
    else:
        price_col = None

    volume_col = "Volume" if "Volume" in daily.columns else "volume" if "volume" in daily.columns else None

    if price_col is None or volume_col is None:
        note = "Volume data were unavailable or price data were unavailable; liquidity diagnostic could not be computed."
        by_ticker = pd.DataFrame(
            {
                "ticker": universe_tickers,
                "months_observed": 0,
                "pct_months_above_5m": np.nan,
                "median_trailing_3m_avg_daily_dollar_volume": np.nan,
                "passes_full_sample_liquidity": False,
                "note": note,
            }
        )
        by_month = pd.DataFrame(columns=["date", "total_reits", "n_liquid", "pct_liquid", "note"])
        summary = pd.DataFrame(
            [
                {
                    "total_reits": len(universe_tickers),
                    "reits_passing_full_sample_liquidity": 0,
                    "median_pct_months_above_5m": np.nan,
                    "min_pct_months_above_5m": np.nan,
                    "note": note,
                }
            ]
        )
        by_ticker.to_csv(OUT_BY_TICKER, index=False)
        by_month.to_csv(OUT_BY_MONTH, index=False)
        summary.to_csv(OUT_SUMMARY, index=False)
        print(summary.to_string(index=False))
        return

    require_columns(daily, [date_col, price_col, volume_col, "ticker"], DAILY_PRICES_PATH.name)
    daily = daily.rename(columns={date_col: "date", price_col: "price", volume_col: "volume"}).copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily = daily[daily["ticker"].isin(universe_tickers)].copy()
    daily["price"] = pd.to_numeric(daily["price"], errors="coerce")
    daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce")
    daily = daily.dropna(subset=["date", "ticker", "price", "volume"]).copy()
    daily["daily_dollar_volume"] = daily["price"] * daily["volume"]
    daily["month"] = month_end_dates(daily["date"])

    monthly = (
        daily.groupby(["ticker", "month"], as_index=False)
        .agg(avg_daily_dollar_volume=("daily_dollar_volume", "mean"), trading_days=("daily_dollar_volume", "size"))
        .rename(columns={"month": "date"})
        .sort_values(["ticker", "date"])
    )
    monthly["trailing_3m_avg_daily_dollar_volume"] = monthly.groupby("ticker")["avg_daily_dollar_volume"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    monthly["is_liquid"] = monthly["trailing_3m_avg_daily_dollar_volume"] >= LIQUIDITY_THRESHOLD

    by_ticker = (
        monthly.groupby("ticker", as_index=False)
        .agg(
            months_observed=("date", "nunique"),
            pct_months_above_5m=("is_liquid", "mean"),
            median_trailing_3m_avg_daily_dollar_volume=("trailing_3m_avg_daily_dollar_volume", "median"),
        )
        .sort_values("ticker")
    )
    by_ticker["passes_full_sample_liquidity"] = by_ticker["pct_months_above_5m"] >= 1.0
    by_ticker["note"] = "Liquid if trailing 3-month average daily dollar volume is at least USD 5,000,000."

    by_month = (
        monthly.groupby("date", as_index=False)
        .agg(total_reits=("ticker", "nunique"), n_liquid=("is_liquid", "sum"))
        .sort_values("date")
    )
    by_month["pct_liquid"] = by_month["n_liquid"] / by_month["total_reits"]
    by_month["note"] = "Monthly percentage of universe passing USD 5m trailing 3-month average daily dollar volume threshold."

    summary = pd.DataFrame(
        [
            {
                "total_reits": len(universe_tickers),
                "reits_passing_full_sample_liquidity": int(by_ticker["passes_full_sample_liquidity"].sum()),
                "median_pct_months_above_5m": float(by_ticker["pct_months_above_5m"].median()),
                "min_pct_months_above_5m": float(by_ticker["pct_months_above_5m"].min()),
                "note": "Liquidity is based on trailing 3-month average daily dollar volume >= USD 5,000,000 using adjusted close when available.",
            }
        ]
    )

    by_ticker.to_csv(OUT_BY_TICKER, index=False)
    by_month.to_csv(OUT_BY_MONTH, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    plt.figure(figsize=(11, 6))
    plt.plot(by_month["date"], by_month["pct_liquid"], linewidth=2.0)
    plt.title("Percentage of REIT Universe Passing Liquidity Threshold")
    plt.xlabel("Month")
    plt.ylabel("Pct liquid")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=180)
    plt.close()

    print("Saved:", OUT_BY_TICKER)
    print("Saved:", OUT_BY_MONTH)
    print("Saved:", OUT_SUMMARY)
    print("Saved:", OUT_FIG)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
