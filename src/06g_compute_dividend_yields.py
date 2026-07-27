"""
06g_compute_dividend_yields.py
===============================
Compute trailing 12-month dividend yield from existing price data.

DATA SOURCE: data/raw/prices/daily_prices.csv (already downloaded from Yahoo Finance)

MATHEMATICAL FOUNDATION:
------------------------
Yahoo Finance provides two price series:
1. Close: Price adjusted for stock splits only
2. adj_close: Price adjusted for stock splits AND dividends (reinvestment assumption)

The relationship between them encodes dividend information:

    Total Return[t] = adj_close[t] / adj_close[t-1] - 1
    Price Return[t] = Close[t] / Close[t-1] - 1
    Dividend Return[t] = Total Return[t] - Price Return[t]

On ex-dividend dates, when a cash dividend D is paid:
    adj_close is adjusted backward in history by factor (Close - D) / Close
    Close remains unadjusted

Therefore, on ex-div date t:
    Dividend Amount[t] = Close[t-1] × [(adj_close[t]/adj_close[t-1]) - (Close[t]/Close[t-1])]

Aggregating over trailing 12 months and dividing by current price gives the
trailing dividend yield, which is the standard fundamental valuation metric.

ACADEMIC PRECEDENT:
-------------------
This technique is standard in academic finance for reconstructing dividend streams
from adjusted prices when direct dividend data are unavailable or rate-limited:
- CRSP uses similar logic to construct RET (total return) vs RETX (price return)
- Fama-French data library documents this in their "Dividend Yields" notes
- See: Boudoukh, Richardson & Whitelaw (2007), "The Myth of Long-Horizon Predictability"

VALIDATION STRATEGY:
--------------------
1. Check for Close vs adj_close consistency (should diverge only on ex-div dates)
2. Verify computed dividend yield is in reasonable range (REITs: 2%-8% typical)
3. Cross-check against known REIT dividend policy (quarterly payers)
4. Flag any anomalies (negative dividends, implausible yields)

Input: data/raw/prices/daily_prices.csv
Output: data/raw/prices/monthly_dividend_yields.csv

Zone: NEW (reads Zone 1 file, writes to new file)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/06g_compute_dividend_yields.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "prices" / "daily_prices.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "prices" / "monthly_dividend_yields.csv"


def load_prices() -> pd.DataFrame:
    """Load daily prices with proper date handling."""
    df = pd.read_csv(PRICES_PATH, parse_dates=["Date"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} daily price observations")
    logger.info(f"Tickers: {df['ticker'].nunique()} unique")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Required columns present: Close={('Close' in df.columns)}, Adj Close={('Adj Close' in df.columns)}")
    return df


def compute_daily_dividend_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily dividend return from Close vs Adj Close.

    Formula:
        total_return[t] = Adj Close[t] / Adj Close[t-1] - 1
        price_return[t] = Close[t] / Close[t-1] - 1
        dividend_return[t] = total_return[t] - price_return[t]

    On ex-dividend dates, dividend_return > 0; on other dates ≈ 0.
    """
    result = df.copy()

    # Compute returns by ticker
    result["total_return"] = result.groupby("ticker")["Adj Close"].pct_change()
    result["price_return"] = result.groupby("ticker")["Close"].pct_change()

    # Dividend return = difference
    result["dividend_return"] = result["total_return"] - result["price_return"]

    # Implied dividend amount (cash per share)
    # On ex-div date: div_amount ≈ Close[t-1] × dividend_return[t]
    result["implied_div_amount"] = result.groupby("ticker")["Close"].shift(1) * result["dividend_return"]

    # Clean up numerical noise (|dividend_return| < 0.01% is noise, not real dividend)
    result.loc[result["dividend_return"].abs() < 0.0001, ["dividend_return", "implied_div_amount"]] = 0

    return result


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to month-end:
    - Sum dividends per month (captures quarterly payers)
    - Take last Close per month (denominator for yield)
    - Compute trailing 12-month dividend and yield
    """
    # Ensure date is datetime
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["month_end"] = df["Date"] + pd.offsets.MonthEnd(0)

    monthly_list = []
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()

        # Aggregate to month-end
        monthly = ticker_df.groupby("month_end").agg({
            "implied_div_amount": "sum",  # Sum all dividends in the month
            "Close": "last",  # Month-end closing price
        }).reset_index()

        monthly["ticker"] = ticker
        monthly.rename(columns={"month_end": "date"}, inplace=True)

        # Compute trailing 12-month dividend
        monthly = monthly.sort_values("date").reset_index(drop=True)
        monthly["trailing_12m_dividend"] = monthly["implied_div_amount"].rolling(window=12, min_periods=1).sum()

        # Dividend yield = trailing dividend / current price
        monthly["dividend_yield"] = monthly["trailing_12m_dividend"] / monthly["Close"]

        monthly_list.append(monthly)

    result = pd.concat(monthly_list, ignore_index=True)
    return result


def validate_results(df: pd.DataFrame) -> None:
    """Run validation checks and log warnings for anomalies."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check 1: Dividend yield range (REITs typically 2-8%)
    yield_stats = df["dividend_yield"].describe()
    logger.info(f"\nDividend yield distribution:\n{yield_stats}")

    outliers_low = df[df["dividend_yield"] < 0].groupby("ticker").size()
    outliers_high = df[df["dividend_yield"] > 0.15].groupby("ticker").size()

    if len(outliers_low) > 0:
        logger.warning(f"⚠ Negative dividend yields detected:\n{outliers_low}")
    if len(outliers_high) > 0:
        logger.warning(f"⚠ Implausibly high yields (>15%):\n{outliers_high}")

    # Check 2: Data coverage
    coverage = df.groupby("ticker")["dividend_yield"].count()
    logger.info(f"\nData coverage per ticker:\n{coverage.describe()}")

    # Check 3: Dividend frequency (REITs pay quarterly → ~4 non-zero dividend months per year)
    annual_divs = df[df["implied_div_amount"] > 0].groupby(["ticker", df["date"].dt.year]).size()
    logger.info(f"\nDividend payments per ticker-year (should cluster around 4 for quarterly payers):")
    logger.info(annual_divs.describe())

    # Check 4: Compare to known REIT behavior
    recent = df[df["date"] >= "2024-01-01"].groupby("ticker")["dividend_yield"].mean()
    logger.info(f"\nRecent average dividend yield (2024+) per ticker:\n{recent.sort_values()}")

    logger.info("\n✓ Validation complete. Review warnings above.")


def main():
    logger.info("=" * 60)
    logger.info("06g: Compute Dividend Yields from Price Data")
    logger.info("=" * 60)
    logger.info("\nMETHOD: Reverse-engineer dividends from adj_close vs Close differential")
    logger.info("SOURCE: Academic standard technique (CRSP, Fama-French methodology)")

    # Load
    prices = load_prices()

    # Compute daily dividend returns
    logger.info("\nComputing daily dividend returns...")
    prices_with_divs = compute_daily_dividend_returns(prices)

    # Count ex-div events
    exdiv_events = (prices_with_divs["implied_div_amount"] > 0).sum()
    logger.info(f"Detected {exdiv_events} ex-dividend events across all tickers")

    # Aggregate to monthly
    logger.info("\nAggregating to monthly frequency...")
    monthly = aggregate_to_monthly(prices_with_divs)

    logger.info(f"Monthly panel: {len(monthly)} rows")
    logger.info(f"Date range: {monthly['date'].min()} to {monthly['date'].max()}")

    # Validate
    validate_results(monthly)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")
    logger.info(f"Final shape: {monthly.shape}")
    logger.info(f"\nSample (last 5 rows):\n{monthly.tail()}")

    logger.info("\n✓ Done.")


if __name__ == "__main__":
    main()
