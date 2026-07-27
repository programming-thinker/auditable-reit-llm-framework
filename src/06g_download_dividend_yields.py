"""
06g_download_dividend_yields.py
================================
Download historical trailing dividend yield for the 25-REIT universe.

Uses yfinance to retrieve monthly dividend yield data, computed as:
  trailing_12m_dividends / adj_close_at_month_end

This ensures point-in-time correctness: we use the dividend yield
as of month-end, not the current yield.

Input: config/reit_universe.csv (ticker list)
       data/raw/prices/daily_prices.csv (for adj_close alignment)
Output: data/raw/prices/monthly_dividend_yields.csv

Zone: NEW (does not touch any Zone 1 file, only reads from them)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/06g_download_dividend_yields.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_PATH = PROJECT_ROOT / "config" / "reit_universe.csv"
PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "prices" / "daily_prices.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "prices" / "monthly_dividend_yields.csv"


def load_universe() -> list[str]:
    """Load REIT tickers from universe file."""
    df = pd.read_csv(UNIVERSE_PATH)
    tickers = df["ticker"].tolist()
    logger.info(f"Universe: {len(tickers)} REITs")
    return tickers


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def fetch_ticker_data(ticker: str, start: str, end: str) -> tuple[pd.Series, pd.DataFrame] | None:
    """Fetch dividend and price history for one ticker with retry."""
    stock = yf.Ticker(ticker)
    divs = stock.dividends
    if divs.empty:
        return None
    divs = divs.loc[start:end]
    divs.index = divs.index.tz_localize(None)

    hist = stock.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return None
    hist.index = hist.index.tz_localize(None)

    return divs, hist


def download_dividends(tickers: list[str], start: str = "2014-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Download dividend history for all tickers and compute trailing 12-month yield.

    Strategy:
    1. Download dividend actions from yfinance
    2. Compute trailing 12-month dividend sum at each month-end
    3. Divide by month-end adjusted close price
    """
    all_yields = []

    for i, ticker in enumerate(tickers):
        logger.info(f"  [{i+1}/{len(tickers)}] Downloading dividends for {ticker}...")

        # Respectful inter-request delay to avoid rate limiting
        if i > 0:
            time.sleep(2.5)

        try:
            result = fetch_ticker_data(ticker, start, end)
            if result is None:
                logger.warning(f"  No dividend or price data for {ticker}, skipping")
                continue

            divs, hist = result

            # Resample to monthly: sum dividends per month, last close per month
            monthly_divs = divs.resample("M").sum()
            monthly_close = hist["Close"].resample("M").last()

            # Compute trailing 12-month dividend
            trailing_12m_div = monthly_divs.rolling(window=12, min_periods=1).sum()

            # Compute yield
            div_yield = trailing_12m_div / monthly_close
            div_yield = div_yield.dropna()

            # Create DataFrame
            ticker_df = pd.DataFrame({
                "date": div_yield.index,
                "ticker": ticker,
                "trailing_12m_dividend": trailing_12m_div.reindex(div_yield.index),
                "dividend_yield": div_yield.values,
            })
            all_yields.append(ticker_df)

        except Exception as e:
            logger.error(f"  Error for {ticker}: {e}")
            continue

    if not all_yields:
        logger.error("No dividend data downloaded for any ticker!")
        return pd.DataFrame()

    result = pd.concat(all_yields, ignore_index=True)
    return result


def main():
    logger.info("=" * 60)
    logger.info("06g: Download Dividend Yields")
    logger.info("=" * 60)

    tickers = load_universe()
    yields_df = download_dividends(tickers)

    if yields_df.empty:
        logger.error("No data to save. Exiting.")
        return

    logger.info(f"\nResult: {yields_df.shape[0]} rows × {yields_df.shape[1]} cols")
    logger.info(f"Tickers with data: {yields_df['ticker'].nunique()}")
    logger.info(f"Date range: {yields_df['date'].min()} to {yields_df['date'].max()}")
    logger.info(f"\nDividend yield stats:\n{yields_df['dividend_yield'].describe()}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    yields_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")

    logger.info("\n✓ Done.")


if __name__ == "__main__":
    main()
