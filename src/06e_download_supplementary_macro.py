"""
06e_download_supplementary_macro.py
====================================
Download supplementary macro indicators from FRED that are critical for REIT pricing
but missing from the original panel.

New indicators:
- MORTGAGE30US: 30-Year Fixed Mortgage Rate (weekly → monthly avg)
- BAMLC0A4CBBB: BBB Corporate Bond OAS Spread (daily → monthly avg)
- BAMLH0A0HYM2: High Yield Bond OAS Spread (daily → monthly avg)
- CSUSHPISA: Case-Shiller Home Price Index (monthly, seasonally adjusted)
- HOUST: Housing Starts (monthly, thousands)
- PERMIT: Building Permits (monthly, thousands)
- INDPRO: Industrial Production Index (monthly)
- DSPIC96: Real Disposable Personal Income (monthly, billions 2017$)

Output: data/raw/macro/fred_supplementary_macro.csv

Zone: NEW (does not touch any Zone 1 file)
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/06e_download_supplementary_macro.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not found in .env")

# --- Configuration ---
FRED_SERIES = {
    "MORTGAGE30US": {
        "description": "30-Year Fixed Rate Mortgage Average",
        "frequency": "weekly",
        "agg": "avg",
    },
    "BAA10Y": {
        "description": "Moody's Baa Corporate Bond Yield Spread to 10Y Treasury",
        "frequency": "daily",
        "agg": "avg",
    },
    "BAA": {
        "description": "Moody's Seasoned Baa Corporate Bond Yield",
        "frequency": "monthly",
        "agg": None,
    },
    "CSUSHPISA": {
        "description": "S&P/Case-Shiller U.S. National Home Price Index (SA)",
        "frequency": "monthly",
        "agg": None,
    },
    "HOUST": {
        "description": "Housing Starts: Total (Thousands)",
        "frequency": "monthly",
        "agg": None,
    },
    "PERMIT": {
        "description": "New Privately-Owned Housing Units Authorized (Thousands)",
        "frequency": "monthly",
        "agg": None,
    },
    "INDPRO": {
        "description": "Industrial Production: Total Index",
        "frequency": "monthly",
        "agg": None,
    },
    "DSPIC96": {
        "description": "Real Disposable Personal Income (Billions, 2017$)",
        "frequency": "monthly",
        "agg": None,
    },
}

START_DATE = "2014-01-01"  # Extra buffer for lag computation
END_DATE = "2025-12-31"

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "fred_supplementary_macro.csv"


class TruncatedDownloadError(Exception):
    """Raised when a FRED response appears truncated (silent partial download)."""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, TruncatedDownloadError)),
    reraise=True,
)
def download_single_series(fred: Fred, series_id: str, start: str, end: str, agg: str | None) -> pd.Series:
    """
    Download a FRED series, aggregated to monthly frequency SERVER-SIDE.

    Root-cause fix for SSL truncation: large daily series (e.g. credit spreads
    with ~7,500 daily observations) get silently truncated by an unstable SSL
    connection — fredapi returns only a partial tail without raising. By asking
    FRED to aggregate to monthly frequency on the server (frequency='m'), each
    response is only ~144 observations regardless of the native frequency, so
    the payload stays tiny and the connection completes reliably.

    For series that are already monthly, agg=None and we still pass frequency='m'
    (a no-op that returns the native monthly values).
    """
    # FRED aggregation_method: 'avg', 'sum', or 'eop' (end of period). Default 'avg'.
    aggregation_method = agg if agg is not None else "avg"
    raw = fred.get_series(
        series_id,
        observation_start=start,
        observation_end=end,
        frequency="m",
        aggregation_method=aggregation_method,
    )
    raw = raw.dropna()

    if raw.empty:
        raise TruncatedDownloadError(f"{series_id}: empty response")

    requested_start = pd.Timestamp(start)
    actual_start = raw.index.min()
    # Allow up to ~95 days slack. A start later than this for a series requested
    # from 2014 (all our targets have pre-2014 history) signals a truncated download.
    if (actual_start - requested_start).days > 95:
        raise TruncatedDownloadError(
            f"{series_id}: data starts {actual_start.date()} but requested "
            f"{requested_start.date()} — likely truncated download"
        )
    return raw


def download_and_resample() -> pd.DataFrame:
    """Download all FRED series (server-side monthly aggregation) and align to month-end."""
    fred = Fred(api_key=FRED_API_KEY)
    monthly_frames = []

    for series_id, meta in FRED_SERIES.items():
        logger.info(f"Downloading {series_id}: {meta['description']}")
        try:
            raw = download_single_series(
                fred, series_id, START_DATE, END_DATE, agg=meta["agg"]
            )
            logger.info(f"  → {len(raw)} observations ({raw.index.min()} to {raw.index.max()})")

            # Sleep between requests to be respectful to FRED API
            time.sleep(0.5)

            # FRED returns monthly data indexed at the first of the month.
            # Normalize to month-end to match the project panel (e.g. 2015-02-28).
            monthly = raw.resample("M").last()
            monthly.name = series_id
            monthly_frames.append(monthly)

        except Exception as e:
            logger.error(f"  ✗ Failed to download {series_id}: {e}")
            continue

    if not monthly_frames:
        raise RuntimeError("No series downloaded successfully")

    # Combine all series
    df = pd.concat(monthly_frames, axis=1)
    df.index.name = "date"

    logger.info(f"\nCombined panel: {df.shape[0]} months × {df.shape[1]} series")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    return df


def main():
    logger.info("=" * 60)
    logger.info("06e: Download Supplementary Macro Indicators from FRED")
    logger.info("=" * 60)

    df = download_and_resample()

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_PATH)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nSample (last 5 rows):\n{df.tail()}")

    logger.info("\n✓ Done.")


if __name__ == "__main__":
    main()
