"""
06f_build_supplementary_features.py
====================================
Build derived features from the supplementary macro indicators.

Input: data/raw/macro/fred_supplementary_macro.csv
Output: data/processed/supplementary_macro_features.csv

Derived features:
- mortgage_spread: MORTGAGE30US - DGS10 (mortgage risk premium)
- credit_spread_bbb: BAMLC0A4CBBB (already a spread)
- credit_spread_hy: BAMLH0A0HYM2 (already a spread)
- mortgage30_chg_1m: 1-month change in mortgage rate
- mortgage30_chg_3m: 3-month change in mortgage rate
- credit_bbb_chg_1m: 1-month change in BBB spread
- credit_hy_chg_1m: 1-month change in HY spread
- home_price_yoy: Case-Shiller YoY % change
- home_price_mom: Case-Shiller MoM % change
- housing_starts_yoy: Housing starts YoY % change
- permits_yoy: Building permits YoY % change
- indpro_yoy: Industrial production YoY % change
- real_income_yoy: Real disposable income YoY % change

All features get a _lag1 version (1-month lag to prevent look-ahead bias).

Zone: NEW (does not touch any Zone 1 file)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/06f_build_supplementary_features.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_SUPPLEMENTARY = PROJECT_ROOT / "data" / "raw" / "macro" / "fred_supplementary_macro.csv"
INPUT_ORIGINAL_MACRO = PROJECT_ROOT / "data" / "raw" / "macro" / "fred_macro.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "supplementary_macro_features.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load supplementary and original macro data."""
    supp = pd.read_csv(INPUT_SUPPLEMENTARY, index_col="date", parse_dates=True)
    logger.info(f"Supplementary macro: {supp.shape[0]} rows × {supp.shape[1]} cols")

    orig = pd.read_csv(INPUT_ORIGINAL_MACRO, index_col="date", parse_dates=True)
    logger.info(f"Original macro: {orig.shape[0]} rows × {orig.shape[1]} cols")

    return supp, orig


def compute_derived_features(supp: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features from supplementary macro data."""
    features = pd.DataFrame(index=supp.index)

    # --- Raw levels (useful for LLM context) ---
    features["mortgage30"] = supp["MORTGAGE30US"]
    features["baa_spread_10y"] = supp["BAA10Y"]
    features["baa_yield"] = supp["BAA"]
    features["home_price_index"] = supp["CSUSHPISA"]
    features["housing_starts"] = supp["HOUST"]
    features["building_permits"] = supp["PERMIT"]
    features["industrial_production"] = supp["INDPRO"]
    features["real_disp_income"] = supp["DSPIC96"]

    # --- Mortgage spread: mortgage rate minus 10Y Treasury ---
    # Need DGS10 from original macro, resampled to month-end
    if "DGS10" in orig.columns:
        dgs10_monthly = orig["DGS10"].resample("M").last()
        # Align indexes
        aligned_dgs10 = dgs10_monthly.reindex(supp.index, method="ffill")
        features["mortgage_spread"] = supp["MORTGAGE30US"] - aligned_dgs10
    else:
        logger.warning("DGS10 not found in original macro — mortgage_spread skipped")

    # --- Rate changes ---
    features["mortgage30_chg_1m"] = supp["MORTGAGE30US"].diff(1)
    features["mortgage30_chg_3m"] = supp["MORTGAGE30US"].diff(3)
    features["baa_spread_chg_1m"] = supp["BAA10Y"].diff(1)
    features["baa_spread_chg_3m"] = supp["BAA10Y"].diff(3)
    features["baa_yield_chg_1m"] = supp["BAA"].diff(1)
    features["baa_yield_chg_3m"] = supp["BAA"].diff(3)

    # --- YoY growth rates ---
    features["home_price_yoy"] = supp["CSUSHPISA"].pct_change(12)
    features["home_price_mom"] = supp["CSUSHPISA"].pct_change(1)
    features["housing_starts_yoy"] = supp["HOUST"].pct_change(12)
    features["permits_yoy"] = supp["PERMIT"].pct_change(12)
    features["indpro_yoy"] = supp["INDPRO"].pct_change(12)
    features["real_income_yoy"] = supp["DSPIC96"].pct_change(12)

    # --- Lag all features by 1 month (prevent look-ahead bias) ---
    lag1_features = features.shift(1)
    lag1_features.columns = [f"{c}_lag1" for c in lag1_features.columns]

    # Combine current + lagged
    result = pd.concat([features, lag1_features], axis=1)
    result.index.name = "date"

    return result


def main():
    logger.info("=" * 60)
    logger.info("06f: Build Supplementary Macro Features")
    logger.info("=" * 60)

    supp, orig = load_data()
    features = compute_derived_features(supp, orig)

    logger.info(f"\nFinal features panel: {features.shape[0]} rows × {features.shape[1]} cols")
    logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
    logger.info(f"\nMissing values (top 10):\n{features.isnull().sum().sort_values(ascending=False).head(10)}")
    logger.info(f"\nFeature list:\n{list(features.columns)}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")

    logger.info("\n✓ Done.")


if __name__ == "__main__":
    main()
