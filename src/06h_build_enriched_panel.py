"""
06h_build_enriched_panel.py
============================
Merge supplementary macro features and dividend yields into the original
backtest-ready panel, producing an enriched panel.

IMPORTANT: This script READS the original panel but NEVER modifies it.
The original `backtest_ready_panel.csv` stays untouched to preserve V5
reproducibility. Output goes to a new file.

Inputs (all read-only):
- data/processed/backtest_ready_panel.csv (Zone 1, read-only)
- data/processed/supplementary_macro_features.csv (new)
- data/raw/prices/monthly_dividend_yields.csv (new)

Output:
- data/processed/backtest_ready_panel_enriched.csv

Zone: NEW (reads Zone 1, writes only to new file)
"""

import logging
from pathlib import Path

import pandas as pd

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/06h_build_enriched_panel.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PANEL = PROJECT_ROOT / "data" / "processed" / "backtest_ready_panel.csv"
SUPP_FEATURES = PROJECT_ROOT / "data" / "processed" / "supplementary_macro_features.csv"
DIV_YIELDS = PROJECT_ROOT / "data" / "raw" / "prices" / "monthly_dividend_yields.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_ready_panel_enriched.csv"


def main():
    logger.info("=" * 60)
    logger.info("06h: Build Enriched Panel")
    logger.info("=" * 60)

    # --- Load original panel (READ ONLY) ---
    panel = pd.read_csv(ORIGINAL_PANEL, parse_dates=["date"])
    original_cols = set(panel.columns)
    original_rows = len(panel)
    logger.info(f"Original panel: {original_rows} rows × {len(original_cols)} cols")

    # --- Merge supplementary macro features (by date) ---
    supp = pd.read_csv(SUPP_FEATURES, parse_dates=["date"])
    logger.info(f"Supplementary macro: {len(supp)} rows × {len(supp.columns)} cols")
    panel = panel.merge(supp, on="date", how="left")
    logger.info(f"After macro merge: {len(panel)} rows × {len(panel.columns)} cols")

    # --- Merge dividend yields (by ticker + date) ---
    divs = pd.read_csv(DIV_YIELDS, parse_dates=["date"])
    logger.info(f"Dividend yields: {len(divs)} rows × {len(divs.columns)} cols")
    # Lag dividend yield by 1 month to be safe (point-in-time)
    divs = divs.sort_values(["ticker", "date"])
    divs["dividend_yield_lag1"] = divs.groupby("ticker")["dividend_yield"].shift(1)
    panel = panel.merge(
        divs[["date", "ticker", "dividend_yield", "dividend_yield_lag1"]],
        on=["date", "ticker"],
        how="left",
    )
    logger.info(f"After dividend merge: {len(panel)} rows × {len(panel.columns)} cols")

    # --- Sanity checks ---
    assert len(panel) == original_rows, (
        f"Row count changed! {original_rows} → {len(panel)}. "
        "Merge introduced duplicates — check join keys."
    )

    new_cols = set(panel.columns) - original_cols
    logger.info(f"\n✓ Row count preserved: {len(panel)}")
    logger.info(f"✓ New columns added: {len(new_cols)}")
    logger.info(f"New columns:\n{sorted(new_cols)}")

    # --- Coverage report for new columns ---
    logger.info("\nNew column coverage (% non-null):")
    for col in sorted(new_cols):
        pct = 100 * panel[col].notna().mean()
        logger.info(f"  {col:35s}: {pct:5.1f}%")

    # --- Save ---
    panel.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved enriched panel to: {OUTPUT_PATH}")
    logger.info(f"Final shape: {panel.shape}")

    logger.info("\n✓ Done. Original panel left untouched.")


if __name__ == "__main__":
    main()
