"""Full inventory of the 90-column enriched panel: categorize every column,
empirically measure cross-sectional variation, and document why each was kept
or dropped for the classification model.

Cross-sectional variation = std across REITs WITHIN each month, averaged over
months. A feature with ~0 cross-sectional std takes the same value for every
REIT in a given month -> it cannot discriminate between REITs in a classifier,
even though it may carry time-series information useful for OLS regression.

Read-only. Writes a single inventory CSV to outputs/llm_deepseek_test/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/llm_deepseek_test/feature_inventory_90.csv"

# The 13 numeric + sector actually used by src/11_quant_only_model.py
SELECTED = {
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
    "dividend_yield_lag1", "FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1",
    "term_spread_10y_2y_lag1", "cpi_yoy_lag1", "UNRATE_lag1", "sector",
}

# Non-feature roles
IDENTIFIERS = {"ticker", "date", "company", "latest_filing_date"}
RAW_PRICE = {"adj_close", "Close", "High", "Low", "Open", "Volume",
             "daily_return", "running_max"}
TARGET = {"future_ret_1m", "label"}
TEXT = {"latest_item_1a_text", "latest_item_1a_chars"}
RISK_FREE = {"monthly_rf"}


def categorize(col: str) -> str:
    if col in IDENTIFIERS:
        return "identifier"
    if col in TARGET:
        return "target (leak if used)"
    if col in RAW_PRICE:
        return "raw price (intermediate)"
    if col in TEXT:
        return "text -> LLM layer"
    if col in RISK_FREE:
        return "risk-free (macro)"
    if col == "sector":
        return "firm categorical"
    fl = {"ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
          "dividend_yield", "dividend_yield_lag1"}
    if col in fl:
        return "firm-level signal"
    return "macro variable"


def main() -> None:
    df = pd.read_csv(PANEL)
    rows = []
    for col in df.columns:
        cat = categorize(col)
        cs_std = np.nan
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric and col not in TARGET:
            # mean within-month cross-sectional std across REITs
            cs_std = df.groupby("date")[col].std(ddof=0).mean()
        has_xs = (not np.isnan(cs_std)) and cs_std > 1e-9
        selected = col in SELECTED

        if selected:
            reason = "USED: cross-sectional variation" if has_xs else \
                     "USED: kept from V5 baseline (NO cross-sectional variation)"
        elif cat in ("identifier", "target (leak if used)", "raw price (intermediate)",
                     "text -> LLM layer", "risk-free (macro)"):
            reason = f"excluded: {cat}"
        elif col == "dividend_yield":
            reason = "excluded: contemporaneous (look-ahead) -> use _lag1"
        elif cat == "macro variable":
            reason = ("dropped: macro, NO cross-sectional variation"
                      if not has_xs else "dropped: macro (has xs var? check)")
        else:
            reason = "excluded: not a model feature"

        rows.append({
            "column": col, "category": cat,
            "cross_sectional_std": round(cs_std, 6) if not np.isnan(cs_std) else None,
            "has_xs_variation": has_xs if is_numeric else None,
            "selected_for_classifier": selected, "reason": reason,
        })

    inv = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(OUT, index=False)

    # ---- summary ----
    macro = inv[inv["category"] == "macro variable"]
    firm = inv[inv["category"] == "firm-level signal"]
    print(f"Total columns: {len(inv)}")
    print(f"  Selected for classifier (numeric+sector): {inv['selected_for_classifier'].sum()}")
    print(f"  Macro variables total: {len(macro)}  | with xs variation: {macro['has_xs_variation'].sum()}")
    print(f"  Firm-level signals total: {len(firm)} | with xs variation: {firm['has_xs_variation'].sum()}")
    print()
    print("=== Firm-level signals (the only real classification signal) ===")
    print(firm[["column", "cross_sectional_std", "selected_for_classifier"]].to_string(index=False))
    print()
    print("=== Selected MACRO features: do they actually have xs variation? ===")
    sel_macro = inv[(inv["category"] == "macro variable") & (inv["selected_for_classifier"])]
    print(sel_macro[["column", "cross_sectional_std", "has_xs_variation", "reason"]].to_string(index=False))
    print()
    print("=== How many of the 90 were DROPPED purely for zero xs variation? ===")
    dropped_macro = macro[~macro["selected_for_classifier"]]
    print(f"  {len(dropped_macro)} macro columns dropped; "
          f"{int((~dropped_macro['has_xs_variation'].astype(bool)).sum())} have ZERO xs std")
    print(f"\nFull inventory -> {OUT}")


if __name__ == "__main__":
    main()
