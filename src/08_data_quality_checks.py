from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PANEL_PATH = ROOT / "data" / "processed" / "reit_monthly_panel.csv"
BT_PATH = ROOT / "data" / "processed" / "backtest_ready_panel.csv"
TEXT_PATH = ROOT / "data" / "interim" / "extracted_filing_text.csv"
OUT_PATH = ROOT / "data" / "processed" / "data_quality_report.csv"

rows = []

panel = pd.read_csv(PANEL_PATH)
panel["ticker"] = panel["ticker"].astype(str).str.upper()

rows.append({"check": "panel_rows", "value": len(panel)})
rows.append({"check": "panel_tickers", "value": panel["ticker"].nunique()})
rows.append({"check": "panel_start_date", "value": panel["date"].min()})
rows.append({"check": "panel_end_date", "value": panel["date"].max()})

for label, count in panel["label"].value_counts(dropna=False).items():
    rows.append({"check": f"label_count_{label}", "value": count})

for col, missing_rate in panel.isna().mean().items():
    rows.append({"check": f"missing_rate_{col}", "value": round(float(missing_rate), 4)})

lag_cols = [
    "FEDFUNDS_lag1",
    "DGS10_lag1",
    "DGS2_lag1",
    "term_spread_10y_2y_lag1",
    "cpi_yoy_lag1",
    "UNRATE_lag1",
]

lag_cols_present = all(col in panel.columns for col in lag_cols)
rows.append({"check": "macro_lag_columns_present", "value": bool(lag_cols_present)})

for col in lag_cols:
    rows.append({"check": f"lag_col_exists_{col}", "value": bool(col in panel.columns)})
    if col in panel.columns:
        rows.append(
            {
                "check": f"missing_rate_{col}",
                "value": round(float(panel[col].isna().mean()), 4),
            }
        )

monthly_rf_present = "monthly_rf" in panel.columns
rows.append({"check": "monthly_rf_present", "value": bool(monthly_rf_present)})
if monthly_rf_present:
    rows.append(
        {
            "check": "missing_rate_monthly_rf",
            "value": round(float(panel["monthly_rf"].isna().mean()), 4),
        }
    )

benchmark_etfs = {"SPY", "VNQ", "XLRE"}
benchmark_present_count = int(panel["ticker"].isin(benchmark_etfs).sum())
rows.append({"check": "benchmark_etf_rows_in_panel", "value": benchmark_present_count})
rows.append({"check": "benchmark_etfs_excluded", "value": benchmark_present_count == 0})

missing_future_ret = int(panel["future_ret_1m"].isna().sum())
rows.append({"check": "missing_future_ret_1m_count", "value": missing_future_ret})
rows.append({"check": "missing_future_ret_1m_zero", "value": missing_future_ret == 0})

if BT_PATH.exists():
    bt = pd.read_csv(BT_PATH)
    rows.append({"check": "backtest_rows", "value": len(bt)})
    rows.append({"check": "backtest_tickers", "value": bt["ticker"].nunique()})
    rows.append({"check": "backtest_start_date", "value": bt["date"].min()})
    rows.append({"check": "backtest_end_date", "value": bt["date"].max()})

if TEXT_PATH.exists():
    text = pd.read_csv(TEXT_PATH)
    rows.append({"check": "filing_text_rows", "value": len(text)})
    rows.append({"check": "filing_text_tickers", "value": text["ticker"].nunique()})

    if "item_1a_chars" in text.columns:
        rows.append({"check": "item_1a_chars_mean", "value": round(float(text["item_1a_chars"].mean()), 2)})
        rows.append({"check": "item_1a_chars_median", "value": round(float(text["item_1a_chars"].median()), 2)})
        rows.append({"check": "item_1a_missing_or_zero", "value": int((text["item_1a_chars"] == 0).sum())})

report = pd.DataFrame(rows)
report.to_csv(OUT_PATH, index=False)

print(report.head(60))
print(f"Saved to {OUT_PATH}")
