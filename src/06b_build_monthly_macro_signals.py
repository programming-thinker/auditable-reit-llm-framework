from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

ROOT = Path(__file__).resolve().parents[1]
MACRO_PATH = ROOT / "data" / "raw" / "macro" / "fred_macro.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

macro = pd.read_csv(MACRO_PATH, parse_dates=["date"])

try:
    to_offset("ME")
    month_end_freq = "ME"
except ValueError:
    month_end_freq = "M"

macro_monthly = (
    macro.set_index("date")
    .resample(month_end_freq)
    .last()
    .ffill()
    .reset_index()
)

if "FEDFUNDS" in macro_monthly.columns:
    macro_monthly["fedfunds_chg_1m"] = macro_monthly["FEDFUNDS"].diff(1)
    macro_monthly["fedfunds_chg_3m"] = macro_monthly["FEDFUNDS"].diff(3)

if "DGS10" in macro_monthly.columns:
    macro_monthly["dgs10_chg_1m"] = macro_monthly["DGS10"].diff(1)
    macro_monthly["dgs10_chg_3m"] = macro_monthly["DGS10"].diff(3)

if {"DGS10", "DGS2"}.issubset(macro_monthly.columns):
    macro_monthly["term_spread_10y_2y"] = macro_monthly["DGS10"] - macro_monthly["DGS2"]

if "CPIAUCSL" in macro_monthly.columns:
    macro_monthly["cpi_yoy"] = macro_monthly["CPIAUCSL"].pct_change(12)

lag_candidates = [
    "FEDFUNDS",
    "DGS10",
    "DGS2",
    "UNRATE",
    "CPIAUCSL",
    "term_spread_10y_2y",
    "cpi_yoy",
    "fedfunds_chg_1m",
    "fedfunds_chg_3m",
    "dgs10_chg_1m",
    "dgs10_chg_3m",
]

generated_lag_cols = []
for col in lag_candidates:
    if col in macro_monthly.columns:
        lag_col = f"{col}_lag1"
        macro_monthly[lag_col] = macro_monthly[col].shift(1)
        generated_lag_cols.append(lag_col)

if "FEDFUNDS" in macro_monthly.columns:
    macro_monthly["monthly_rf"] = macro_monthly["FEDFUNDS"] / 100.0 / 12.0
else:
    macro_monthly["monthly_rf"] = pd.NA

out_path = OUT_DIR / "monthly_macro_signals.csv"
macro_monthly.to_csv(out_path, index=False)

print(macro_monthly.shape)
print(macro_monthly.head())
print("Generated lagged macro columns:", generated_lag_cols)
print("monthly_rf exists:", "monthly_rf" in macro_monthly.columns)
