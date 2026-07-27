from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRICE_SIGNALS = ROOT / "data" / "processed" / "monthly_price_signals.csv"
MACRO_SIGNALS = ROOT / "data" / "processed" / "monthly_macro_signals.csv"
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"
TEXT_PATH = ROOT / "data" / "interim" / "extracted_filing_text.csv"

OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_TICKERS = {"SPY", "VNQ", "XLRE"}

prices = pd.read_csv(PRICE_SIGNALS, parse_dates=["date"])
macro = pd.read_csv(MACRO_SIGNALS, parse_dates=["date"])
universe = pd.read_csv(UNIVERSE_PATH)

universe["ticker"] = universe["ticker"].astype(str).str.upper()
prices["ticker"] = prices["ticker"].astype(str).str.upper()

universe_tickers = set(universe["ticker"])
allowed_tickers = universe_tickers - BENCHMARK_TICKERS

prices = prices[prices["ticker"].isin(allowed_tickers)].copy()

panel = prices.merge(
    universe[["ticker", "company", "sector"]],
    on="ticker",
    how="left",
)

panel = panel.merge(macro, on="date", how="left")

panel = panel.sort_values(["ticker", "date"])
panel["future_ret_1m"] = panel.groupby("ticker")["ret_1m"].shift(-1)

missing_future_before = int(panel["future_ret_1m"].isna().sum())
panel = panel.dropna(subset=["future_ret_1m"]).copy()

threshold = 0.02
panel["label"] = np.select(
    [
        panel["future_ret_1m"] > threshold,
        panel["future_ret_1m"] < -threshold,
    ],
    [
        "increase",
        "reduce",
    ],
    default="hold",
)

if TEXT_PATH.exists():
    text = pd.read_csv(TEXT_PATH)
    text["ticker"] = text["ticker"].astype(str).str.upper()
    text["filing_date"] = pd.to_datetime(text["filing_date"], errors="coerce")
    text = text[text["form"] == "10-K"].copy()
    text = text.sort_values(["ticker", "filing_date"])

    rows = []
    for _, row in panel.iterrows():
        ticker = row["ticker"]
        date = row["date"]

        candidates = text[(text["ticker"] == ticker) & (text["filing_date"] <= date)]

        if len(candidates) > 0:
            latest = candidates.iloc[-1]
            row["latest_item_1a_text"] = latest.get("item_1a_text", "")
            row["latest_item_1a_chars"] = latest.get("item_1a_chars", 0)
            row["latest_filing_date"] = latest.get("filing_date")
        else:
            row["latest_item_1a_text"] = ""
            row["latest_item_1a_chars"] = 0
            row["latest_filing_date"] = pd.NaT

        rows.append(row)

    panel = pd.DataFrame(rows)

out_path = OUT_DIR / "reit_monthly_panel.csv"
panel.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Final shape:", panel.shape)
print("Ticker count:", panel["ticker"].nunique())
print("Date range:", panel["date"].min(), "to", panel["date"].max())
print("Missing future_ret_1m before drop:", missing_future_before)
print("Missing future_ret_1m after drop:", int(panel["future_ret_1m"].isna().sum()))
print("Contains benchmark ETFs:", bool(panel["ticker"].isin(list(BENCHMARK_TICKERS)).any()))
print("Label distribution:")
print(panel["label"].value_counts(dropna=False).to_string())
