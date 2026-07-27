from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "raw" / "prices" / "daily_prices.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

prices = pd.read_csv(PRICE_PATH)

date_col = "Date" if "Date" in prices.columns else "date"
prices[date_col] = pd.to_datetime(prices[date_col])

if "Adj Close" in prices.columns:
    price_col = "Adj Close"
elif "adj_close" in prices.columns:
    price_col = "adj_close"
elif "Close" in prices.columns:
    price_col = "Close"
else:
    raise ValueError("Cannot find price column. Expected Adj Close, adj_close, or Close.")

prices = prices.rename(columns={date_col: "date", price_col: "adj_close"})
prices = prices.sort_values(["ticker", "date"])

prices["daily_return"] = prices.groupby("ticker")["adj_close"].pct_change()

try:
    to_offset("ME")
    month_end_freq = "ME"
except ValueError:
    month_end_freq = "M"

monthly = (
    prices.set_index("date")
    .groupby("ticker")
    .resample(month_end_freq)
    .last()
    .drop(columns=["ticker"], errors="ignore")
    .reset_index()
)

monthly = monthly.sort_values(["ticker", "date"])

monthly["ret_1m"] = monthly.groupby("ticker")["adj_close"].pct_change(1)
monthly["ret_3m"] = monthly.groupby("ticker")["adj_close"].pct_change(3)
monthly["ret_6m"] = monthly.groupby("ticker")["adj_close"].pct_change(6)
monthly["ret_12m"] = monthly.groupby("ticker")["adj_close"].pct_change(12)

vol = (
    prices.set_index("date")
    .groupby("ticker")["daily_return"]
    .resample(month_end_freq)
    .std()
    .reset_index()
    .rename(columns={"daily_return": "vol_daily_month"})
)

vol["vol_annualized"] = vol["vol_daily_month"] * np.sqrt(252)

monthly = monthly.merge(
    vol[["ticker", "date", "vol_annualized"]],
    on=["ticker", "date"],
    how="left"
)

monthly["running_max"] = monthly.groupby("ticker")["adj_close"].cummax()
monthly["drawdown"] = monthly["adj_close"] / monthly["running_max"] - 1

monthly.to_csv(OUT_DIR / "monthly_price_signals.csv", index=False)

print(monthly.shape)
print(monthly.head())
