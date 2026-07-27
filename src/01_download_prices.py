from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"
CONFIG_PATH = ROOT / "config" / "config.yaml"
OUT_DIR = ROOT / "data" / "raw" / "prices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not UNIVERSE_PATH.exists():
    raise FileNotFoundError("Missing config/reit_universe.csv. Run src/00_make_project_files.py first.")

if not CONFIG_PATH.exists():
    raise FileNotFoundError("Missing config/config.yaml. Run src/00_make_project_files.py first.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

universe = pd.read_csv(UNIVERSE_PATH)
reit_tickers = universe["ticker"].dropna().astype(str).str.upper().unique().tolist()
benchmark_tickers = cfg["prices"].get("benchmark_tickers", [])
tickers = sorted(set(reit_tickers + benchmark_tickers))

start = cfg["project"]["start_date"]
end = cfg["project"]["end_date"]

all_rows = []
failed = []

for ticker in tickers:
    print(f"Downloading price data for {ticker}...")
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            interval=cfg["prices"].get("interval", "1d"),
            threads=False,
        )

        if df.empty:
            print(f"WARNING: no data for {ticker}")
            failed.append(ticker)
            continue

        df = df.reset_index()

        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if c[0] else c[1] for c in df.columns]

        df["ticker"] = ticker
        all_rows.append(df)

    except Exception as e:
        print(f"ERROR downloading {ticker}: {e}")
        failed.append(ticker)

if not all_rows:
    raise RuntimeError("No price data downloaded. Check internet connection and tickers.")

prices = pd.concat(all_rows, ignore_index=True)

out_path = OUT_DIR / "daily_prices.csv"
prices.to_csv(out_path, index=False)

print(f"\nSaved {out_path}")
print("Shape:", prices.shape)
print("Tickers downloaded:", prices["ticker"].nunique())

if failed:
    print("Failed tickers:", failed)
