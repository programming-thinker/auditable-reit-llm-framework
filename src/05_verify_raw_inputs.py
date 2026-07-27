from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

checks = [
    ROOT / "config" / "reit_universe.csv",
    ROOT / "config" / "config.yaml",
    ROOT / "data" / "raw" / "prices" / "daily_prices.csv",
    ROOT / "data" / "raw" / "macro" / "fred_macro.csv",
    ROOT / "data" / "interim" / "filing_metadata.csv",
]

print("Checking required raw inputs...\n")

all_ok = True

for path in checks:
    if path.exists() and path.stat().st_size > 0:
        print(f"OK: {path.relative_to(ROOT)}")
    else:
        print(f"MISSING OR EMPTY: {path.relative_to(ROOT)}")
        all_ok = False

html_files = list((ROOT / "filings" / "raw_html").glob("*.html"))
if html_files:
    print(f"OK: filings/raw_html/*.html count = {len(html_files)}")
else:
    print("MISSING: filings/raw_html/*.html count = 0")
    all_ok = False

print("\nSummary tables:")

price_path = ROOT / "data" / "raw" / "prices" / "daily_prices.csv"
if price_path.exists() and price_path.stat().st_size > 0:
    prices = pd.read_csv(price_path)
    print("\nPrices:", prices.shape)
    if "ticker" in prices.columns:
        print("Price tickers:", prices["ticker"].nunique())

macro_path = ROOT / "data" / "raw" / "macro" / "fred_macro.csv"
if macro_path.exists() and macro_path.stat().st_size > 0:
    macro = pd.read_csv(macro_path)
    print("\nMacro:", macro.shape)
    print(macro.head())

meta_path = ROOT / "data" / "interim" / "filing_metadata.csv"
if meta_path.exists() and meta_path.stat().st_size > 0:
    meta = pd.read_csv(meta_path)
    print("\nFiling metadata:", meta.shape)
    if "form" in meta.columns:
        print(meta.groupby("form").size())

if all_ok:
    print("\nALL RAW INPUTS READY. You can now run the data processing README.")
else:
    print("\nSome raw inputs are still missing. Fix the missing items before processing.")
