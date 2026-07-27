from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

for folder in [
    "config",
    "data/raw/prices",
    "data/raw/macro",
    "data/raw/sec_submissions",
    "data/interim",
    "data/processed",
    "filings/raw_html",
    "filings/clean_text",
    "logs",
    "outputs/tables",
    "outputs/figures",
]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

reit_universe = [
    {"ticker": "PLD", "company": "Prologis", "sector": "Industrial"},
    {"ticker": "AMT", "company": "American Tower", "sector": "Infrastructure"},
    {"ticker": "EQIX", "company": "Equinix", "sector": "Data Center"},
    {"ticker": "WELL", "company": "Welltower", "sector": "Healthcare"},
    {"ticker": "SPG", "company": "Simon Property Group", "sector": "Retail"},
    {"ticker": "O", "company": "Realty Income", "sector": "Net Lease"},
    {"ticker": "AVB", "company": "AvalonBay Communities", "sector": "Residential"},
    {"ticker": "EQR", "company": "Equity Residential", "sector": "Residential"},
    {"ticker": "DLR", "company": "Digital Realty", "sector": "Data Center"},
    {"ticker": "PSA", "company": "Public Storage", "sector": "Self Storage"},
    {"ticker": "VTR", "company": "Ventas", "sector": "Healthcare"},
    {"ticker": "BXP", "company": "BXP", "sector": "Office"},
    {"ticker": "KIM", "company": "Kimco Realty", "sector": "Retail"},
    {"ticker": "REG", "company": "Regency Centers", "sector": "Retail"},
    {"ticker": "UDR", "company": "UDR", "sector": "Residential"},
    {"ticker": "ESS", "company": "Essex Property Trust", "sector": "Residential"},
    {"ticker": "ARE", "company": "Alexandria Real Estate Equities", "sector": "Life Science Office"},
    {"ticker": "EXR", "company": "Extra Space Storage", "sector": "Self Storage"},
    {"ticker": "MAA", "company": "Mid-America Apartment Communities", "sector": "Residential"},
    {"ticker": "CPT", "company": "Camden Property Trust", "sector": "Residential"},
    {"ticker": "CCI", "company": "Crown Castle", "sector": "Infrastructure"},
    {"ticker": "VICI", "company": "VICI Properties", "sector": "Gaming Net Lease"},
    {"ticker": "INVH", "company": "Invitation Homes", "sector": "Single-Family Rental"},
    {"ticker": "CUBE", "company": "CubeSmart", "sector": "Self Storage"},
    {"ticker": "EPR", "company": "EPR Properties", "sector": "Experiential"},
]

universe_path = ROOT / "config" / "reit_universe.csv"
if not universe_path.exists():
    pd.DataFrame(reit_universe).to_csv(universe_path, index=False)
    print(f"Created {universe_path}")
else:
    print(f"Already exists: {universe_path}")

config = {
    "project": {
        "start_date": "2015-01-01",
        "end_date": "2025-12-31",
        "frequency": "monthly",
    },
    "sec": {
        "user_agent": "Your Name your.email@example.com",
        "request_sleep_seconds": 0.25,
        "forms": ["10-K", "10-Q", "8-K"],
    },
    "prices": {
        "benchmark_tickers": ["VNQ", "XLRE", "SPY"],
        "interval": "1d",
    },
    "macro": {
        "fred_series": {
            "FEDFUNDS": "Effective Federal Funds Rate",
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DGS2": "2-Year Treasury Constant Maturity Rate",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
            "UNRATE": "Unemployment Rate",
        }
    },
}

config_path = ROOT / "config" / "config.yaml"
if not config_path.exists():
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Created {config_path}")
else:
    print(f"Already exists: {config_path}")

env_example = ROOT / ".env.example"
if not env_example.exists():
    env_example.write_text("FRED_API_KEY=your_fred_api_key_here\n", encoding="utf-8")
    print(f"Created {env_example}")

env_path = ROOT / ".env"
if not env_path.exists():
    env_path.write_text("FRED_API_KEY=your_fred_api_key_here\n", encoding="utf-8")
    print(f"Created {env_path}. Please replace with your actual FRED API key.")

print("\nNext steps:")
print("1. Edit config/config.yaml and replace SEC user_agent with your name/email.")
print("2. Edit .env and replace FRED_API_KEY with your actual FRED API key.")
