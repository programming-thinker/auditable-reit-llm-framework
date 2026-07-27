from pathlib import Path
import json
import time

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"
CONFIG_PATH = ROOT / "config" / "config.yaml"
OUT_DIR = ROOT / "data" / "raw" / "sec_submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

USER_AGENT = cfg["sec"]["user_agent"]
if "your.email" in USER_AGENT or "Your Name" in USER_AGENT:
    raise ValueError("Please edit config/config.yaml and set a real SEC user_agent with your name/email.")

SLEEP = float(cfg["sec"].get("request_sleep_seconds", 0.25))
FORMS = set(cfg["sec"].get("forms", ["10-K", "10-Q", "8-K"]))
START_DATE = pd.to_datetime(cfg["project"]["start_date"])
END_DATE = pd.to_datetime(cfg["project"]["end_date"])

headers = {"User-Agent": USER_AGENT}

universe = pd.read_csv(UNIVERSE_PATH)
universe["ticker"] = universe["ticker"].astype(str).str.upper()

print("Downloading SEC company tickers lookup...")
lookup_url = "https://www.sec.gov/files/company_tickers.json"
r = requests.get(lookup_url, headers=headers, timeout=30)
if r.status_code != 200:
    raise RuntimeError(f"Failed to download SEC company tickers lookup: {r.status_code}")

lookup_raw = r.json()
lookup = pd.DataFrame.from_dict(lookup_raw, orient="index")
lookup["ticker"] = lookup["ticker"].astype(str).str.upper()
lookup["cik_str"] = lookup["cik_str"].astype(int).astype(str).str.zfill(10)

universe = universe.merge(
    lookup[["ticker", "cik_str", "title"]],
    on="ticker",
    how="left",
)

missing = universe[universe["cik_str"].isna()]["ticker"].tolist()
if missing:
    print("WARNING: SEC CIK not found for:", missing)

universe.to_csv(ROOT / "config" / "reit_universe_with_cik.csv", index=False)

all_rows = []

for _, row in universe.dropna(subset=["cik_str"]).iterrows():
    ticker = row["ticker"]
    cik = row["cik_str"]
    company = row.get("company", row.get("title", ""))
    sector = row.get("sector", "")

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    print(f"Downloading submissions for {ticker} CIK {cik}")

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"WARNING: failed {ticker}: {resp.status_code}")
        continue

    data = resp.json()

    with open(OUT_DIR / f"{ticker}_{cik}_submissions.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for form, accession, filing_date, report_date, primary_doc in zip(
        forms, accession_numbers, filing_dates, report_dates, primary_docs
    ):
        if form not in FORMS:
            continue

        filing_dt = pd.to_datetime(filing_date, errors="coerce")
        if pd.isna(filing_dt):
            continue
        if filing_dt < START_DATE or filing_dt > END_DATE:
            continue

        accession_nodash = accession.replace("-", "")

        all_rows.append(
            {
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "cik": cik,
                "form": form,
                "accession_number": accession,
                "accession_nodash": accession_nodash,
                "filing_date": filing_date,
                "report_date": report_date,
                "primary_document": primary_doc,
                "filing_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_doc}",
            }
        )

    time.sleep(SLEEP)

metadata = pd.DataFrame(all_rows)
out_path = ROOT / "data" / "interim" / "filing_metadata.csv"
metadata.to_csv(out_path, index=False)

print(f"\nSaved {out_path}")
print("Shape:", metadata.shape)

if not metadata.empty:
    print(metadata[["ticker", "form", "filing_date", "filing_url"]].head())
    print(metadata.groupby("form").size())
else:
    print("WARNING: no filing metadata collected.")
