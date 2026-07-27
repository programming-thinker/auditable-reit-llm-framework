from pathlib import Path
import time

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
METADATA_PATH = ROOT / "data" / "interim" / "filing_metadata.csv"
OUT_DIR = ROOT / "filings" / "raw_html"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

USER_AGENT = cfg["sec"]["user_agent"]
SLEEP = float(cfg["sec"].get("request_sleep_seconds", 0.25))
headers = {"User-Agent": USER_AGENT}

if not METADATA_PATH.exists():
    raise FileNotFoundError("Missing data/interim/filing_metadata.csv. Run src/03_download_sec_metadata.py first.")

meta = pd.read_csv(METADATA_PATH)

if meta.empty:
    raise RuntimeError("filing_metadata.csv is empty. Check src/03_download_sec_metadata.py output.")

failed = []

for i, row in meta.iterrows():
    ticker = row["ticker"]
    form = row["form"]
    filing_date = row["filing_date"]
    accession = row["accession_number"]
    url = row["filing_url"]

    safe_form = str(form).replace("/", "-")
    safe_accession = str(accession).replace("-", "")
    out_path = OUT_DIR / f"{ticker}_{safe_form}_{filing_date}_{safe_accession}.html"

    if out_path.exists() and out_path.stat().st_size > 1000:
        continue

    print(f"[{i + 1}/{len(meta)}] Downloading {ticker} {form} {filing_date}")

    try:
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"WARNING failed {url}: {r.status_code}")
            failed.append(url)
            continue

        out_path.write_text(r.text, encoding="utf-8", errors="ignore")

    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
        failed.append(url)

    time.sleep(SLEEP)

print("\nDone")
print("HTML files:", len(list(OUT_DIR.glob("*.html"))))

if failed:
    failed_path = ROOT / "logs" / "failed_sec_html_downloads.txt"
    failed_path.write_text("\n".join(failed), encoding="utf-8")
    print(f"Failed downloads saved to {failed_path}")
