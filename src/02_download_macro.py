from pathlib import Path
import os

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
OUT_DIR = ROOT / "data" / "raw" / "macro"
OUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

api_key = os.getenv("FRED_API_KEY")
if not api_key or api_key == "your_fred_api_key_here":
    raise ValueError(
        "Missing FRED_API_KEY. Edit .env and set FRED_API_KEY=your_actual_key."
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

series = cfg["macro"]["fred_series"]
start = cfg["project"]["start_date"]
end = cfg["project"]["end_date"]

frames = []

for series_id, description in series.items():
    print(f"Downloading FRED series {series_id}: {description}")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(
            f"FRED network request failed for {series_id}. "
            "Check connectivity and API key setup."
        ) from e
    if r.status_code != 200:
        raise RuntimeError(
            f"FRED request failed for {series_id}: {r.status_code} {r.text[:500]}"
        )

    data = r.json()
    observations = data.get("observations", [])

    rows = []
    for obs in observations:
        value = obs.get("value")
        if value == ".":
            value = None
        rows.append({"date": obs.get("date"), series_id: value})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    frames.append(df)

macro = frames[0]
for df in frames[1:]:
    macro = macro.merge(df, on="date", how="outer")

macro = macro.sort_values("date")

out_path = OUT_DIR / "fred_macro.csv"
macro.to_csv(out_path, index=False)

print(f"\nSaved {out_path}")
print("Shape:", macro.shape)
print(macro.head())
