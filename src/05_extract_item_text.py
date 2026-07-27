from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT / "filings" / "raw_html"
OUT_TEXT_DIR = ROOT / "filings" / "clean_text"
OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

rows = []


def clean_html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rough_extract_item_1a(text):
    lower = text.lower()

    start_patterns = [
        "item 1a. risk factors",
        "item 1a risk factors",
        "item 1a—risk factors",
        "item 1a - risk factors",
    ]

    end_patterns = [
        "item 1b. unresolved staff comments",
        "item 1b unresolved staff comments",
        "item 2. properties",
        "item 2 properties",
    ]

    start_positions = []
    for p in start_patterns:
        pos = lower.find(p)
        if pos != -1:
            start_positions.append(pos)

    if not start_positions:
        return ""

    start = min(start_positions)

    end_positions = []
    for p in end_patterns:
        pos = lower.find(p, start + 20)
        if pos != -1:
            end_positions.append(pos)

    end = min(end_positions) if end_positions else min(len(text), start + 100000)
    return text[start:end].strip()


for path in HTML_DIR.glob("*.html"):
    print(f"Extracting {path.name}")

    html = path.read_text(encoding="utf-8", errors="ignore")
    full_text = clean_html_to_text(html)
    item_1a = rough_extract_item_1a(full_text)

    out_txt = OUT_TEXT_DIR / path.with_suffix(".txt").name
    out_txt.write_text(full_text, encoding="utf-8")

    parts = path.stem.split("_")
    ticker = parts[0]
    form = parts[1]
    filing_date = parts[2]

    rows.append(
        {
            "ticker": ticker,
            "form": form,
            "filing_date": filing_date,
            "file_name": path.name,
            "text_path": str(out_txt.relative_to(ROOT)),
            "full_text_chars": len(full_text),
            "item_1a_text": item_1a,
            "item_1a_chars": len(item_1a),
        }
    )

out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "interim" / "extracted_filing_text.csv", index=False)

print(out.shape)
print(out[["ticker", "form", "filing_date", "item_1a_chars"]].head())
