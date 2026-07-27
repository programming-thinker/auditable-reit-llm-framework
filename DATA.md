# Raw Data

This page documents every raw input behind the study. The raw layer ships as its own
release asset, **`data_raw.tar.gz`** (13 MB), on release `v1.0-submission`; extracting
it into the repository root recreates `data/raw/` exactly as used. Everything
downstream (`data/interim/`, `data/processed/`, `outputs/`) is derived from these
files by the scripts listed below and ships in the companion archives.

## Sources at a glance

| Source | What we take | Where it lands | Fetched by |
|---|---|---|---|
| Yahoo Finance | Dividend-adjusted daily prices and volumes, monthly dividend yields, 25 REITs, 2015–2025 | `data/raw/prices/` | `src/01_download_prices.py`, `src/06g_download_dividend_yields.py` |
| FRED (St. Louis Fed) | Federal funds rate, 10Y/2Y Treasury yields, term spread inputs, CPI, unemployment + supplementary series | `data/raw/macro/` | `src/02_download_macro.py`, `src/06e_download_supplementary_macro.py` |
| SEC EDGAR (submissions API) | Per-company filing indexes (accession numbers, forms, `filed` dates) for the 25 CIKs | `data/raw/sec_submissions/` | `src/03_download_sec_metadata.py` |
| SEC EDGAR (documents) | 10-K / 10-Q / 8-K raw HTML and cleaned text | `filings/` (separate asset `filings.tar.gz`) | `src/04_download_sec_filings.py`, `src/05_extract_item_text.py` |
| SEC EDGAR (XBRL companyfacts API) | Point-in-time accounting facts behind the nine reconstructed fundamentals | fetched live from `data.sec.gov/api/xbrl/companyfacts/` (free, no key) | `analysis/build_fundamentals.py` |

## `data/raw/` inventory (30 files, 13 MB)

- **`prices/daily_prices.csv`** — dividend-adjusted daily close and volume, one row per
  ticker-day, 25 tickers, 2015-01 to 2025-11. Basis of every return, volatility,
  drawdown, and illiquidity feature (the "Prices" source in Table A1).
- **`prices/monthly_dividend_yields.csv`** — trailing dividend yields by ticker-month.
- **`macro/fred_macro.csv`** — the six core FRED series (FEDFUNDS, DGS10, DGS2,
  term-spread inputs, CPI, UNRATE) at monthly frequency.
- **`macro/fred_supplementary_macro.csv`** — supplementary FRED series used in the
  enriched-panel robustness checks.
- **`sec_submissions/<TICKER>_<CIK>_submissions.json`** (25 files) — EDGAR submission
  indexes; the `filed` dates in these files drive the point-in-time discipline
  (a value or document enters month *t* only if filed on or before *t*).

Universe definition (tickers, CIKs, sectors): `config/reit_universe.csv` in the
repository itself.

## Rebuilding everything from raw

```bash
tar xzf data_raw.tar.gz            # -> data/raw/
python3 src/06a_build_monthly_price_signals.py
python3 src/06b_build_monthly_macro_signals.py
python3 src/06c_build_reit_monthly_panel.py
python3 src/06d_create_backtest_ready_panel.py
python3 src/06h_build_enriched_panel.py
python3 src/06i_create_enriched_splits.py
make reproduce_v6                  # must match golden snapshots to 6 dp
```

The download scripts (`src/01–04`) can refresh the raw layer from the live sources,
but note that Yahoo prices are subject to vendor revisions; the archived
`data_raw.tar.gz` is the exact snapshot behind every number in the thesis and in
`CANONICAL_RESULTS.md`.

## Licence and provenance notes

- Prices: retrieved via the public Yahoo Finance endpoints for academic research use.
- FRED series: public domain (St. Louis Fed).
- SEC EDGAR submissions, filings, and XBRL facts: U.S. government public records.
- No proprietary or paywalled data enters the pipeline at any point.
