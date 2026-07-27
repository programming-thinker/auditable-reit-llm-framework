# Raw Data

This page documents every raw input behind the study, file by file. The raw layer
ships as its own release asset, **`data_raw.tar.gz`** (2.8 MB compressed, 13.2 MB
extracted), on release `v1.0-submission`; extracting it into the repository root
recreates `data/raw/` exactly as used. Everything downstream (`data/interim/`,
`data/processed/`, `outputs/`) is derived from these files by the scripts listed
below and ships in the companion archives.

```bash
gh release download v1.0-submission -p 'data_raw.tar.gz'
tar xzf data_raw.tar.gz            # -> data/raw/ (30 files, listed below)
```

## Sources at a glance

| Source | What we take | Where it lands | Fetched by |
|---|---|---|---|
| Yahoo Finance | Dividend-adjusted daily prices and volumes, monthly dividend yields, 25 REITs, 2015–2025 | `data/raw/prices/` | `src/01_download_prices.py`, `src/06g_download_dividend_yields.py` |
| FRED (St. Louis Fed) | Federal funds rate, 10Y/2Y Treasury yields, term spread inputs, CPI, unemployment + supplementary series | `data/raw/macro/` | `src/02_download_macro.py`, `src/06e_download_supplementary_macro.py` |
| SEC EDGAR (submissions API) | Per-company filing indexes (accession numbers, forms, `filed` dates) for the 25 CIKs | `data/raw/sec_submissions/` | `src/03_download_sec_metadata.py` |
| SEC EDGAR (documents) | 10-K / 10-Q / 8-K raw HTML and cleaned text | `filings/` (separate asset `filings.tar.gz`) | `src/04_download_sec_filings.py`, `src/05_extract_item_text.py` |
| SEC EDGAR (XBRL companyfacts API) | Point-in-time accounting facts behind the nine reconstructed fundamentals | fetched live from `data.sec.gov/api/xbrl/companyfacts/` (free, no key) | `analysis/build_fundamentals.py` |

## `data/raw/` inventory — every file (30 files, 13.2 MB)

### Prices and dividends (`data/raw/prices/`, 2 files)

| File | Size | Rows | Coverage | Contents |
|---|---|---:|---|---|
| `prices/daily_prices.csv` | 8.5 MB | 75,948 | 2015-01-02 → 2025-12-30 | Daily open/high/low/close, **dividend-adjusted close**, and volume, one row per ticker-day. 28 tickers: the 25 REITs plus three benchmark ETFs (SPY, VNQ, XLRE) used for market-model residuals and market-adjusted labels. Basis of every return, volatility, drawdown, and illiquidity feature (the "Prices" source in Table A1 of `SUPPLEMENT.md`). |
| `prices/monthly_dividend_yields.csv` | 293 KB | 3,626 | 2015-01 → 2025-12 | Trailing-12-month dividends, implied dividend amounts, and dividend yields by ticker-month, same 28 tickers. |

### Macro series (`data/raw/macro/`, 2 files)

| File | Size | Rows | Coverage | Contents |
|---|---|---:|---|---|
| `macro/fred_macro.csv` | 70 KB | 2,907 | 2015-01-01 → 2025-12-31 | The five core FRED series behind the six lagged macro signals: `FEDFUNDS`, `DGS10`, `DGS2`, `CPIAUCSL`, `UNRATE` (the term spread is computed as DGS10 − DGS2). Daily date grid; monthly series are populated on their observation dates. |
| `macro/fred_supplementary_macro.csv` | 9 KB | 144 | 2014-01 → 2025-12 | Supplementary monthly FRED series for the enriched-panel robustness checks: `MORTGAGE30US`, `BAA10Y`, `BAA`, `CSUSHPISA`, `HOUST`, `PERMIT`, `INDPRO`, `DSPIC96`. |

### EDGAR submission indexes (`data/raw/sec_submissions/`, 25 files)

One JSON per REIT, straight from the SEC submissions API. The `filed` dates in these
files drive the point-in-time discipline: a value or document enters decision month
*t* only if filed on or before *t*.

| File | Ticker | CIK | Size |
|---|---|---|---|
| `AMT_0001053507_submissions.json` | AMT | 0001053507 | 175 KB |
| `ARE_0001035443_submissions.json` | ARE | 0001035443 | 174 KB |
| `AVB_0000915912_submissions.json` | AVB | 0000915912 | 170 KB |
| `BXP_0001037540_submissions.json` | BXP | 0001037540 | 175 KB |
| `CCI_0001051470_submissions.json` | CCI | 0001051470 | 180 KB |
| `CPT_0000906345_submissions.json` | CPT | 0000906345 | 172 KB |
| `CUBE_0001298675_submissions.json` | CUBE | 0001298675 | 167 KB |
| `DLR_0001297996_submissions.json` | DLR | 0001297996 | 173 KB |
| `EPR_0001045450_submissions.json` | EPR | 0001045450 | 176 KB |
| `EQIX_0001101239_submissions.json` | EQIX | 0001101239 | 170 KB |
| `EQR_0000906107_submissions.json` | EQR | 0000906107 | 170 KB |
| `ESS_0000920522_submissions.json` | ESS | 0000920522 | 179 KB |
| `EXR_0001289490_submissions.json` | EXR | 0001289490 | 169 KB |
| `INVH_0001687229_submissions.json` | INVH | 0001687229 | 120 KB |
| `KIM_0000879101_submissions.json` | KIM | 0000879101 | 174 KB |
| `MAA_0000912595_submissions.json` | MAA | 0000912595 | 172 KB |
| `O_0000726728_submissions.json` | O | 0000726728 | 175 KB |
| `PLD_0001045609_submissions.json` | PLD | 0001045609 | 171 KB |
| `PSA_0001393311_submissions.json` | PSA | 0001393311 | 170 KB |
| `REG_0000910606_submissions.json` | REG | 0000910606 | 166 KB |
| `SPG_0001063761_submissions.json` | SPG | 0001063761 | 170 KB |
| `UDR_0000074208_submissions.json` | UDR | 0000074208 | 171 KB |
| `VICI_0001705696_submissions.json` | VICI | 0001705696 | 136 KB |
| `VTR_0000740260_submissions.json` | VTR | 0000740260 | 175 KB |
| `WELL_0000766704_submissions.json` | WELL | 0000766704 | 173 KB |

### Raw inputs that live outside `data/raw/`

- **SEC filing documents** (10-K/10-Q/8-K raw HTML + cleaned text) — `filings/`, own
  release asset `filings.tar.gz` (291 MB compressed, 4.6 GB extracted).
- **XBRL company facts** — not stored as files; fetched live from the free
  `data.sec.gov/api/xbrl/companyfacts/` API by `analysis/build_fundamentals.py`.
- **Universe definition** (tickers, CIKs, sectors) — `config/reit_universe.csv`,
  committed in the repository itself.

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
