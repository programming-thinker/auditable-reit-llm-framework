# Supplementary Materials

*Supplement to* **An Auditable Multi-Agent LLM Framework for U.S. REIT Investment
Decisions: Decomposable Rationales Beyond a Tuned Structured-ML Baseline** *(Lankun
Chen, MPhil in Real Estate Finance, University of Cambridge, 2026). Sections A–D are
the material the thesis cites as "the Supplement": the universe and variable
definitions, the agent prompts with their SHA pins, the reproducibility map, and the
audit-log schema. Every number here reconciles to
[`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md). If you want to verify or rerun any of
it, the final section — [How to reproduce](#how-to-reproduce-with-this-supplement) —
maps each section to concrete files and commands in this repository.*

---

## A — Universe, labels, and variable definitions

**A.1 The 25-REIT universe.** The panel comprises the following large-cap U.S. equity
REITs (`config/reit_universe.csv`):

| Ticker | Company | Sector | Ticker | Company | Sector |
|---|---|---|---|---|---|
| PLD | Prologis | Industrial | ESS | Essex Property Trust | Residential |
| AMT | American Tower | Infrastructure | ARE | Alexandria Real Estate | Life Science Office |
| EQIX | Equinix | Data Center | EXR | Extra Space Storage | Self Storage |
| WELL | Welltower | Healthcare | MAA | Mid-America Apartment | Residential |
| SPG | Simon Property Group | Retail | CPT | Camden Property Trust | Residential |
| O | Realty Income | Net Lease | CCI | Crown Castle | Infrastructure |
| AVB | AvalonBay Communities | Residential | VICI | VICI Properties | Gaming Net Lease |
| EQR | Equity Residential | Residential | INVH | Invitation Homes | Single-Family Rental |
| DLR | Digital Realty | Data Center | CUBE | CubeSmart | Self Storage |
| PSA | Public Storage | Self Storage | EPR | EPR Properties | Experiential |
| VTR | Ventas | Healthcare | | | |
| BXP | BXP | Office | | | |
| KIM | Kimco Realty | Retail | | | |
| REG | Regency Centers | Retail | | | |
| UDR | UDR | Residential | | | |

**A.2 Label-threshold sensitivity.** The ±2% band is robust: varying the threshold shifts
the class proportions monotonically but does not change the qualitative findings (full
panel, source `outputs/tables/label_threshold_sensitivity_label_distribution.csv`). *Note: this
sensitivity panel is computed on the 3,214-firm-month pre-filter (V5-era) panel used by
the original sensitivity script, so its ±2% row differs from the 3,189-row analysis
panel of Table 1 by ≈0.2 percentage points.*

| Threshold | Increase | Hold | Reduce |
|---|---:|---:|---:|
| ±1% | 48.4% | 12.5% | 39.1% |
| **±2% (used)** | **41.9%** | **25.3%** | **32.8%** |
| ±3% | 35.6% | 36.8% | 27.6% |
| ±5% | 24.1% | 58.1% | 17.8% |

**A.3 Feature inventory.** A full inventory of all 90 enriched panel columns, with each
column's mean within-month cross-sectional standard deviation, is in
`outputs/llm_deepseek_test/feature_inventory_90.csv` (from the `outputs.tar.gz` release
asset); it confirms that the macro block has zero cross-sectional dispersion while the
firm-level features vary across firms.

**A.4 Variable definitions (Table A1).** Definitions, construction, timing, and source
for the 13 classifier inputs and 9 reconstructed fundamentals.

*Panel A. Classifier inputs (13 numeric features + sector)*

| Variable | Definition / construction | Timing | Source |
|---|---|---|---|
| `ret_1m` | Trailing 1-month total return (adjusted close) | Computed through month-end *t* | Prices |
| `ret_3m` | Trailing 3-month total return | Computed through month-end *t* | Prices |
| `ret_6m` | Trailing 6-month total return | Computed through month-end *t* | Prices |
| `ret_12m` | Trailing 12-month total return | Computed through month-end *t* | Prices |
| `vol_annualized` | Annualized standard deviation of daily returns, 60-day rolling window | Computed through month-end *t* | Prices |
| `drawdown` | Decline of adjusted close from its running maximum | Computed through month-end *t* | Prices |
| `dividend_yield_lag1` | Trailing dividend yield (dividends over price) | Lagged one month (*t*−1) | Prices |
| `FEDFUNDS_lag1` | Effective federal funds rate | Lagged one month (*t*−1) | FRED |
| `DGS10_lag1` | 10-year Treasury constant-maturity yield | Lagged one month (*t*−1) | FRED |
| `DGS2_lag1` | 2-year Treasury constant-maturity yield | Lagged one month (*t*−1) | FRED |
| `term_spread_10y_2y_lag1` | 10-year minus 2-year Treasury yield | Lagged one month (*t*−1) | FRED |
| `cpi_yoy_lag1` | CPI year-over-year change | Lagged one month (*t*−1) | FRED |
| `UNRATE_lag1` | Civilian unemployment rate | Lagged one month (*t*−1) | FRED |
| `sector` | REIT property-sector classification (one-hot dummies) | Static | Universe file |

*Panel B. Reconstructed REIT fundamentals (9)*

| Variable | Definition / construction | Timing | Source |
|---|---|---|---|
| `leverage` | Long-term debt over total assets (`LongTermDebt`/`Assets`) | Most recent annual value with SEC `filed` date ≤ *t* | XBRL |
| `debt_to_equity` | Total liabilities over stockholders' equity | Most recent annual value with SEC `filed` date ≤ *t* | XBRL |
| `interest_cover` | (Net income + interest expense + D&A) over interest expense | Most recent annual value with SEC `filed` date ≤ *t* | XBRL |
| `ffo_yield_proxy` | (Net income + D&A) over market capitalization; NAREIT FFO logic (Vincent, 1999) | Most recent annual value with SEC `filed` date ≤ *t*; price at *t* | XBRL, prices |
| `book_to_market` | Stockholders' equity over market capitalization (Fama–French, 1992) | Most recent annual value with SEC `filed` date ≤ *t*; price at *t* | XBRL, prices |
| `ln_mktcap` | Log market capitalization (shares outstanding × price) | Shares filed ≤ *t*; price at *t* | XBRL, prices |
| `amihud_illiq` | Mean of \|daily return\| over dollar volume, trailing 21 trading days (Amihud, 2002) | Computed through month-end *t* | Prices |
| `idio_vol` | Annualized standard deviation of market-model residuals, trailing 63 trading days (Ang et al., 2006) | Computed through month-end *t* | Prices |
| `navprem_book_adj` | Market capitalization over depreciation-adjusted book equity (equity + accumulated real-estate depreciation); NAV-premium proxy (Clayton & MacKinnon, 2003) | Most recent annual value with SEC `filed` date ≤ *t*; price at *t* | XBRL, prices |

*Notes: "Lagged one month" means the value known at the end of month t−1 enters the
decision at t; accounting items follow the strict point-in-time rule that a value is
attached to decision month t only if its SEC `filed` date is on or before t (missing
values are left missing, never imputed). Sources: **Prices** = the dividend-adjusted
daily price/volume series (Yahoo Finance, `data/raw/prices/`, see
[`DATA.md`](DATA.md)), from which the variable is computed; **FRED** = Federal Reserve
Economic Data; **XBRL** = SEC company-facts filings; **Universe file** = the REIT
universe definition (`config/reit_universe.csv`). "XBRL, prices" marks variables
needing both an accounting item and a market price.*

## B — Agent prompts (versions, SHAs, and role summaries)

Prompts are versioned and SHA-pinned in `config/config.yaml`; the lock timestamp is
`2026-06-26T06:00:00Z`. SHAs are computed by `Orchestrator._compute_prompts_sha`. The five
prompt files live in `llm/prompts/`. Versions/SHAs: disclosure v1 `8207c14f54ac29f0`;
macro v1 `eff0e398f65f1dba`; price v1 `f3744184febf00b7`; fundamentals v1 `1ea5e30b29a865c6`;
aggregator v2 `d440daa99c6524da`. Each agent returns strict, Pydantic-validated JSON; the
binding rule across all four specialists is *"facts_cited must reference specific content
from the provided inputs; do not fabricate facts not present in the input."* A cache
audit matches all 2,875 calls of the reported run to responses served under the single
model string `deepseek-v4-flash`, so the version pin held in practice; the Disclosure
agent returned its instructed uniform no-signal default in 18 of 575 decisions (3.1%).

**B.1 Disclosure agent (v1, `8207c14f54ac29f0`).** Role: analyse 10-K Item 1A / 10-Q / 8-K
text for a single REIT and assess whether disclosure implies elevated, neutral, or reduced
risk. Output: `probabilities{increase,hold,reduce}`, `rationale`, `facts_cited`, and `sentiment`
∈ {positive, neutral, negative}. Scoring: *negative* = new/worsening risks (impairments,
covenant concerns, tenant defaults, litigation, going-concern); *positive* = improving
fundamentals or positive material events; *neutral* = routine/boilerplate. If no filing
text is available, default to neutral with (0.34, 0.33, 0.33).

**B.2 Macro agent (v1, `eff0e398f65f1dba`).** Role: assess whether the rate/macro regime is
favourable, neutral, or unfavourable for the REIT's *sector*. Inputs: the six lagged macro
signals (`FEDFUNDS_lag1`, `DGS10_lag1`, `DGS2_lag1`, `term_spread_10y_2y_lag1`,
`cpi_yoy_lag1`, `UNRATE_lag1`). Output adds `regime_label` ∈ {rising_rates, falling_rates,
flat, inverted_curve, stagflation, recovery}. Guidance enforces sector-specific
sensitivity (office/net-lease rate-sensitive; data-centre/industrial demand-driven) rather
than a blanket market call, and explicitly discourages defaulting to neutral.

**B.3 Price agent (v1, `f3744184febf00b7`).** Role: assess momentum and risk-adjusted trend
from `ret_1m/3m/6m/12m`, `vol_annualized`, `drawdown`. Output adds `momentum_state`
∈ {strong_up, mild_up, flat, mild_down, strong_down}. Guidance cross-references momentum
with risk (strong returns at low vol are more bullish than at high vol) and treats
`not_available` values as missing.

**B.4 Fundamentals agent (v1, `1ea5e30b29a865c6`).** Role: diagnose financial health and
valuation from the nine reconstructed fundamentals (`ffo_yield_proxy`, `leverage`,
`debt_to_equity`, `interest_cover`, `navprem_book_adj`, `book_to_market`, `ln_mktcap`,
`amihud_illiq`, `idio_vol`). Output adds `financial_health` ∈ {strong, adequate, stressed}.
The prompt explicitly frames the agent as *diagnosing* financial risk (citing
Campbell–Hilscher–Szilagyi 2008 distress logic) and instructs it *not* to forecast returns
from memory—the design choice that makes the grounding/factuality audit meaningful.

**B.5 Aggregator agent (v2, `d440daa99c6524da`).** Role: combine the four specialist outputs
into a consensus distribution and an `agreement_score` ∈ [0,1]. Guidance instructs it to
weight by evidence coherence rather than simply average, to lean to the majority while
noting dissent, and to widen toward uniform when agents broadly disagree. Section 4.3 of
the thesis shows it agrees with a deterministic mean on 77% of decisions and that neither
rule clears random-at-budget by a margin surviving sampling uncertainty, so the
aggregation choice does not drive the predictive conclusion.

*The complete, unabridged prompt text (including the strict JSON schemas) is in the
versioned files `llm/prompts/{disclosure_v1, macro_v1, price_v1, fundamentals_v1,
aggregator_v2}.md`, retained for audit-log replayability. Run `make prompt_sha` to
recompute the hashes and confirm they match the lock above.*

## C — Reproducibility and provenance

All analyses are reproducible from `analysis/` (Zone 3; no Zone-1 numbers recomputed).
Selected scripts and their outputs (output paths are relative to `outputs/` unless
stated; the `paper/…` targets belong to the thesis typesetting tree, which is not part
of this repository):

| Script | Purpose | Output |
|---|---|---|
| `make_tables_tex.py` | Generate the eight typeset booktabs tables | `paper/_tables/table{1..8}.tex` (thesis-side) |
| `make_tables.py` | Markdown table renditions (pre-restructure numbering) | `outputs/tables_v2/` |
| `make_figures.py` | Ten result figures, PNG and vector PDF (seven used in the thesis) | `outputs/figures_v2/` |
| `make_architecture_figure.py` | Architecture preview PNG (Figure 1 typesets from native TikZ) | `paper/_figures/fig_architecture.tex` (thesis-side) |
| `build_llm_comparison.py` | LLM vs trivial baselines + bootstrap | `headline_comparison.csv`, `significance_bootstrap.csv` |
| `inference_robustness.py` | Leave-one-out market R², month-block bootstrap, TOST | `inference_robustness.csv` |
| `text_baseline.py` | TF-IDF + LM-sentiment transparent text baselines | `text_baseline.csv` |
| `factuality_audit.py` | Factual accuracy of cited facts | `factuality_audit.csv` |
| `contamination_audit.py` | Look-ahead + grounding audit | `contamination_summary.csv` |
| `ablation_channels.py` | Agent-subset re-aggregation | `ablation_channels.csv` |
| `aggregation_robustness.py` | LLM-agg vs deterministic mean | `aggregation_robustness.csv` |
| `tuned_robustness.py` | Elastic-net-tuned reduce recall | `tuned_robustness.csv` |
| `extended_ols_fmb.py` | OLS + Fama–MacBeth on extended set | `extended_ols.csv`, `extended_fama_macbeth.csv` |
| `systematic_vs_idiosyncratic.py` | Variance decomposition | `systematic_vs_idiosyncratic.csv` |
| `build_fundamentals.py`, `build_nav_proxy.py` | Reconstruct fundamentals from XBRL | `firm_fundamentals_panel.csv`, `nav_proxy_panel.csv` |
| `regime_validation.py`, `proto_regime_timing.py` | Bad-month regime exploration | `regime_validation.csv`, `regime_timing.csv` |
| `proto_textsim_lazyprices.py`, `proto_8k_events.py`, `proto_change_detector.py` | Cheap signal prototypes | `proto_textsim.csv`, `proto_8k_events.csv`, `proto_change_signal.csv` |
| `case_studies.py`, `reasoner_subset.py`, `disclosure_fix_subset.py` | Case studies, V4-Pro subset, disclosure-fix subset | `case_studies.md`, `reasoner_subset.csv`, `disclosure_fix_subset.csv` |
| `fama_macbeth_nw.py` | Newey–West HAC robustness for Fama–MacBeth | `fama_macbeth_nw.csv` |
| `restatement_audit.py` | Restatement / point-in-time audit | `restatement_audit_summary.csv` |
| `rationale_quality_audit.py` | Automated judge of rationale quality + spot-check sheet | `rationale_quality_audit.csv`, `spotcheck_agreement.csv` |
| `power_mde.py` | Bootstrap power / minimum-detectable-effect simulation | `power_mde.csv` |
| `market_adjusted_labels.py` | Market-adjusted (LOO-excess) relabelling robustness | `market_adjusted_labels.csv` |
| `masked_identity_subset.py` | Masked-identity contamination probe (70-decision subset) | `masked_probe_comparison.csv` |
| `budget_sweep.py`, `calibration_llm.py` | All-budget recall sweep; Brier/Murphy calibration; tie analysis | `budget_sweep.csv`, `calibration_llm.csv` |
| `sector_decomposition.py`, `timing_and_agreement.py`, `model_version_backfill.py` | Sector allocation vs selection; timing and agreement diagnostics; served-model audit | `sector_decomposition.csv`, `timing_agreement.csv`, `model_version_backfill.csv` |

Every number in the thesis is reconciled in [`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md),
which maps each result to a specific `outputs/*.csv` or `audit_log` file; the eight
typeset tables are generated by `analysis/make_tables_tex.py` (manual deltas are
enumerated in its header NOTE). The bibliography (85 sources, all identifiers verified)
is part of the thesis, which is submitted separately.

## D — Audit architecture and decision schema

Every decision is appended as one JSON line to `audit_log/decisions.jsonl` (append-only;
overwriting is a hard tripwire). The schema:

```json
{
  "decision_date_t": "YYYY-MM-DD",
  "ticker": "AMT",
  "model_main": "deepseek-chat:<version>",
  "prompts_sha": {"disclosure": "8207c14f...", "macro": "eff0e398...",
                  "price": "f3744184...", "fundamentals": "1ea5e30b...",
                  "aggregator": "d440daa9..."},
  "inputs_hash": "sha256:...",
  "agent_outputs": {
    "disclosure": {"probabilities": {...}, "rationale": "...", "facts_cited": [...], "sentiment": "..."},
    "macro":      {"probabilities": {...}, "rationale": "...", "facts_cited": [...], "regime_label": "..."},
    "price":      {"probabilities": {...}, "rationale": "...", "facts_cited": [...], "momentum_state": "..."},
    "fundamentals":{"probabilities": {...}, "rationale": "...", "facts_cited": [...], "financial_health": "..."},
    "aggregator": {"probabilities": {...}, "rationale": "...", "agreement_score": 0.0}
  },
  "final_probabilities": {"increase": 0.0, "hold": 0.0, "reduce": 0.0},
  "tokens": {"prompt": 0, "completion": 0, "cost_usd": 0.0},
  "latency_sec": 0.0
}
```

Determinism is enforced by greedy decoding (temperature 0, pinned model version) and a
`diskcache` keyed on `sha256(prompt + model + decoding-params)`; the predictions CSV schema
(`decision_date, ticker, prob_increase, prob_hold, prob_reduce, predicted_label,
true_label, model_run_id`) matches the V5 baseline field-for-field. Together these make any
decision regenerable and the run fully replayable from logged provenance. The full field
reference is `audit_log/SCHEMA.md`.

---

## How to reproduce with this supplement

Everything documented above is verifiable from this repository plus the
`v1.0-submission` release assets. For a hand-held, zero-assumption walkthrough with
the expected output of every command, follow [`TUTORIAL.md`](TUTORIAL.md); the
section-by-section path below maps this supplement onto it. In order:

**Step 0 — Set up (once).**

```bash
git clone https://github.com/programming-thinker/auditable-reit-llm-framework.git
cd auditable-reit-llm-framework
gh release download v1.0-submission -p 'data.tar.gz' -p 'outputs.tar.gz' -p 'audit_log_local.tar.gz'
for f in *.tar.gz; do tar xzf "$f"; done
pip install -r requirements.txt
```

**The raw data ships separately.** `data_raw.tar.gz` (2.8 MB) contains `data/raw/`
alone — the Yahoo price series, the FRED macro series, and the 25 EDGAR submission
indexes — and every one of its 29 files is listed individually, with sizes, row
counts, and coverage, in [`DATA.md`](DATA.md):

```bash
gh release download v1.0-submission -p 'data_raw.tar.gz'
tar xzf data_raw.tar.gz            # -> data/raw/
```

(No GitHub CLI? The same archives are on the [Releases page](../../releases). The
SEC filing text itself is committed at `filings/clean_text/`; `filings.tar.gz` adds
only its raw-HTML source — see
[`REPLICATION_GUIDE.md`](REPLICATION_GUIDE.md) §1 for what each asset is for.)

**Step 1 — Verify Section A (universe, labels, variables).**
The universe table is `config/reit_universe.csv`. The feature inventory of A.3 is
`outputs/llm_deepseek_test/feature_inventory_90.csv` after extraction. Every variable in
Table A1 is constructed by the scripts listed in [`DATA.md`](DATA.md); to rebuild the
panels from the raw layer and confirm they match the shipped ones, follow the
"Rebuilding everything from raw" block there, which ends in the 6-decimal-place check:

```bash
make reproduce_v6
```

**Step 2 — Verify Section B (prompts).**
Read the five prompt files in `llm/prompts/`, then confirm the SHA pins:

```bash
make prompt_sha
```

This recomputes the hashes from the prompt files and checks them against the
`config/config.yaml` lock — the same values printed in Section B and stamped into every
decision record.

**Step 3 — Rerun Section C (analyses).**
`make reproduce_v6` re-derives the baseline results and diffs them against golden
snapshots; individual exhibit scripts in the table above run as
`python3 analysis/<script>.py` once the archives are extracted. The exhibit-by-exhibit
map (which script feeds which thesis table/figure, and from which CSV) is
[`REPLICATION_GUIDE.md`](REPLICATION_GUIDE.md) §4. Cross-check any number against
[`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md).

**Step 4 — Replay Section D (LLM decisions, no API key).**
All 575 test decisions are lines in `audit_log/decisions.jsonl` (from
`audit_log_local.tar.gz`), each carrying the prompt SHAs, an input hash, all five agent
outputs, and the final probabilities. Rebuild the predictions table from the log and
diff it against the shipped one (`true_label` is joined from the labelled panel later
in the pipeline, so compare the prediction columns):

```bash
python3 -c "
from pathlib import Path
import pandas as pd
from llm.postprocess import jsonl_to_predictions_csv
out = jsonl_to_predictions_csv(output_csv=Path('predictions_replay.csv'))
a = pd.read_csv(out).drop(columns=['true_label'])
b = pd.read_csv('audit_log/predictions.csv').drop(columns=['true_label'])
print('rows:', len(a), '— identical prediction columns:', a.equals(b))
"
```

Expected output: `rows: 575 — identical prediction columns: True`. From there,
`analysis/build_llm_comparison.py` produces the headline comparison. A DeepSeek API key
is needed only to regenerate decisions from scratch (`make llm_dev_run` for a
1-REIT × 2-month smoke test); replay and every analysis above run without one.
