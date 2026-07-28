# Tutorial — Reproducing This Study from a Fresh Clone

This is a zero-assumption, step-by-step walkthrough. Every command below was executed
verbatim, in this order, in a fresh clone on a clean Python virtual environment before
this file was committed; the "expected output" blocks are pasted from that run. Total
time: about 15 minutes, most of it the data download. **No API key is needed for
anything in this tutorial.**

For reference material rather than a walkthrough, see
[`REPLICATION_GUIDE.md`](REPLICATION_GUIDE.md) (exhibit-by-exhibit map),
[`DATA.md`](DATA.md) (raw data, file by file), [`SUPPLEMENT.md`](SUPPLEMENT.md)
(variable definitions, prompts, schema), and
[`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md) (the number ledger).

## 0. What you need

| Requirement | Notes |
|---|---|
| git | any recent version |
| Python **3.9+** | results verified on 3.9 with the pinned environment, and on 3.9 with current package versions; see [Step 3](#3-install-the-python-environment) |
| ~2 GB free disk | clone + archives + extracted data |
| macOS / Linux / WSL | on Windows, use WSL so that `make` and `tar` are available (or run the underlying commands listed in the [appendix](#appendix-the-commands-behind-each-make-target)) |
| GitHub CLI (`gh`) | *optional* — only a convenience for downloading the data archives; a browser works too |

Check your Python version first:

```bash
python3 --version
```

## 1. Clone the repository

```bash
git clone https://github.com/programming-thinker/auditable-reit-llm-framework.git
cd auditable-reit-llm-framework
```

## 2. Download and extract the data archives

The data ships as release assets (they are too large for git). Either use the GitHub
CLI:

```bash
gh release download v1.0-submission -p 'data.tar.gz' -p 'outputs.tar.gz' -p 'audit_log_local.tar.gz'
```

…or open the repository's **Releases** page in a browser, click release
`v1.0-submission`, download `data.tar.gz`, `outputs.tar.gz`, and
`audit_log_local.tar.gz`, and put them in the repository root. Then extract, from the
repository root:

```bash
for f in data.tar.gz outputs.tar.gz audit_log_local.tar.gz; do tar xzf "$f"; done
```

This creates `data/`, fills `outputs/`, and adds the decision log
(`audit_log/decisions.jsonl`, `audit_log/predictions.csv`) next to the committed audit
files. The other two assets are optional here: `data_raw.tar.gz` is the raw layer
alone (every file listed in `DATA.md`), and `filings.tar.gz` (4.6 GB extracted) is
needed only to rebuild disclosure inputs from scratch.

## 3. Install the Python environment

Use a virtual environment so nothing touches your system Python. Two options:

**Option A — exact verified versions (recommended; installs on Python 3.8–3.11):**
installs the precise package set under which every golden snapshot in this repository
was produced and verified (verification ran on Python 3.8/3.9).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

**Option B — current versions (any recent Python):** installs the same packages
without exact pins. Also verified: at the time this tutorial was written, Option B
resolved to scikit-learn 1.6.1 / pandas 2.3.3 / numpy 2.0.2 and Step 4 still passed
to 6 decimal places. If a future version ever drifts, Step 4 will say so explicitly
rather than silently.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Verify the headline results (one command)

```bash
make reproduce_v6
```

This re-runs the structured-ML baseline end to end — 4-fold purged cross-validation,
hyperparameter search, test-window evaluation — and diffs the regenerated key tables
against the golden snapshots shipped in `outputs/`. Expected output (about half a
minute on a typical laptop):

```text
=== Diffing regenerated tables against golden snapshots (tol=1e-6) ===
OK: quant_only_confusion_matrix_test.csv
OK: quant_only_selected_hyperparameters_cv.csv
OK: quant_only_test_metrics.csv
OK: quant_only_reduce_probability_diagnostics_test.csv

reproduce_v6 PASSED: all key tables match V6 reference to 6 dp.
```

That `PASSED` line is the core claim of the package: the baseline whose 0.000 reduce
recall the thesis reports regenerates bit-for-bit from the shipped data.

## 5. Run the unit tests

```bash
make test
```

Expected: **51 passed, 10 skipped** (a few seconds; LLM tests run against recorded
responses, no network or key involved). The 10 skipped tests are the Item-1A
extractor tests, which need the SEC filing text: they skip cleanly unless you have
extracted the optional `filings.tar.gz`, after which the full count is **61 passed**.

## 6. Verify the prompt lock

```bash
make prompt_sha
```

This recomputes each agent prompt's hash exactly as the framework does at decision
time and compares it with the lock in `config/config.yaml`. Expected output:

```text
Prompt lock timestamp: 2026-06-26T06:00:00Z
  disclosure_v1: computed 8207c14f54ac29f0  locked 8207c14f54ac29f0  OK
  macro_v1: computed eff0e398f65f1dba  locked eff0e398f65f1dba  OK
  price_v1: computed f3744184febf00b7  locked f3744184febf00b7  OK
  fundamentals_v1: computed 1ea5e30b29a865c6  locked 1ea5e30b29a865c6  OK
  aggregator_v2: computed d440daa99c6524da  locked d440daa99c6524da  OK
prompt_sha PASSED: all prompt hashes match the config lock.
```

These are the same five hashes stamped into every decision record in the audit log.

## 7. Replay the 575 LLM decisions (no API key)

Every test-window decision is one JSON line in `audit_log/decisions.jsonl`. Rebuild
the predictions table from the log and compare it with the shipped one (`true_label`
is joined from the labelled panel later in the pipeline, so the prediction columns are
what must match):

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

Expected output:

```text
rows: 575 — identical prediction columns: True
```

## 8. Regenerate any individual exhibit

Each thesis table and figure comes from one script in `analysis/`, reading the CSVs
you extracted in Step 2. The full exhibit → script → data map is
[`REPLICATION_GUIDE.md`](REPLICATION_GUIDE.md) §4. For example:

```bash
python3 analysis/build_llm_comparison.py   # Table 4: LLM vs baselines + bootstrap
python3 analysis/make_figures.py           # Figures 2-8 into outputs/figures_v2/
```

Cross-check any number you see against [`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md),
which maps every reported result to its source CSV.

## Going deeper (optional)

- **Rebuild all panels from the raw layer.** Download `data_raw.tar.gz` and follow
  "Rebuilding everything from raw" in [`DATA.md`](DATA.md); it ends in the same
  `make reproduce_v6` check.
- **Rebuild disclosure inputs from SEC filings.** Download `filings.tar.gz`
  (4.6 GB extracted); `src/05_extract_item_text.py` re-extracts the Item 1A /
  10-Q / 8-K text the Disclosure agent reads.
- **Fresh LLM calls.** The only step that needs an API key. Prerequisites: extract
  `filings.tar.gz` (the Disclosure agent reads the filing text from `filings/`), copy
  `.env.example` to `.env`, and set **both** `DEEPSEEK_API_KEY` and
  `DEEPSEEK_BASE_URL=https://api.deepseek.com`. Then run `make llm_dev_run`
  (1 REIT × 2 months smoke test; it targets the pinned thesis model config
  `deepseek_v4_flash`). Decoding is deterministic (temperature 0, pinned model
  version), and the full test window is intentionally lock-guarded against
  accidental reruns.

## What success looks like

| Level | Check | You have verified |
|---|---|---|
| 1 | `make reproduce_v6` → `PASSED` | the tuned baseline and its 0.000 reduce recall regenerate exactly from the shipped data |
| 2 | `make test` → `51 passed, 10 skipped` | the pipeline's unit behaviour, including recorded-response LLM tests |
| 3 | `make prompt_sha` → all `OK` | the prompts on disk are byte-identical to the locked versions used in the run |
| 4 | Step 7 → `identical prediction columns: True` | the framework's 575 decisions replay exactly from the append-only audit log |
| 5 | Step 8 scripts rerun cleanly | every exhibit re-derives from the shipped CSVs |

## Troubleshooting

- **`pip install -r requirements-lock.txt` fails to resolve** — your Python is newer
  than 3.11 (some pinned versions predate it). Use Option B (`requirements.txt`,
  verified to pass Step 4 too), or create the venv with an older Python
  (`python3.11 -m venv .venv`).
- **`make reproduce_v6` reports a numeric diff** — almost always package-version
  drift under Option B. Recreate the venv with Option A on Python 3.9–3.11; the
  golden snapshots are guaranteed against the lock file.
- **`ModuleNotFoundError`** — the venv is not active (`source .venv/bin/activate`),
  or Step 3 was skipped.
- **`tar: ... Cannot open`** — the archives are not in the repository root; move them
  next to `Makefile` and re-extract.
- **No `make` (plain Windows)** — use WSL, or run the underlying commands from the
  appendix below in PowerShell (replacing `python3` with `python`).
- **`gh: command not found`** — skip the CLI and download the three archives from the
  Releases page in a browser.

## Appendix: the commands behind the make targets used in this tutorial

| Target | Underlying command |
|---|---|
| `make reproduce_v6` | `python3 tests/test_v6_reproduction.py` |
| `make test` | `pytest tests/ -v` |
| `make prompt_sha` | `python3 tests/check_prompt_sha.py` |
| `make llm_dev_run` | `python3 -m llm.run --mode dev --model-config deepseek_v4_flash` |
