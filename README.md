# An Auditable Multi-Agent LLM Framework for U.S. REIT Investment Decisions

Replication package for an MPhil dissertation (Real Estate Finance, University of
Cambridge, 2026). The project asks whether monthly downside risk in U.S. equity REITs
can be predicted at the firm level, and whether a five-agent LLM framework reading
SEC disclosure improves on a tuned, interpretable structured-ML baseline.

**Findings in brief.** Across every specification and evaluation criterion tested, we
find no statistically or economically meaningful predictive improvement: the tuned
logistic baseline attains 0.000 out-of-sample reduce recall, and the framework's 0.206
matches a random floor of 0.204 at the same firing rate. A descriptive variance
decomposition attributes far more in-sample variation to calendar-month effects
(R² 0.43) than to observed firm features (0.007), and the null survives market-adjusted
relabelling and a corrected disclosure feed. What survives is auditable diagnosis:
rationales are grounded in verifiable filing facts, every decision replays from an
append-only audit log, and an outcome-blind judge's entailment reading is corroborated
by two human raters, one of them independent. All headline numbers reconcile to
[`CANONICAL_RESULTS.md`](CANONICAL_RESULTS.md).

The dissertation text itself is not distributed here before examination.

## Quick start

```bash
git clone https://github.com/programming-thinker/auditable-reit-llm-framework.git
cd auditable-reit-llm-framework
gh release download v1.0-submission -p 'data.tar.gz' -p 'outputs.tar.gz' -p 'audit_log_local.tar.gz'
for f in *.tar.gz; do tar xzf "$f"; done      # extract into the repository root
pip install -r requirements.txt
make reproduce_v6                              # golden-snapshot regression, 6-dp match
make test                                      # unit tests (recorded LLM responses, no API)
```

`data_raw.tar.gz` (13 MB) is the raw layer on its own — Yahoo prices, FRED series,
EDGAR submission indexes — documented in [`DATA.md`](DATA.md). `filings.tar.gz`
(291 MB compressed, 4.6 GB extracted) is needed only to rebuild disclosure inputs
from scratch. Archives can also be downloaded from the
[Releases page](../../releases) in a browser.

## Repository map

| Path | Contents |
|---|---|
| `src/` | Structured-ML pipeline: panel construction, features, baseline estimation |
| `analysis/` | One script per thesis exhibit (tables, figures, bootstrap inference, audits) |
| `llm/` | Five-agent framework: agents, orchestrator, EDGAR client, post-processing |
| `audit_log/` | Append-only decision evidence: schema, cost ledgers, masked-identity probe, corrected-feed rerun, human-rater materials |
| `config/` | REIT universe definition and the SHA-pinned prompt lock |
| `tests/` | Unit tests + the `reproduce_v6` golden-snapshot regression |
| `DATA.md` | Raw-data page: sources, inventory, and rebuild-from-raw instructions |
| `CANONICAL_RESULTS.md` | The single ledger every reported number traces to |
| `REPLICATION_GUIDE.md` | Exhibit-by-exhibit script/data map and full instructions |

## Replaying the LLM decisions (no API key)

Each of the 575 test decisions is one line in `audit_log/decisions.jsonl` (from the
`audit_log_local.tar.gz` archive), carrying prompt SHAs, an input hash, all five agent
outputs, and final probabilities. `llm/postprocess.py` rebuilds `predictions.csv` from
the log; `make prompt_sha` verifies the prompt hashes against `config/config.yaml`
(prompt lock 2026-06-26). Fresh API calls are needed only to regenerate decisions from
scratch and are deterministic (temperature 0, pinned model version).

## Citation

Chen, L. (2026). *An Auditable Multi-Agent LLM Framework for U.S. REIT Investment
Decisions: Decomposable Rationales Beyond a Tuned Structured-ML Baseline* (MPhil
dissertation, University of Cambridge).
