# Audit Log Schema

> This file documents the schema for audit_log/ artifacts.
> See CLAUDE.md Section 7 for the authoritative data contracts.

## predictions.csv

Must match `outputs/tables/quant_only_test_predictions.csv` field-for-field.

| Column | Type | Description |
|---|---|---|
| `decision_date` | str (YYYY-MM-DD) | Month-end decision date |
| `ticker` | str | REIT ticker symbol |
| `prob_increase` | float | P(increase) |
| `prob_hold` | float | P(hold) |
| `prob_reduce` | float | P(reduce) |
| `predicted_label` | str | Argmax label |
| `true_label` | str | Realized label |
| `model_run_id` | str | Unique run identifier |

New columns may be appended at the end. Document additions here.

## decisions.jsonl (append-only)

One JSON object per line. See CLAUDE.md Section 7 for the full schema.

## cost_ledger.jsonl (append-only)

Tracks cumulative API spend. Hard budget: USD 200 total.
