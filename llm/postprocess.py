"""Convert JSONL audit log to predictions.csv.

Output schema must match outputs/tables/quant_only_test_predictions.csv
field-for-field (CLAUDE.md Section 7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

PREDICTIONS_COLUMNS = [
    "decision_date",
    "ticker",
    "prob_increase",
    "prob_hold",
    "prob_reduce",
    "predicted_label",
    "true_label",
    "model_run_id",
]


def _argmax_label(prob_increase: float, prob_hold: float, prob_reduce: float) -> str:
    """Return the label with highest probability."""
    mapping = {"increase": prob_increase, "hold": prob_hold, "reduce": prob_reduce}
    return max(mapping, key=mapping.get)  # type: ignore[arg-type]


def jsonl_to_predictions_csv(
    decisions_jsonl: Path = Path("audit_log/decisions.jsonl"),
    output_csv: Path = Path("audit_log/predictions.csv"),
    true_labels: pd.DataFrame | None = None,
) -> Path:
    """Read decisions.jsonl and produce predictions.csv.

    Parameters
    ----------
    decisions_jsonl : Path
        Path to the decisions JSONL file.
    output_csv : Path
        Path to write the predictions CSV.
    true_labels : DataFrame | None
        Optional DataFrame with columns (decision_date, ticker, true_label)
        to join with predictions.  If None, true_label is set to "unknown".

    Returns
    -------
    Path
        Path to the written CSV.
    """
    if not decisions_jsonl.exists():
        raise FileNotFoundError(f"Decisions file not found: {decisions_jsonl}")

    rows: List[dict] = []
    with decisions_jsonl.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "jsonl_parse_error", line_num=line_num, error=str(exc)
                )
                continue

            probs = record["final_probabilities"]
            rows.append(
                {
                    "decision_date": record["decision_date_t"],
                    "ticker": record["ticker"],
                    "prob_increase": probs["increase"],
                    "prob_hold": probs["hold"],
                    "prob_reduce": probs["reduce"],
                    "predicted_label": _argmax_label(
                        probs["increase"], probs["hold"], probs["reduce"]
                    ),
                    "true_label": "unknown",
                    "model_run_id": record.get("model_main", "unknown"),
                }
            )

    df = pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS)

    # de-duplicate: decisions.jsonl is append-only, so a re-run of the same
    # (decision_date, ticker) appends a new record rather than replacing the
    # old one. Keep the most recent record per key.
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["decision_date", "ticker"], keep="last"
    ).reset_index(drop=True)
    if len(df) < n_before:
        logger.info(
            "duplicate_decisions_dropped",
            n_dropped=n_before - len(df),
            n_kept=len(df),
        )

    # join true labels if provided
    if true_labels is not None:
        true_labels = true_labels[["decision_date", "ticker", "true_label"]].copy()
        df = df.drop(columns=["true_label"]).merge(
            true_labels, on=["decision_date", "ticker"], how="left"
        )
        df["true_label"] = df["true_label"].fillna("unknown")
        # ensure column order
        df = df[PREDICTIONS_COLUMNS]

    df.to_csv(output_csv, index=False)
    logger.info(
        "predictions_csv_written",
        path=str(output_csv),
        n_rows=len(df),
    )
    return output_csv
