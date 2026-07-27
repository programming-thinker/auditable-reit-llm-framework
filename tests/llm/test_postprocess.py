"""Tests for postprocess: JSONL -> predictions.csv."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from llm.postprocess import jsonl_to_predictions_csv, PREDICTIONS_COLUMNS


def _write_decisions_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


SAMPLE_RECORDS = [
    {
        "decision_date_t": "2023-06-30",
        "ticker": "EQIX",
        "model_main": "qwen3_8b_non_thinking",
        "final_probabilities": {"increase": 0.50, "hold": 0.30, "reduce": 0.20},
    },
    {
        "decision_date_t": "2023-06-30",
        "ticker": "AMT",
        "model_main": "qwen3_8b_non_thinking",
        "final_probabilities": {"increase": 0.20, "hold": 0.30, "reduce": 0.50},
    },
]


def test_basic_conversion() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "decisions.jsonl"
        csv_path = Path(tmpdir) / "predictions.csv"
        _write_decisions_jsonl(jsonl_path, SAMPLE_RECORDS)

        result = jsonl_to_predictions_csv(jsonl_path, csv_path)

        assert result == csv_path
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == PREDICTIONS_COLUMNS
        assert df.iloc[0]["predicted_label"] == "increase"
        assert df.iloc[1]["predicted_label"] == "reduce"
        assert df.iloc[0]["true_label"] == "unknown"


def test_with_true_labels() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "decisions.jsonl"
        csv_path = Path(tmpdir) / "predictions.csv"
        _write_decisions_jsonl(jsonl_path, SAMPLE_RECORDS)

        true_labels = pd.DataFrame([
            {"decision_date": "2023-06-30", "ticker": "EQIX", "true_label": "hold"},
            {"decision_date": "2023-06-30", "ticker": "AMT", "true_label": "reduce"},
        ])

        result = jsonl_to_predictions_csv(jsonl_path, csv_path, true_labels=true_labels)
        df = pd.read_csv(result)
        assert df.iloc[0]["true_label"] == "hold"
        assert df.iloc[1]["true_label"] == "reduce"


def test_schema_matches_v5() -> None:
    """Verify column names match CLAUDE.md Section 7."""
    expected = [
        "decision_date", "ticker", "prob_increase", "prob_hold",
        "prob_reduce", "predicted_label", "true_label", "model_run_id",
    ]
    assert PREDICTIONS_COLUMNS == expected


def test_empty_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "decisions.jsonl"
        csv_path = Path(tmpdir) / "predictions.csv"
        jsonl_path.write_text("")

        result = jsonl_to_predictions_csv(jsonl_path, csv_path)
        df = pd.read_csv(result)
        assert len(df) == 0
        assert list(df.columns) == PREDICTIONS_COLUMNS


def test_malformed_line_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "decisions.jsonl"
        csv_path = Path(tmpdir) / "predictions.csv"
        with jsonl_path.open("w") as f:
            f.write(json.dumps(SAMPLE_RECORDS[0]) + "\n")
            f.write("THIS IS NOT JSON\n")
            f.write(json.dumps(SAMPLE_RECORDS[1]) + "\n")

        result = jsonl_to_predictions_csv(jsonl_path, csv_path)
        df = pd.read_csv(result)
        assert len(df) == 2  # bad line skipped
