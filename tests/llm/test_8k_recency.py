"""Regression test for the 8-K recency fix in the disclosure input builder.

EdgarClient.get_filings_in_window returns filings ASCENDING by filing_date.
The as-run code took eight_k_filings[:5], silently keeping the 5 OLDEST 8-Ks
whenever more than 5 fell in the 6-month window. The fix keeps the 5 NEWEST
([-5:]) while preserving ascending order within the kept five.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm.orchestrator import Orchestrator


def _fake_8k(i: int) -> dict:
    return {
        "filing_date": f"2021-{i:02d}-15",
        "accession_nodash": f"acc{i:02d}",
        "text": f"8-K body number {i}",
    }


@patch("llm.orchestrator.LLMClient")
@patch("llm.orchestrator.EdgarClient")
def test_disclosure_inputs_keep_newest_five_8ks(mock_edgar_cls, mock_llm_cls) -> None:
    mock_llm_cls.return_value = MagicMock()

    mock_edgar = MagicMock()
    mock_edgar.get_latest_annual_and_quarterly.return_value = {
        "10-K": None,
        "10-Q": None,
    }
    # 7 filings, ascending by filing_date (EdgarClient contract)
    mock_edgar.get_filings_in_window.return_value = [_fake_8k(i) for i in range(1, 8)]
    mock_edgar_cls.return_value = mock_edgar

    with tempfile.TemporaryDirectory() as tmpdir:
        orch = Orchestrator(
            audit_log_dir=Path(tmpdir) / "audit_log",
            panel_path=Path("data/processed/backtest_ready_panel.csv"),
        )
        base = orch._build_base_inputs("AMT", "2022-01-31")
        inputs = orch._build_disclosure_inputs(base, "2022-01-31")

    # the NEWEST five (3..7), still in ascending order
    assert inputs["filing_8k_texts"] == [f"8-K body number {i}" for i in range(3, 8)]


@patch("llm.orchestrator.LLMClient")
@patch("llm.orchestrator.EdgarClient")
def test_disclosure_inputs_fewer_than_five_8ks_all_kept(
    mock_edgar_cls, mock_llm_cls
) -> None:
    mock_llm_cls.return_value = MagicMock()

    mock_edgar = MagicMock()
    mock_edgar.get_latest_annual_and_quarterly.return_value = {
        "10-K": None,
        "10-Q": None,
    }
    mock_edgar.get_filings_in_window.return_value = [_fake_8k(i) for i in (1, 2, 3)]
    mock_edgar_cls.return_value = mock_edgar

    with tempfile.TemporaryDirectory() as tmpdir:
        orch = Orchestrator(
            audit_log_dir=Path(tmpdir) / "audit_log",
            panel_path=Path("data/processed/backtest_ready_panel.csv"),
        )
        base = orch._build_base_inputs("AMT", "2022-01-31")
        inputs = orch._build_disclosure_inputs(base, "2022-01-31")

    assert inputs["filing_8k_texts"] == [f"8-K body number {i}" for i in (1, 2, 3)]
