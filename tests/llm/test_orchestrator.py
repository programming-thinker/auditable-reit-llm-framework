"""Integration tests for Orchestrator using mocked clients."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from llm.orchestrator import Orchestrator, _compute_inputs_hash, _estimate_cost


# ── helper tests ──────────────────────────────────────────────────────────


def test_compute_inputs_hash_deterministic() -> None:
    h1 = _compute_inputs_hash({"a": 1, "b": "x"})
    h2 = _compute_inputs_hash({"a": 1, "b": "x"})
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_inputs_hash_different() -> None:
    h1 = _compute_inputs_hash({"a": 1})
    h2 = _compute_inputs_hash({"a": 2})
    assert h1 != h2


def test_estimate_cost() -> None:
    cost = _estimate_cost(1_000_000, 1_000_000)
    assert abs(cost - (0.27 + 1.10)) < 0.01

    cost_zero = _estimate_cost(0, 0)
    assert cost_zero == 0.0


def test_estimate_cost_small() -> None:
    cost = _estimate_cost(500, 100)
    assert cost > 0
    assert cost < 0.01  # very small amount


# ── orchestrator integration ──────────────────────────────────────────────

MOCK_LLM_RESPONSE_DISCLOSURE = json.dumps({
    "ticker": "AMT",
    "formation_date": "2022-01-31",
    "probabilities": {"increase": 0.40, "hold": 0.35, "reduce": 0.25},
    "rationale": "Stable risk profile.",
    "facts_cited": ["no new risks"],
    "sentiment": "neutral",
})

MOCK_LLM_RESPONSE_MACRO = json.dumps({
    "ticker": "AMT",
    "formation_date": "2022-01-31",
    "probabilities": {"increase": 0.35, "hold": 0.40, "reduce": 0.25},
    "rationale": "Low rates support REITs.",
    "facts_cited": ["FEDFUNDS=0.08"],
    "regime_label": "flat",
})

MOCK_LLM_RESPONSE_PRICE = json.dumps({
    "ticker": "AMT",
    "formation_date": "2022-01-31",
    "probabilities": {"increase": 0.30, "hold": 0.35, "reduce": 0.35},
    "rationale": "Mixed momentum.",
    "facts_cited": ["ret_1m=-0.03"],
    "momentum_state": "flat",
})

MOCK_LLM_RESPONSE_FUNDAMENTALS = json.dumps({
    "ticker": "AMT",
    "formation_date": "2022-01-31",
    "probabilities": {"increase": 0.45, "hold": 0.35, "reduce": 0.20},
    "rationale": "Healthy leverage and coverage.",
    "facts_cited": ["leverage=0.30"],
    "financial_health": "strong",
})

MOCK_LLM_RESPONSE_AGGREGATOR = json.dumps({
    "ticker": "AMT",
    "formation_date": "2022-01-31",
    "probabilities": {"increase": 0.35, "hold": 0.37, "reduce": 0.28},
    "rationale": "Consensus is cautiously neutral.",
    "agreement_score": 0.70,
})

_CALL_COUNT = 0


def _mock_query_messages(messages, **kwargs):
    """Return different responses for sequential agent calls
    (order: disclosure, macro, price, fundamentals, aggregator)."""
    global _CALL_COUNT
    responses = [
        MOCK_LLM_RESPONSE_DISCLOSURE,
        MOCK_LLM_RESPONSE_MACRO,
        MOCK_LLM_RESPONSE_PRICE,
        MOCK_LLM_RESPONSE_FUNDAMENTALS,
        MOCK_LLM_RESPONSE_AGGREGATOR,
    ]
    content = responses[_CALL_COUNT % 5]
    _CALL_COUNT += 1
    return {
        "content": content,
        "model": "qwen3-8b",
        "usage": {"prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480},
        "cached": False,
        "latency_sec": 1.0,
        "has_reasoning_content": False,
    }


@patch("llm.orchestrator.LLMClient")
@patch("llm.orchestrator.EdgarClient")
def test_orchestrator_run_single(mock_edgar_cls, mock_llm_cls) -> None:
    global _CALL_COUNT
    _CALL_COUNT = 0

    # mock LLM client
    mock_llm = MagicMock()
    mock_llm.query_messages.side_effect = _mock_query_messages
    mock_llm_cls.return_value = mock_llm

    # mock EDGAR client
    mock_edgar = MagicMock()
    mock_edgar.get_latest_annual_and_quarterly.return_value = {
        "10-K": "Risk factors text...",
        "10-Q": "Quarterly results...",
    }
    mock_edgar.get_filings_in_window.return_value = []
    mock_edgar_cls.return_value = mock_edgar

    with tempfile.TemporaryDirectory() as tmpdir:
        audit_dir = Path(tmpdir) / "audit_log"

        orch = Orchestrator(
            audit_log_dir=audit_dir,
            panel_path=Path("data/processed/backtest_ready_panel.csv"),
        )

        record = orch.run_single("AMT", "2022-01-31")

        # verify record structure
        assert record.ticker == "AMT"
        assert record.decision_date_t == "2022-01-31"
        assert record.tokens.prompt == 400 * 5  # 5 agent calls (incl. fundamentals)
        assert record.tokens.completion == 80 * 5
        assert record.tokens.cost_usd > 0
        assert record.latency_sec > 0
        assert record.agent_outputs.fundamentals is not None
        assert record.agent_outputs.fundamentals.financial_health == "strong"

        # verify audit log written
        decisions_path = audit_dir / "decisions.jsonl"
        assert decisions_path.exists()
        with decisions_path.open() as f:
            lines = f.readlines()
        assert len(lines) == 1
        decision = json.loads(lines[0])
        assert decision["ticker"] == "AMT"

        # verify cost ledger written
        cost_path = audit_dir / "cost_ledger.jsonl"
        assert cost_path.exists()
        with cost_path.open() as f:
            cost_lines = f.readlines()
        assert len(cost_lines) == 1
        cost_entry = json.loads(cost_lines[0])
        assert cost_entry["prompt_tokens"] == 2000  # 400 * 5 agents
        assert cost_entry["cost_usd"] > 0
