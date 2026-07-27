"""Tests for the Macro Agent using recorded responses."""

import json
from unittest.mock import MagicMock

from llm.agents.macro import run_macro_agent
from llm.schemas import MacroAgentOutput

RECORDED_RESPONSE = json.dumps(
    {
        "ticker": "EQIX",
        "formation_date": "2023-06-30",
        "probabilities": {"increase": 0.40, "hold": 0.35, "reduce": 0.25},
        "rationale": "Fed funds rate elevated at 5.08% but data centre demand remains structurally resilient.",
        "facts_cited": [
            "FEDFUNDS_lag1=5.080000 indicates tight monetary policy",
            "term_spread_10y_2y_lag1=-0.820000 indicates inverted yield curve",
        ],
        "regime_label": "inverted_curve",
    }
)

SAMPLE_INPUTS = {
    "ticker": "EQIX",
    "company": "Equinix",
    "sector": "Data Center",
    "formation_date": "2023-06-30",
    "FEDFUNDS_lag1": 5.08,
    "DGS10_lag1": 3.81,
    "DGS2_lag1": 4.63,
    "term_spread_10y_2y_lag1": -0.82,
    "cpi_yoy_lag1": 4.05,
    "UNRATE_lag1": 3.70,
}


def _make_mock_client(content: str) -> MagicMock:
    mock = MagicMock()
    mock.query_messages.return_value = {
        "content": content,
        "model": "qwen3-8b",
        "usage": {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
        "cached": False,
        "latency_sec": 1.2,
        "has_reasoning_content": False,
    }
    return mock


def test_macro_agent_valid_response() -> None:
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_macro_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, MacroAgentOutput)
    assert result.regime_label == "inverted_curve"
    assert abs(result.probabilities.increase - 0.40) < 1e-6
    assert len(result.facts_cited) == 2
    assert meta["usage"]["prompt_tokens"] == 300
    client.query_messages.assert_called_once()


def test_macro_agent_fallback_on_invalid_json() -> None:
    client = _make_mock_client("broken response")
    result, meta = run_macro_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, MacroAgentOutput)
    assert result.regime_label == "flat"
    assert "Fallback" in result.rationale
    assert "usage" in meta


def test_macro_agent_missing_values() -> None:
    inputs_missing = {**SAMPLE_INPUTS, "cpi_yoy_lag1": None, "UNRATE_lag1": None}
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_macro_agent(inputs_missing, client)

    assert isinstance(result, MacroAgentOutput)
    call_args = client.query_messages.call_args[0][0]
    user_content = call_args[1]["content"]
    assert "not_available" in user_content
