"""Tests for the Fundamentals Agent using recorded responses."""

import json
from unittest.mock import MagicMock

from llm.agents.fundamentals import FUND_FEATURES, run_fundamentals_agent
from llm.schemas import FundamentalsAgentOutput

RECORDED_RESPONSE = json.dumps(
    {
        "ticker": "EQIX",
        "formation_date": "2023-06-30",
        "probabilities": {"increase": 0.50, "hold": 0.30, "reduce": 0.20},
        "rationale": "Low leverage and strong interest cover; FFO yield stable.",
        "facts_cited": [
            "leverage: 0.35 (below sector norm)",
            "interest_cover: 5.2x",
        ],
        "financial_health": "strong",
    }
)

SAMPLE_INPUTS = {
    "ticker": "EQIX",
    "company": "Equinix",
    "sector": "Data Center",
    "formation_date": "2023-06-30",
    "ffo_yield_proxy": 0.045,
    "leverage": 0.35,
    "debt_to_equity": 1.1,
    "interest_cover": 5.2,
    "navprem_book_adj": 0.12,
    "book_to_market": 0.4,
    "ln_mktcap": 11.2,
    "amihud_illiq": 0.0001,
    "idio_vol": 0.02,
}


def _make_mock_client(content: str) -> MagicMock:
    mock = MagicMock()
    mock.query_messages.return_value = {
        "content": content,
        "model": "deepseek-v4-flash",
        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        "cached": False,
        "latency_sec": 1.5,
        "has_reasoning_content": False,
    }
    return mock


def test_fundamentals_agent_valid_response() -> None:
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_fundamentals_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, FundamentalsAgentOutput)
    assert result.financial_health == "strong"
    assert abs(result.probabilities.increase - 0.50) < 1e-6
    assert abs(result.probabilities.hold - 0.30) < 1e-6
    assert abs(result.probabilities.reduce - 0.20) < 1e-6
    assert len(result.facts_cited) == 2
    assert "leverage" in result.rationale.lower()
    assert "usage" in meta
    assert meta["usage"]["prompt_tokens"] == 500
    client.query_messages.assert_called_once()


def test_fundamentals_agent_fallback_on_invalid_json() -> None:
    client = _make_mock_client("This is not JSON at all.")
    result, meta = run_fundamentals_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, FundamentalsAgentOutput)
    assert result.financial_health == "adequate"
    assert "Fallback" in result.rationale
    assert abs(result.probabilities.increase - 0.34) < 1e-6
    assert "usage" in meta


def test_fundamentals_agent_fallback_on_missing_field() -> None:
    incomplete = json.dumps({"ticker": "EQIX", "probabilities": {"increase": 0.5}})
    client = _make_mock_client(incomplete)
    result, meta = run_fundamentals_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, FundamentalsAgentOutput)
    assert result.financial_health == "adequate"
    assert "Fallback" in result.rationale


def test_fundamentals_agent_not_available_rendering() -> None:
    """Missing fundamentals must render as 'not_available' in the prompt.

    The orchestrator passes the literal string "not_available" for features
    absent from the fundamentals panel; features missing from the inputs
    dict entirely (None) must render identically.
    """
    inputs_missing = {
        "ticker": "EQIX",
        "company": "Equinix",
        "sector": "Data Center",
        "formation_date": "2023-06-30",
        # explicit orchestrator-style marker for one feature
        "leverage": "not_available",
        # all other FUND_FEATURES intentionally omitted -> None
    }
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_fundamentals_agent(inputs_missing, client)

    assert isinstance(result, FundamentalsAgentOutput)
    call_args = client.query_messages.call_args[0][0]
    user_content = call_args[1]["content"]
    for feature in FUND_FEATURES:
        assert f"- {feature}: not_available" in user_content
