"""Tests for the Aggregator Agent using recorded responses."""

import json
from unittest.mock import MagicMock

from llm.agents.aggregator import run_aggregator_agent
from llm.schemas import (
    AggregatorAgentOutput,
    DisclosureAgentOutput,
    MacroAgentOutput,
    PriceAgentOutput,
    Probabilities,
)

RECORDED_RESPONSE = json.dumps(
    {
        "ticker": "EQIX",
        "formation_date": "2023-06-30",
        "probabilities": {"increase": 0.42, "hold": 0.33, "reduce": 0.25},
        "rationale": "Disclosure and Macro agents both lean positive (data centre resilience, stable risk factors). "
        "Price agent is mildly negative. Consensus leans cautiously positive with moderate agreement.",
        "agreement_score": 0.65,
    }
)

SAMPLE_INPUTS = {
    "ticker": "EQIX",
    "company": "Equinix",
    "sector": "Data Center",
    "formation_date": "2023-06-30",
}

DISCLOSURE_OUTPUT = DisclosureAgentOutput(
    probabilities=Probabilities(increase=0.45, hold=0.35, reduce=0.20),
    rationale="Stable risk factors, dividend increase announced.",
    facts_cited=["no new material risks", "dividend up 5%"],
    sentiment="positive",
)

MACRO_OUTPUT = MacroAgentOutput(
    probabilities=Probabilities(increase=0.40, hold=0.35, reduce=0.25),
    rationale="Inverted curve but data centre demand resilient.",
    facts_cited=["FEDFUNDS=5.08", "inverted curve"],
    regime_label="inverted_curve",
)

PRICE_OUTPUT = PriceAgentOutput(
    probabilities=Probabilities(increase=0.30, hold=0.35, reduce=0.35),
    rationale="Mixed short-term momentum, moderate volatility.",
    facts_cited=["ret_1m=0.01", "vol_annualized=0.22"],
    momentum_state="flat",
)


def _make_mock_client(content: str) -> MagicMock:
    mock = MagicMock()
    mock.query_messages.return_value = {
        "content": content,
        "model": "qwen3-8b",
        "usage": {"prompt_tokens": 600, "completion_tokens": 120, "total_tokens": 720},
        "cached": False,
        "latency_sec": 2.0,
        "has_reasoning_content": False,
    }
    return mock


def test_aggregator_agent_valid_response() -> None:
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_aggregator_agent(
        SAMPLE_INPUTS, DISCLOSURE_OUTPUT, MACRO_OUTPUT, PRICE_OUTPUT, client
    )

    assert isinstance(result, AggregatorAgentOutput)
    assert abs(result.agreement_score - 0.65) < 1e-6
    assert abs(result.probabilities.increase - 0.42) < 1e-6
    assert "Disclosure" in result.rationale or "Price" in result.rationale
    assert meta["usage"]["prompt_tokens"] == 600
    client.query_messages.assert_called_once()


def test_aggregator_agent_fallback_on_invalid_json() -> None:
    client = _make_mock_client("invalid")
    result, meta = run_aggregator_agent(
        SAMPLE_INPUTS, DISCLOSURE_OUTPUT, MACRO_OUTPUT, PRICE_OUTPUT, client
    )

    assert isinstance(result, AggregatorAgentOutput)
    assert result.agreement_score == 0.0
    assert "Fallback" in result.rationale
    assert "usage" in meta


def test_aggregator_prompt_contains_agent_outputs() -> None:
    client = _make_mock_client(RECORDED_RESPONSE)
    run_aggregator_agent(
        SAMPLE_INPUTS, DISCLOSURE_OUTPUT, MACRO_OUTPUT, PRICE_OUTPUT, client
    )

    call_args = client.query_messages.call_args[0][0]
    user_content = call_args[1]["content"]
    assert "Disclosure Agent Output" in user_content
    assert "Macro Agent Output" in user_content
    assert "Price Agent Output" in user_content
    assert "positive" in user_content  # disclosure sentiment
    assert "inverted_curve" in user_content  # macro regime
    assert "flat" in user_content  # price momentum
