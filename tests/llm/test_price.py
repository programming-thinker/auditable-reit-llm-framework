"""Tests for the Price Agent using recorded responses."""

import json
from unittest.mock import MagicMock

from llm.agents.price import run_price_agent
from llm.schemas import PriceAgentOutput

RECORDED_RESPONSE = json.dumps(
    {
        "ticker": "AMT",
        "formation_date": "2023-03-31",
        "probabilities": {"increase": 0.25, "hold": 0.35, "reduce": 0.40},
        "rationale": "Negative momentum across all horizons with elevated volatility signals continued weakness.",
        "facts_cited": [
            "ret_1m=-0.045 indicates short-term decline",
            "ret_12m=-0.180 indicates persistent weakness",
        ],
        "momentum_state": "mild_down",
    }
)

SAMPLE_INPUTS = {
    "ticker": "AMT",
    "company": "American Tower",
    "sector": "Infrastructure",
    "formation_date": "2023-03-31",
    "ret_1m": -0.045,
    "ret_3m": -0.082,
    "ret_6m": -0.120,
    "ret_12m": -0.180,
    "vol_annualized": 0.285,
    "drawdown": -0.15,
}


def _make_mock_client(content: str) -> MagicMock:
    mock = MagicMock()
    mock.query_messages.return_value = {
        "content": content,
        "model": "qwen3-8b",
        "usage": {"prompt_tokens": 250, "completion_tokens": 70, "total_tokens": 320},
        "cached": False,
        "latency_sec": 0.9,
        "has_reasoning_content": False,
    }
    return mock


def test_price_agent_valid_response() -> None:
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_price_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, PriceAgentOutput)
    assert result.momentum_state == "mild_down"
    assert abs(result.probabilities.reduce - 0.40) < 1e-6
    assert len(result.facts_cited) == 2
    assert meta["usage"]["prompt_tokens"] == 250
    client.query_messages.assert_called_once()


def test_price_agent_fallback_on_invalid_json() -> None:
    client = _make_mock_client("not json")
    result, meta = run_price_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, PriceAgentOutput)
    assert result.momentum_state == "flat"
    assert "Fallback" in result.rationale
    assert "usage" in meta


def test_price_agent_missing_returns() -> None:
    inputs_missing = {
        **SAMPLE_INPUTS,
        "ret_3m": None,
        "ret_6m": None,
        "ret_12m": None,
    }
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_price_agent(inputs_missing, client)

    assert isinstance(result, PriceAgentOutput)
    call_args = client.query_messages.call_args[0][0]
    user_content = call_args[1]["content"]
    assert user_content.count("not_available") >= 3
