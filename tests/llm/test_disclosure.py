"""Tests for the Disclosure Agent using recorded responses."""

import json
from unittest.mock import MagicMock

from llm.agents.disclosure import run_disclosure_agent
from llm.schemas import DisclosureAgentOutput

RECORDED_RESPONSE = json.dumps(
    {
        "ticker": "EQIX",
        "formation_date": "2023-06-30",
        "probabilities": {"increase": 0.45, "hold": 0.35, "reduce": 0.20},
        "rationale": "10-K risk factors show stable tenant base. Recent 8-K announced dividend increase.",
        "facts_cited": [
            "Item 1A: no new material risk factors added",
            "8-K: quarterly dividend increased by 5%",
        ],
        "sentiment": "positive",
    }
)

SAMPLE_INPUTS = {
    "ticker": "EQIX",
    "company": "Equinix",
    "sector": "Data Center",
    "formation_date": "2023-06-30",
    "filing_10k_text": "Item 1A Risk Factors: The company faces risks...",
    "filing_10q_text": "Quarterly revenue increased 8% year-over-year...",
    "filing_8k_texts": ["The Board declared a quarterly dividend of $3.41..."],
}


def _make_mock_client(content: str) -> MagicMock:
    mock = MagicMock()
    mock.query_messages.return_value = {
        "content": content,
        "model": "qwen3-8b",
        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
        "cached": False,
        "latency_sec": 1.5,
        "has_reasoning_content": False,
    }
    return mock


def test_disclosure_agent_valid_response() -> None:
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_disclosure_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, DisclosureAgentOutput)
    assert result.sentiment == "positive"
    assert abs(result.probabilities.increase - 0.45) < 1e-6
    assert abs(result.probabilities.hold - 0.35) < 1e-6
    assert abs(result.probabilities.reduce - 0.20) < 1e-6
    assert len(result.facts_cited) == 2
    assert "dividend" in result.rationale.lower()
    assert "usage" in meta
    assert meta["usage"]["prompt_tokens"] == 500
    client.query_messages.assert_called_once()


def test_disclosure_agent_fallback_on_invalid_json() -> None:
    client = _make_mock_client("This is not JSON at all.")
    result, meta = run_disclosure_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, DisclosureAgentOutput)
    assert result.sentiment == "neutral"
    assert "Fallback" in result.rationale
    assert abs(result.probabilities.increase - 0.34) < 1e-6
    assert "usage" in meta


def test_disclosure_agent_fallback_on_missing_field() -> None:
    incomplete = json.dumps({"ticker": "EQIX", "probabilities": {"increase": 0.5}})
    client = _make_mock_client(incomplete)
    result, meta = run_disclosure_agent(SAMPLE_INPUTS, client)

    assert isinstance(result, DisclosureAgentOutput)
    assert result.sentiment == "neutral"
    assert "Fallback" in result.rationale


def test_disclosure_agent_no_filings() -> None:
    inputs_no_filings = {
        **SAMPLE_INPUTS,
        "filing_10k_text": None,
        "filing_10q_text": None,
        "filing_8k_texts": [],
    }
    client = _make_mock_client(RECORDED_RESPONSE)
    result, meta = run_disclosure_agent(inputs_no_filings, client)

    assert isinstance(result, DisclosureAgentOutput)
    # verify prompt included "Not available" for missing filings
    call_args = client.query_messages.call_args[0][0]
    user_content = call_args[1]["content"]
    assert "Not available" in user_content
