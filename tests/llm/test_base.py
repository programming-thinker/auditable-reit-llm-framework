"""Tests for llm/agents/base.py utility functions."""

import pytest

from llm.agents.base import (
    build_messages,
    extract_json_object,
    load_prompt_template,
    render_input_section,
    safe_parse_and_validate,
)
from llm.schemas import PriceAgentOutput


# ── extract_json_object ───────────────────────────────────────────────────


def test_extract_plain_json() -> None:
    result = extract_json_object('{"a": 1, "b": "hello"}')
    assert result == {"a": 1, "b": "hello"}


def test_extract_json_with_markdown_fence() -> None:
    text = '```json\n{"a": 1}\n```'
    result = extract_json_object(text)
    assert result == {"a": 1}


def test_extract_json_with_surrounding_text() -> None:
    text = 'Here is my answer:\n{"a": 1}\nThat was my answer.'
    result = extract_json_object(text)
    assert result == {"a": 1}


def test_extract_json_raises_on_no_json() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json_object("no json here")


def test_extract_json_raises_on_list() -> None:
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")


# ── render_input_section ──────────────────────────────────────────────────


def test_render_input_section_basic() -> None:
    result = render_input_section({"ticker": "AMT", "ret_1m": 0.05})
    assert "- ticker: AMT" in result
    assert "- ret_1m: 0.050000" in result


def test_render_input_section_none_values() -> None:
    result = render_input_section({"a": None, "b": float("nan")})
    assert "- a: not_available" in result
    assert "- b: not_available" in result


# ── build_messages ────────────────────────────────────────────────────────


def test_build_messages() -> None:
    msgs = build_messages("sys prompt", "user prompt")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys prompt"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "user prompt"


# ── load_prompt_template ──────────────────────────────────────────────────


def test_load_prompt_template_exists() -> None:
    body = load_prompt_template("disclosure", version=1)
    assert "Disclosure" in body
    assert "---" not in body  # frontmatter stripped


def test_load_prompt_template_missing() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt_template("nonexistent_agent", version=99)


# ── safe_parse_and_validate ───────────────────────────────────────────────


def test_safe_parse_valid() -> None:
    import json
    content = json.dumps({
        "ticker": "AMT",
        "formation_date": "2023-01-31",
        "probabilities": {"increase": 0.3, "hold": 0.4, "reduce": 0.3},
        "rationale": "test",
        "facts_cited": ["fact1"],
        "momentum_state": "flat",
    })
    result, error = safe_parse_and_validate(content, PriceAgentOutput, "price")
    assert result is not None
    assert error == ""
    assert result.momentum_state == "flat"


def test_safe_parse_invalid_json() -> None:
    result, error = safe_parse_and_validate("not json", PriceAgentOutput, "price")
    assert result is None
    assert "json_extraction_failed" in error


def test_safe_parse_schema_mismatch() -> None:
    import json
    content = json.dumps({"ticker": "AMT"})  # missing required fields
    result, error = safe_parse_and_validate(content, PriceAgentOutput, "price")
    assert result is None
    assert "schema_validation_failed" in error
