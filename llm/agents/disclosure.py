"""Disclosure Agent: analyses SEC filing text to assess risk sentiment."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import structlog

from llm.agents.base import (
    build_messages,
    load_prompt_template,
    render_input_section,
    safe_parse_and_validate,
)
from llm.llm_client import LLMClient
from llm.schemas import DisclosureAgentOutput

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Disclosure Risk Agent in a multi-agent REIT investment system. "
    "Produce only strict JSON matching the requested schema."
)


def run_disclosure_agent(
    inputs: dict[str, Any],
    client: LLMClient,
    *,
    prompt_version: int = 1,
) -> Tuple[DisclosureAgentOutput, Dict[str, Any]]:
    """Run the Disclosure Agent.

    Parameters
    ----------
    inputs : dict
        Must contain keys:
        - ticker, company, sector, formation_date
        - filing_10k_text (str | None)
        - filing_10q_text (str | None)
        - filing_8k_texts (list[str] | None)
    client : LLMClient
        Configured LLM client.
    prompt_version : int
        Prompt template version to use.

    Returns
    -------
    tuple[DisclosureAgentOutput, dict]
        Agent output and response metadata (usage, latency_sec, cached).
    """
    template = load_prompt_template("disclosure", prompt_version)

    # ── build user prompt ─────────────────────────────────────────────
    identity_section = render_input_section(
        {
            "ticker": inputs["ticker"],
            "company": inputs["company"],
            "sector": inputs["sector"],
            "formation_date": inputs["formation_date"],
        }
    )

    filing_10k = inputs.get("filing_10k_text") or "Not available."
    filing_10q = inputs.get("filing_10q_text") or "Not available."

    filing_8k_texts = inputs.get("filing_8k_texts") or []
    if filing_8k_texts:
        filing_8k_section = "\n\n---\n\n".join(
            f"8-K Filing {i + 1}:\n{text}" for i, text in enumerate(filing_8k_texts)
        )
    else:
        filing_8k_section = "No recent 8-K filings available."

    user_prompt = (
        f"{template}\n\n"
        f"## Input Data\n\n"
        f"REIT Identity:\n{identity_section}\n\n"
        f"Latest 10-K Item 1A Risk Factors excerpt:\n{filing_10k}\n\n"
        f"Latest 10-Q excerpt:\n{filing_10q}\n\n"
        f"Recent 8-K filings (prior 6 months):\n{filing_8k_section}"
    )

    # ── call LLM ─────────────────────────────────────────────────────
    messages = build_messages(_SYSTEM_PROMPT, user_prompt)
    response = client.query_messages(messages)

    # ── parse and validate ────────────────────────────────────────────
    result, error = safe_parse_and_validate(
        response["content"], DisclosureAgentOutput, "disclosure"
    )

    meta = {
        "usage": response.get("usage", {}),
        "latency_sec": response.get("latency_sec", 0.0),
        "cached": response.get("cached", False),
    }

    if result is not None:
        logger.info(
            "disclosure_agent_done",
            ticker=inputs["ticker"],
            sentiment=result.sentiment,
            cached=meta["cached"],
        )
        return result, meta

    # fallback: neutral output when parsing fails
    logger.error(
        "disclosure_agent_fallback",
        ticker=inputs["ticker"],
        error=error,
    )
    return DisclosureAgentOutput(
        probabilities={"increase": 0.34, "hold": 0.33, "reduce": 0.33},
        rationale=f"Fallback: LLM response could not be parsed. Error: {error}",
        facts_cited=[],
        sentiment="neutral",
    ), meta
