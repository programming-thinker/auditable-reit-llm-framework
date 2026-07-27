"""Fundamentals Agent: diagnoses firm financial health from REIT fundamentals."""

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
from llm.schemas import FundamentalsAgentOutput

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Fundamentals/Financial-Health Agent in a multi-agent REIT investment "
    "system. Produce only strict JSON matching the requested schema."
)

FUND_FEATURES = [
    "ffo_yield_proxy", "leverage", "debt_to_equity", "interest_cover",
    "navprem_book_adj", "book_to_market", "ln_mktcap", "amihud_illiq", "idio_vol",
]


def run_fundamentals_agent(
    inputs: dict[str, Any],
    client: LLMClient,
    *,
    prompt_version: int = 1,
) -> Tuple[FundamentalsAgentOutput, Dict[str, Any]]:
    """Run the Fundamentals Agent.

    Parameters
    ----------
    inputs : dict
        Must contain: ticker, company, sector, formation_date, and the 9
        fundamental features in FUND_FEATURES (values may be "not_available").
    client : LLMClient
    prompt_version : int
    """
    template = load_prompt_template("fundamentals", prompt_version)

    identity_section = render_input_section(
        {
            "ticker": inputs["ticker"],
            "company": inputs["company"],
            "sector": inputs["sector"],
            "formation_date": inputs["formation_date"],
        }
    )
    fund_section = render_input_section({k: inputs.get(k) for k in FUND_FEATURES})

    user_prompt = (
        f"{template}\n\n"
        f"## Input Data\n\n"
        f"REIT Identity:\n{identity_section}\n\n"
        f"Firm fundamentals observed at decision date t:\n{fund_section}"
    )

    messages = build_messages(_SYSTEM_PROMPT, user_prompt)
    response = client.query_messages(messages)

    result, error = safe_parse_and_validate(
        response["content"], FundamentalsAgentOutput, "fundamentals"
    )

    meta = {
        "usage": response.get("usage", {}),
        "latency_sec": response.get("latency_sec", 0.0),
        "cached": response.get("cached", False),
    }

    if result is not None:
        logger.info(
            "fundamentals_agent_done",
            ticker=inputs["ticker"],
            financial_health=result.financial_health,
            cached=meta["cached"],
        )
        return result, meta

    logger.error("fundamentals_agent_fallback", ticker=inputs["ticker"], error=error)
    return FundamentalsAgentOutput(
        probabilities={"increase": 0.34, "hold": 0.33, "reduce": 0.33},
        rationale=f"Fallback: LLM response could not be parsed. Error: {error}",
        facts_cited=[],
        financial_health="adequate",
    ), meta
