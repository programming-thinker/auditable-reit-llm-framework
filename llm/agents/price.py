"""Price Agent: assesses momentum and risk signals."""

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
from llm.schemas import PriceAgentOutput

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Price/Momentum Agent in a multi-agent REIT investment system. "
    "Produce only strict JSON matching the requested schema."
)


def run_price_agent(
    inputs: dict[str, Any],
    client: LLMClient,
    *,
    prompt_version: int = 1,
) -> Tuple[PriceAgentOutput, Dict[str, Any]]:
    """Run the Price Agent.

    Parameters
    ----------
    inputs : dict
        Must contain keys:
        - ticker, company, sector, formation_date
        - ret_1m, ret_3m, ret_6m, ret_12m
        - vol_annualized, drawdown
    client : LLMClient
        Configured LLM client.
    prompt_version : int
        Prompt template version to use.

    Returns
    -------
    tuple[PriceAgentOutput, dict]
        Agent output and response metadata (usage, latency_sec, cached).
    """
    template = load_prompt_template("price", prompt_version)

    # ── build user prompt ─────────────────────────────────────────────
    identity_section = render_input_section(
        {
            "ticker": inputs["ticker"],
            "company": inputs["company"],
            "sector": inputs["sector"],
            "formation_date": inputs["formation_date"],
        }
    )

    price_section = render_input_section(
        {
            "ret_1m": inputs.get("ret_1m"),
            "ret_3m": inputs.get("ret_3m"),
            "ret_6m": inputs.get("ret_6m"),
            "ret_12m": inputs.get("ret_12m"),
            "vol_annualized": inputs.get("vol_annualized"),
            "drawdown": inputs.get("drawdown"),
        }
    )

    user_prompt = (
        f"{template}\n\n"
        f"## Input Data\n\n"
        f"REIT Identity:\n{identity_section}\n\n"
        f"Price and risk signals observed at formation date t:\n{price_section}"
    )

    # ── call LLM ─────────────────────────────────────────────────────
    messages = build_messages(_SYSTEM_PROMPT, user_prompt)
    response = client.query_messages(messages)

    # ── parse and validate ────────────────────────────────────────────
    result, error = safe_parse_and_validate(
        response["content"], PriceAgentOutput, "price"
    )

    meta = {
        "usage": response.get("usage", {}),
        "latency_sec": response.get("latency_sec", 0.0),
        "cached": response.get("cached", False),
    }

    if result is not None:
        logger.info(
            "price_agent_done",
            ticker=inputs["ticker"],
            momentum_state=result.momentum_state,
            cached=meta["cached"],
        )
        return result, meta

    logger.error("price_agent_fallback", ticker=inputs["ticker"], error=error)
    return PriceAgentOutput(
        probabilities={"increase": 0.34, "hold": 0.33, "reduce": 0.33},
        rationale=f"Fallback: LLM response could not be parsed. Error: {error}",
        facts_cited=[],
        momentum_state="flat",
    ), meta
