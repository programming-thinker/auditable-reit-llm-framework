"""Macro Agent: assesses macroeconomic environment for a REIT sector."""

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
from llm.schemas import MacroAgentOutput

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Macro/Sector Agent in a multi-agent REIT investment system. "
    "Produce only strict JSON matching the requested schema."
)


def run_macro_agent(
    inputs: dict[str, Any],
    client: LLMClient,
    *,
    prompt_version: int = 1,
) -> Tuple[MacroAgentOutput, Dict[str, Any]]:
    """Run the Macro Agent.

    Parameters
    ----------
    inputs : dict
        Must contain keys:
        - ticker, company, sector, formation_date
        - FEDFUNDS_lag1, DGS10_lag1, DGS2_lag1
        - term_spread_10y_2y_lag1, cpi_yoy_lag1, UNRATE_lag1
    client : LLMClient
        Configured LLM client.
    prompt_version : int
        Prompt template version to use.

    Returns
    -------
    tuple[MacroAgentOutput, dict]
        Agent output and response metadata (usage, latency_sec, cached).
    """
    template = load_prompt_template("macro", prompt_version)

    # ── build user prompt ─────────────────────────────────────────────
    identity_section = render_input_section(
        {
            "ticker": inputs["ticker"],
            "company": inputs["company"],
            "sector": inputs["sector"],
            "formation_date": inputs["formation_date"],
        }
    )

    macro_section = render_input_section(
        {
            "FEDFUNDS_lag1": inputs.get("FEDFUNDS_lag1"),
            "DGS10_lag1": inputs.get("DGS10_lag1"),
            "DGS2_lag1": inputs.get("DGS2_lag1"),
            "term_spread_10y_2y_lag1": inputs.get("term_spread_10y_2y_lag1"),
            "cpi_yoy_lag1": inputs.get("cpi_yoy_lag1"),
            "UNRATE_lag1": inputs.get("UNRATE_lag1"),
        }
    )

    user_prompt = (
        f"{template}\n\n"
        f"## Input Data\n\n"
        f"REIT Identity:\n{identity_section}\n\n"
        f"Lagged macro signals (available as of the prior month):\n{macro_section}"
    )

    # ── call LLM ─────────────────────────────────────────────────────
    messages = build_messages(_SYSTEM_PROMPT, user_prompt)
    response = client.query_messages(messages)

    # ── parse and validate ────────────────────────────────────────────
    result, error = safe_parse_and_validate(
        response["content"], MacroAgentOutput, "macro"
    )

    meta = {
        "usage": response.get("usage", {}),
        "latency_sec": response.get("latency_sec", 0.0),
        "cached": response.get("cached", False),
    }

    if result is not None:
        logger.info(
            "macro_agent_done",
            ticker=inputs["ticker"],
            regime_label=result.regime_label,
            cached=meta["cached"],
        )
        return result, meta

    logger.error("macro_agent_fallback", ticker=inputs["ticker"], error=error)
    return MacroAgentOutput(
        probabilities={"increase": 0.34, "hold": 0.33, "reduce": 0.33},
        rationale=f"Fallback: LLM response could not be parsed. Error: {error}",
        facts_cited=[],
        regime_label="flat",
    ), meta
