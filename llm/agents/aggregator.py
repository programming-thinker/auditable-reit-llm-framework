"""Aggregator Agent: combines three specialist agent outputs into consensus."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import structlog

from llm.agents.base import (
    build_messages,
    load_prompt_template,
    safe_parse_and_validate,
)
from llm.llm_client import LLMClient
from llm.schemas import (
    AggregatorAgentOutput,
    DisclosureAgentOutput,
    MacroAgentOutput,
    PriceAgentOutput,
)

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are the Aggregator Agent in a multi-agent REIT investment system. "
    "Produce only strict JSON matching the requested schema."
)


def _format_agent_output(name: str, output: Any) -> str:
    """Format a single agent's output for the Aggregator prompt."""
    probs = output.probabilities
    lines = [
        f"### {name} Agent Output",
        f"- probabilities: increase={probs.increase:.3f}, hold={probs.hold:.3f}, reduce={probs.reduce:.3f}",
        f"- rationale: {output.rationale}",
    ]
    if hasattr(output, "sentiment"):
        lines.append(f"- sentiment: {output.sentiment}")
    if hasattr(output, "regime_label"):
        lines.append(f"- regime_label: {output.regime_label}")
    if hasattr(output, "momentum_state"):
        lines.append(f"- momentum_state: {output.momentum_state}")
    if hasattr(output, "financial_health"):
        lines.append(f"- financial_health: {output.financial_health}")
    if hasattr(output, "facts_cited"):
        lines.append(f"- facts_cited: {output.facts_cited}")
    return "\n".join(lines)


def run_aggregator_agent(
    inputs: dict[str, Any],
    disclosure_output: DisclosureAgentOutput,
    macro_output: MacroAgentOutput,
    price_output: PriceAgentOutput,
    client: LLMClient,
    *,
    fundamentals_output: Any = None,
    prompt_version: int = 1,
) -> Tuple[AggregatorAgentOutput, Dict[str, Any]]:
    """Run the Aggregator Agent.

    Parameters
    ----------
    inputs : dict
        Must contain keys: ticker, company, sector, formation_date.
    disclosure_output : DisclosureAgentOutput
        Output from the Disclosure Agent.
    macro_output : MacroAgentOutput
        Output from the Macro Agent.
    price_output : PriceAgentOutput
        Output from the Price Agent.
    client : LLMClient
        Configured LLM client.
    prompt_version : int
        Prompt template version to use.

    Returns
    -------
    tuple[AggregatorAgentOutput, dict]
        Agent output and response metadata (usage, latency_sec, cached).
    """
    template = load_prompt_template("aggregator", prompt_version)

    # ── build user prompt ─────────────────────────────────────────────
    sections = [
        _format_agent_output("Disclosure", disclosure_output),
        _format_agent_output("Macro", macro_output),
        _format_agent_output("Price", price_output),
    ]
    if fundamentals_output is not None:
        sections.append(_format_agent_output("Fundamentals", fundamentals_output))

    user_prompt = (
        f"{template}\n\n"
        f"## Input Data\n\n"
        f"REIT: {inputs['ticker']} ({inputs['company']}, {inputs['sector']})\n"
        f"Formation date: {inputs['formation_date']}\n\n"
        f"## Specialist Agent Outputs\n\n"
        + "\n\n".join(sections)
    )

    # ── call LLM ─────────────────────────────────────────────────────
    messages = build_messages(_SYSTEM_PROMPT, user_prompt)
    response = client.query_messages(messages)

    # ── parse and validate ────────────────────────────────────────────
    result, error = safe_parse_and_validate(
        response["content"], AggregatorAgentOutput, "aggregator"
    )

    meta = {
        "usage": response.get("usage", {}),
        "latency_sec": response.get("latency_sec", 0.0),
        "cached": response.get("cached", False),
    }

    if result is not None:
        logger.info(
            "aggregator_agent_done",
            ticker=inputs["ticker"],
            agreement_score=result.agreement_score,
            cached=meta["cached"],
        )
        return result, meta

    logger.error("aggregator_agent_fallback", ticker=inputs["ticker"], error=error)
    return AggregatorAgentOutput(
        probabilities={"increase": 0.34, "hold": 0.33, "reduce": 0.33},
        rationale=f"Fallback: LLM response could not be parsed. Error: {error}",
        agreement_score=0.0,
    ), meta
