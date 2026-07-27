"""Pydantic v2 output models for the multi-agent LLM system.

Data contracts defined in CLAUDE.md Section 7.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Agent-level output
# ---------------------------------------------------------------------------

#: Tolerance for the probability-sum check. Verified 2026-07-04 against all
#: 575 logged decisions in audit_log/decisions.jsonl (deepseek_v4_flash test
#: run): every probability vector (disclosure, macro, price, fundamentals,
#: aggregator, and final) sums to 1 within 1.2e-16, so this validator would
#: not have rejected any logged decision.
PROB_SUM_TOLERANCE = 1e-6


class Probabilities(BaseModel):
    """Three-class probability distribution."""

    increase: float = Field(..., ge=0.0, le=1.0)
    hold: float = Field(..., ge=0.0, le=1.0)
    reduce: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sums_to_one(self) -> "Probabilities":
        """Assert the three probabilities form a valid distribution."""
        total = self.increase + self.hold + self.reduce
        if abs(total - 1.0) >= PROB_SUM_TOLERANCE:
            raise ValueError(
                f"probabilities must sum to 1.0 (got {total!r}: "
                f"increase={self.increase}, hold={self.hold}, "
                f"reduce={self.reduce})"
            )
        return self


class DisclosureAgentOutput(BaseModel):
    """Output from the Disclosure agent."""

    probabilities: Probabilities
    rationale: str
    facts_cited: List[str]
    sentiment: str


class MacroAgentOutput(BaseModel):
    """Output from the Macro agent."""

    probabilities: Probabilities
    rationale: str
    facts_cited: List[str]
    regime_label: str


class PriceAgentOutput(BaseModel):
    """Output from the Price agent."""

    probabilities: Probabilities
    rationale: str
    facts_cited: List[str]
    momentum_state: str


class FundamentalsAgentOutput(BaseModel):
    """Output from the Fundamentals agent (financial-health diagnosis)."""

    probabilities: Probabilities
    rationale: str
    facts_cited: List[str]
    financial_health: str  # strong | adequate | stressed


class AggregatorAgentOutput(BaseModel):
    """Output from the Aggregator agent."""

    probabilities: Probabilities
    rationale: str
    agreement_score: float = Field(..., ge=0.0, le=1.0)


class AgentOutputs(BaseModel):
    """All specialist agent outputs bundled together.

    `fundamentals` is Optional for backward compatibility with v1 (4-agent)
    decision records; the v2 framework populates it.
    """

    disclosure: DisclosureAgentOutput
    macro: MacroAgentOutput
    price: PriceAgentOutput
    aggregator: AggregatorAgentOutput
    fundamentals: Optional[FundamentalsAgentOutput] = None


# ---------------------------------------------------------------------------
# Token / cost tracking
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token counts and cost for a single decision."""

    prompt: int = Field(..., ge=0)
    completion: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Decision record (audit_log/decisions.jsonl schema)
# ---------------------------------------------------------------------------


class PromptsSHA(BaseModel):
    """SHA hashes for each agent's prompt template."""

    disclosure: str
    macro: str
    price: str
    aggregator: str
    fundamentals: Optional[str] = None


class DecisionRecord(BaseModel):
    """One row of audit_log/decisions.jsonl.

    Must conform to CLAUDE.md Section 7 schema exactly.
    """

    decision_date_t: str  # YYYY-MM-DD
    ticker: str
    model_main: str
    prompts_sha: PromptsSHA
    inputs_hash: str  # sha256:...
    agent_outputs: AgentOutputs
    final_probabilities: Probabilities
    tokens: TokenUsage
    latency_sec: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Prediction row (audit_log/predictions.csv schema)
# Must match outputs/tables/quant_only_test_predictions.csv field-for-field.
# ---------------------------------------------------------------------------


class PredictionRow(BaseModel):
    """One row of audit_log/predictions.csv."""

    decision_date: str  # YYYY-MM-DD
    ticker: str
    prob_increase: float
    prob_hold: float
    prob_reduce: float
    predicted_label: str
    true_label: str
    model_run_id: str
