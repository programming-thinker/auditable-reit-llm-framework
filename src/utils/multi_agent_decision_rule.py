from typing import Dict


VALID_AGENT_SCORES = {-2, -1, 0, 1, 2}


def _validate_score(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in [-2, -1, 0, 1, 2].")
    if value not in VALID_AGENT_SCORES:
        raise ValueError(f"{name} must be one of [-2, -1, 0, 1, 2].")
    return value


def apply_multi_agent_decision_rule(
    momentum_score: int,
    macro_sector_score: int,
    disclosure_risk_score: int,
) -> Dict:
    """Transparent score-based Python Decision Agent for Stage 2.

    Qwen supplies only integer sub-agent scores. This function creates the
    final allocation signal without constructing artificial probabilities.
    """
    momentum_score = _validate_score("momentum_score", momentum_score)
    macro_sector_score = _validate_score("macro_sector_score", macro_sector_score)
    disclosure_risk_score = _validate_score("disclosure_risk_score", disclosure_risk_score)

    scores = [momentum_score, macro_sector_score, disclosure_risk_score]
    total_score = sum(scores)
    decision_score = total_score / 3.0
    allocation_score = decision_score
    agent_disagreement = max(scores) - min(scores)

    if total_score >= 2:
        signal = "increase"
    elif total_score <= -2:
        signal = "reduce"
    else:
        signal = "hold"

    if agent_disagreement >= 3 and abs(total_score) < 3:
        signal = "hold"

    if abs(total_score) >= 4 and agent_disagreement <= 2:
        confidence = "high"
    elif abs(total_score) >= 2 and agent_disagreement <= 3:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "momentum_score": momentum_score,
        "macro_sector_score": macro_sector_score,
        "disclosure_risk_score": disclosure_risk_score,
        "total_score": total_score,
        "decision_score": decision_score,
        "allocation_score": allocation_score,
        "agent_disagreement": agent_disagreement,
        "signal": signal,
        "confidence": confidence,
    }
