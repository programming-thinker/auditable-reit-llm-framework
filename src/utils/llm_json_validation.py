import json
from numbers import Real
from typing import Any, Dict


EXPECTED_QWEN_MODEL = "qwen3-8b"
VALID_SMOKE_MODES = {"non_thinking", "thinking"}
VALID_SIGNALS = {"increase", "hold", "reduce"}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_AGENT_VIEWS = {"positive", "neutral", "negative"}
VALID_AGENT_SCORES = {-2, -1, 0, 1, 2}
REQUIRED_AGENT_KEYS = [
    "momentum_agent",
    "macro_sector_agent",
    "disclosure_risk_agent",
]


def _json_type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "NoneType"
    if isinstance(value, (int, float)):
        return type(value).__name__
    return type(value).__name__


def _strip_markdown_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict:
    """Parse model output and require a top-level JSON object."""
    cleaned = _strip_markdown_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Top-level JSON must be a dict; got {_json_type_name(parsed)}.")
    return parsed


def validate_smoke_test_json(obj: dict, expected_mode: str) -> dict:
    errors = []
    top_level_type = _json_type_name(obj)

    if not isinstance(obj, dict):
        errors.append(f"Top-level JSON must be dict; got {top_level_type}.")
        return {
            "json_valid": False,
            "schema_valid": False,
            "top_level_json_type": top_level_type,
            "validation_error": "; ".join(errors),
        }

    required_fields = ["api_test_passed", "model_requested", "mode", "message"]
    missing = [field for field in required_fields if field not in obj]
    if missing:
        errors.append(f"Missing required fields: {missing}.")

    if "api_test_passed" in obj and not isinstance(obj["api_test_passed"], bool):
        errors.append("api_test_passed must be boolean.")
    if obj.get("api_test_passed") is not True:
        errors.append("api_test_passed must be true.")

    if obj.get("model_requested") != EXPECTED_QWEN_MODEL:
        errors.append(f"model_requested must equal {EXPECTED_QWEN_MODEL}.")

    if expected_mode not in VALID_SMOKE_MODES:
        errors.append(f"Internal expected_mode is invalid: {expected_mode}.")
    if obj.get("mode") not in VALID_SMOKE_MODES:
        errors.append("mode must be one of ['non_thinking', 'thinking'].")
    if obj.get("mode") != expected_mode:
        errors.append(f"mode must equal {expected_mode}.")

    if not isinstance(obj.get("message"), str) or not obj.get("message", "").strip():
        errors.append("message must be a non-empty string.")

    return {
        "json_valid": True,
        "schema_valid": not errors,
        "top_level_json_type": top_level_type,
        "validation_error": "; ".join(errors),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_reit_prediction_json(obj: dict) -> dict:
    required_fields = [
        "ticker",
        "formation_date",
        "p_increase",
        "p_hold",
        "p_reduce",
        "signal",
        "confidence",
        "main_drivers",
        "risk_flags",
        "short_reason",
    ]
    errors = []
    top_level_type = _json_type_name(obj)

    result = {
        "json_valid": isinstance(obj, dict),
        "schema_valid": False,
        "probability_sum": None,
        "validation_error": "",
        "top_level_json_type": top_level_type,
    }

    if not isinstance(obj, dict):
        result["validation_error"] = f"Top-level JSON must be dict; got {top_level_type}."
        return result

    missing = [field for field in required_fields if field not in obj]
    if missing:
        errors.append(f"Missing required fields: {missing}.")

    for field in ["ticker", "formation_date", "short_reason"]:
        if field in obj and (not isinstance(obj[field], str) or not obj[field].strip()):
            errors.append(f"{field} must be a non-empty string.")

    probabilities = []
    for field in ["p_increase", "p_hold", "p_reduce"]:
        value = obj.get(field)
        if not _is_number(value):
            errors.append(f"{field} must be numeric.")
            continue
        if value < 0 or value > 1:
            errors.append(f"{field} must be between 0 and 1.")
        probabilities.append(float(value))

    if len(probabilities) == 3:
        probability_sum = sum(probabilities)
        result["probability_sum"] = probability_sum
        if abs(probability_sum - 1.0) > 0.02:
            errors.append("Probability sum must be within 0.02 of 1.0.")

    if obj.get("signal") not in VALID_SIGNALS:
        errors.append("signal must be one of ['increase', 'hold', 'reduce'].")
    if obj.get("confidence") not in VALID_CONFIDENCE:
        errors.append("confidence must be one of ['low', 'medium', 'high'].")
    if "main_drivers" in obj and not _is_str_list(obj["main_drivers"]):
        errors.append("main_drivers must be a list of strings.")
    if "risk_flags" in obj and not _is_str_list(obj["risk_flags"]):
        errors.append("risk_flags must be a list of strings.")

    result["schema_valid"] = not errors
    result["validation_error"] = "; ".join(errors)
    return result


def validate_multi_agent_json(obj: dict) -> dict:
    """Validate Stage 2 raw multi-agent REIT assessments.

    The LLM should only provide structured sub-agent assessments. Final
    probabilities, signals, and confidence are produced later by Python.
    """
    required_fields = [
        "ticker",
        "formation_date",
        "agent_views",
        "main_drivers",
        "risk_flags",
        "short_reason",
    ]
    errors = []
    top_level_type = _json_type_name(obj)

    result = {
        "json_valid": isinstance(obj, dict),
        "schema_valid": False,
        "validation_error": "",
        "top_level_json_type": top_level_type,
    }

    if not isinstance(obj, dict):
        result["validation_error"] = f"Top-level JSON must be dict; got {top_level_type}."
        return result

    missing = [field for field in required_fields if field not in obj]
    if missing:
        errors.append(f"Missing required fields: {missing}.")

    for field in ["ticker", "formation_date", "short_reason"]:
        if field in obj and (not isinstance(obj[field], str) or not obj[field].strip()):
            errors.append(f"{field} must be a non-empty string.")

    agent_views = obj.get("agent_views")
    if not isinstance(agent_views, dict):
        errors.append("agent_views must be a dict.")
    else:
        missing_agents = [agent for agent in REQUIRED_AGENT_KEYS if agent not in agent_views]
        if missing_agents:
            errors.append(f"Missing required agent views: {missing_agents}.")

        for agent_name in REQUIRED_AGENT_KEYS:
            if agent_name not in agent_views:
                continue
            agent_obj = agent_views[agent_name]
            if not isinstance(agent_obj, dict):
                errors.append(f"{agent_name} must be a dict.")
                continue

            for field in ["view", "score", "reason"]:
                if field not in agent_obj:
                    errors.append(f"{agent_name}.{field} is required.")

            view = agent_obj.get("view")
            if view not in VALID_AGENT_VIEWS:
                errors.append(
                    f"{agent_name}.view must be one of ['positive', 'neutral', 'negative']."
                )

            score = agent_obj.get("score")
            if not isinstance(score, int) or isinstance(score, bool):
                errors.append(f"{agent_name}.score must be an integer.")
            elif score not in VALID_AGENT_SCORES:
                errors.append(f"{agent_name}.score must be one of [-2, -1, 0, 1, 2].")

            reason = agent_obj.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{agent_name}.reason must be a non-empty string.")

    if "main_drivers" in obj and not isinstance(obj["main_drivers"], list):
        errors.append("main_drivers must be a list.")
    if "risk_flags" in obj and not isinstance(obj["risk_flags"], list):
        errors.append("risk_flags must be a list.")

    result["schema_valid"] = not errors
    result["validation_error"] = "; ".join(errors)
    return result
