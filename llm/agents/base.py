"""Shared utilities for all agents: prompt rendering, JSON parsing, validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


# ── Prompt loading ────────────────────────────────────────────────────────


def load_prompt_template(agent_name: str, version: int = 1) -> str:
    """Load a versioned prompt markdown file and return its body.

    The YAML frontmatter is stripped; only the markdown body is returned.
    """
    path = PROMPTS_DIR / f"{agent_name}_v{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    raw = path.read_text(encoding="utf-8")
    # strip YAML frontmatter (between --- delimiters)
    body = re.sub(r"^---\n.*?\n---\n", "", raw, count=1, flags=re.DOTALL)
    return body.strip()


# ── Prompt rendering ─────────────────────────────────────────────────────


def render_input_section(data: dict[str, Any]) -> str:
    """Render a flat dict of input variables into a readable text block.

    Produces lines like:
        - key: value
    Missing / NaN values are rendered as "not_available".
    """
    lines: list[str] = []
    for key, value in data.items():
        if value is None or (isinstance(value, float) and value != value):
            display = "not_available"
        elif isinstance(value, float):
            display = f"{value:.6f}"
        else:
            display = str(value)
        lines.append(f"- {key}: {display}")
    return "\n".join(lines)


def build_messages(
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """Build an OpenAI-style messages list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ── JSON extraction & validation ─────────────────────────────────────────


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text string.

    Handles common LLM output issues: leading/trailing whitespace,
    markdown code fences, text before/after the JSON.

    Raises ValueError if no valid JSON object is found.
    """
    # strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # find first { ... } block
    match = re.search(r"\{", cleaned)
    if match is None:
        raise ValueError("No JSON object found in response")

    start = match.start()
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break

    raise ValueError("No valid JSON object found in response")


def validate_output(raw_json: dict[str, Any], schema: type[T]) -> T:
    """Validate a raw JSON dict against a Pydantic model.

    Returns the validated model instance.
    Raises ValidationError on schema mismatch.
    """
    return schema.model_validate(raw_json)


def safe_parse_and_validate(
    content: str,
    schema: type[T],
    agent_name: str,
) -> tuple[T | None, str]:
    """Extract JSON, validate against schema, return (model, error_msg).

    Returns (validated_model, "") on success, or (None, error_description) on failure.
    """
    try:
        raw = extract_json_object(content)
    except ValueError as exc:
        logger.warning("json_extraction_failed", agent=agent_name, error=str(exc))
        return None, f"json_extraction_failed: {exc}"

    try:
        validated = validate_output(raw, schema)
    except ValidationError as exc:
        logger.warning("schema_validation_failed", agent=agent_name, error=str(exc))
        return None, f"schema_validation_failed: {exc}"

    return validated, ""
