"""Prompt-lock regression test.

The five prompt SHAs in config/config.yaml (PROMPT_SHAS) were locked at
PROMPT_LOCK_TIMESTAMP before the one-shot test run. This test asserts that
the prompt templates on disk still hash to exactly those five SHAs via
Orchestrator._compute_prompts_sha(), so the audit log remains replayable
against the locked prompts. If this test fails, a locked prompt file was
edited in place (forbidden: prompts are versioned, never edited).
"""

from pathlib import Path

import yaml

from llm.orchestrator import Orchestrator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

EXPECTED_AGENTS = {"disclosure", "macro", "price", "fundamentals", "aggregator"}


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_prompt_lock_timestamp_is_set() -> None:
    cfg = _load_config()
    assert cfg.get("PROMPT_LOCK_TIMESTAMP"), (
        "PROMPT_LOCK_TIMESTAMP must be set in config/config.yaml "
        "(the test run is one-shot and prompt-locked)."
    )


def test_compute_prompts_sha_matches_locked_config() -> None:
    cfg = _load_config()
    locked = cfg["PROMPT_SHAS"]

    # exactly the five locked agents, no more, no fewer
    assert set(locked.keys()) == EXPECTED_AGENTS

    computed = Orchestrator._compute_prompts_sha()

    assert computed.disclosure == locked["disclosure"]
    assert computed.macro == locked["macro"]
    assert computed.price == locked["price"]
    assert computed.fundamentals == locked["fundamentals"]
    assert computed.aggregator == locked["aggregator"]
