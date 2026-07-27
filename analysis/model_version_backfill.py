"""Backfill API-returned model version strings for the v2 test run.

Evidence set B, part 3 (zero cost: reads the local diskcache only, no API).

``audit_log/decisions.jsonl`` records only the config alias
(``model_main = "deepseek_v4_flash"``). The response cache at
``.cache/llm_responses`` (diskcache) stores, for every call, the model
string actually returned by the API (``result["model"]`` in
llm/llm_client.py). This script recovers the exact API-returned model
version strings for the v2 run's calls.

Method (deterministic): the orchestrator parsed each cached response's JSON
content into the agent outputs stored in decisions.jsonl, so the agent
``rationale`` strings in decisions.jsonl are exact substrings of exactly one
successful cached response each. We index all v2 rationales
(575 decisions x 5 agents = 2875 expected calls), then scan every cache
entry, parse its JSON content, and match on the exact rationale string.
Matched entries are attributed to the v2 run and their API-returned model
strings tallied.

Output: outputs/fundamentals_robustness/model_version_backfill.csv
Run:    python3 analysis/model_version_backfill.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from diskcache import Cache

REPO = Path(__file__).resolve().parents[1]
DECISIONS_JSONL = REPO / "audit_log" / "decisions.jsonl"
CACHE_DIR = REPO / ".cache" / "llm_responses"
OUT_CSV = REPO / "outputs" / "fundamentals_robustness" / "model_version_backfill.csv"

AGENTS = ["disclosure", "macro", "price", "fundamentals", "aggregator"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_content(content: str) -> dict | None:
    """Parse a cached response's content as JSON (tolerating md fences)."""
    for candidate in (content, _FENCE_RE.sub("", content).strip()):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def build_rationale_index() -> dict[str, list[tuple[str, str, str]]]:
    """Map exact rationale string -> [(date, ticker, agent), ...] for v2."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    with DECISIONS_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            for agent in AGENTS:
                rat = r["agent_outputs"][agent]["rationale"]
                index.setdefault(rat, []).append(
                    (r["decision_date_t"], r["ticker"], agent)
                )
    return index


def main() -> None:
    index = build_rationale_index()
    n_slots = sum(len(v) for v in index.values())

    cache = Cache(str(CACHE_DIR))
    cache_total_by_model: Counter[str] = Counter()
    matched_model_counts: Counter[str] = Counter()
    matched_agent_model: Counter[tuple[str, str]] = Counter()
    matched_slots: set[tuple[str, str, str]] = set()
    n_entries = 0
    n_parse_fail = 0

    for key in cache.iterkeys():
        val = cache.get(key)
        if not isinstance(val, dict):
            continue
        n_entries += 1
        model = str(val.get("model"))
        cache_total_by_model[model] += 1

        parsed = _parse_content(val.get("content", ""))
        if parsed is None:
            n_parse_fail += 1
            continue
        rat = parsed.get("rationale")
        if not isinstance(rat, str):
            continue
        slots = index.get(rat)
        if not slots:
            continue
        matched_model_counts[model] += 1
        for slot in slots:
            matched_slots.add(slot)
            matched_agent_model[(slot[2], model)] += 1
    cache.close()

    rows = []
    for model, cnt in sorted(matched_model_counts.items()):
        rows.append(
            {
                "row_type": "v2_matched_model",
                "key": model,
                "count": cnt,
            }
        )
    for (agent, model), cnt in sorted(matched_agent_model.items()):
        rows.append(
            {
                "row_type": "v2_matched_agent_model",
                "key": f"{agent}|{model}",
                "count": cnt,
            }
        )
    for model, cnt in sorted(cache_total_by_model.items()):
        rows.append(
            {
                "row_type": "cache_total_all_runs_model",
                "key": model,
                "count": cnt,
            }
        )
    rows += [
        {"row_type": "summary", "key": "n_cache_entries_total", "count": n_entries},
        {"row_type": "summary", "key": "n_cache_parse_failures", "count": n_parse_fail},
        {"row_type": "summary", "key": "n_v2_expected_calls", "count": n_slots},
        {
            "row_type": "summary",
            "key": "n_v2_calls_matched_to_cache",
            "count": len(matched_slots),
        },
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["row_type", "key", "count"])
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
