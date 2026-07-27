"""Recompute the DeepSeek test-run API cost from the audit ledger.

Background: llm/orchestrator.py hardcodes DeepSeek V3 list prices
(_COST_PER_1M_PROMPT = 0.27, _COST_PER_1M_COMPLETION = 1.10 USD per 1M
tokens), so every `cost_usd` in audit_log/cost_ledger.jsonl was logged at
V3 prices even though the test run used deepseek-v4-flash. This script
recomputes cost from the ledger's *token counts* under clearly-labelled
price scenarios. It never modifies the ledger (append-only, CLAUDE.md
Section 5.4).

Price scenarios:
  - deepseek_v3_prices_as_logged: 0.27 / 1.10 USD per 1M prompt/completion
    tokens, i.e. exactly the constants hardcoded in llm/orchestrator.py.
  - deepseek_v4_flash_list_price: NOT computed. A V4-flash list price was
    searched for in config/ and repo docs (2026-07-04) and none exists
    locally; live API/web lookups are out of scope. The row is emitted
    with the note "V4-flash list price unavailable locally".

Entry scenarios (rows):
  - test_run_575_last_entry_per_key: the reported 575-decision test run.
    The ledger is append-only and contains superseded appends from earlier
    partial runs, so the run is reconstructed as the LAST deepseek_v4_flash
    ledger entry per (decision_date_t, ticker), restricted to the 575 keys
    present in audit_log/decisions.jsonl. Cross-checked: these token counts
    match decisions.jsonl `tokens` field exactly for all 575 decisions.
  - all_deepseek_v4_flash_entries: every deepseek_v4_flash ledger append
    (includes superseded re-run appends and 2 dev-run rows dated 2022);
    upper bound on what was actually spent on this model.

No API. Deterministic. Writes outputs/llm_deepseek_test/cost_recompute.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "audit_log/cost_ledger.jsonl"
DECISIONS = REPO / "audit_log/decisions.jsonl"
OUT = REPO / "outputs/llm_deepseek_test/cost_recompute.csv"

RUN_MODEL = "deepseek_v4_flash"

# V3 prices exactly as hardcoded in llm/orchestrator.py (_COST_PER_1M_*).
V3_PROMPT_USD_PER_1M = 0.27
V3_COMPLETION_USD_PER_1M = 1.10


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def recompute(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens * V3_PROMPT_USD_PER_1M / 1_000_000
        + completion_tokens * V3_COMPLETION_USD_PER_1M / 1_000_000,
        6,
    )


def main() -> None:
    ledger = [e for e in read_jsonl(LEDGER) if e.get("model") == RUN_MODEL]
    decisions = read_jsonl(DECISIONS)
    run_keys = {(d["decision_date_t"], d["ticker"]) for d in decisions}

    # last ledger append per (date, ticker)
    last_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in ledger:
        last_by_key[(e["decision_date_t"], e["ticker"])] = e

    run_entries = [last_by_key[k] for k in sorted(run_keys)]
    if len(run_entries) != len(run_keys):
        raise RuntimeError("Some decisions have no matching ledger entry.")

    # cross-check: last-entry token counts must equal decisions.jsonl tokens
    dec_tokens = {
        (d["decision_date_t"], d["ticker"]): d["tokens"] for d in decisions
    }
    mismatches = [
        k
        for k, e in ((k, last_by_key[k]) for k in run_keys)
        if e["prompt_tokens"] != dec_tokens[k]["prompt"]
        or e["completion_tokens"] != dec_tokens[k]["completion"]
    ]
    if mismatches:
        raise RuntimeError(
            f"Ledger/decisions token mismatch for {len(mismatches)} keys, "
            f"e.g. {sorted(mismatches)[:3]}"
        )

    def row(
        scenario: str,
        entries: Optional[List[Dict[str, Any]]],
        price_scenario: str,
        prompt_price: Optional[float],
        completion_price: Optional[float],
        note: str,
    ) -> Dict[str, Any]:
        if entries is None or prompt_price is None or completion_price is None:
            return {
                "scenario": scenario,
                "price_scenario": price_scenario,
                "price_per_1m_prompt_usd": prompt_price,
                "price_per_1m_completion_usd": completion_price,
                "n_ledger_entries": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "recomputed_cost_usd": None,
                "ledger_logged_cost_usd": None,
                "note": note,
            }
        pt = sum(e["prompt_tokens"] for e in entries)
        ct = sum(e["completion_tokens"] for e in entries)
        return {
            "scenario": scenario,
            "price_scenario": price_scenario,
            "price_per_1m_prompt_usd": prompt_price,
            "price_per_1m_completion_usd": completion_price,
            "n_ledger_entries": len(entries),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "recomputed_cost_usd": round(
                pt * prompt_price / 1_000_000 + ct * completion_price / 1_000_000,
                6,
            ),
            "ledger_logged_cost_usd": round(
                sum(e["cost_usd"] for e in entries), 6
            ),
            "note": note,
        }

    rows = [
        row(
            "test_run_575_last_entry_per_key",
            run_entries,
            "deepseek_v3_prices_as_logged",
            V3_PROMPT_USD_PER_1M,
            V3_COMPLETION_USD_PER_1M,
            "Reported 575-decision test run: last deepseek_v4_flash ledger "
            "append per (decision_date_t, ticker) restricted to keys in "
            "decisions.jsonl; token counts verified identical to "
            "decisions.jsonl tokens for all 575 decisions.",
        ),
        row(
            "all_deepseek_v4_flash_entries",
            ledger,
            "deepseek_v3_prices_as_logged",
            V3_PROMPT_USD_PER_1M,
            V3_COMPLETION_USD_PER_1M,
            "Every deepseek_v4_flash ledger append, incl. superseded re-run "
            "appends and 2 dev-run rows dated 2022; upper bound on actual "
            "spend for this model.",
        ),
        row(
            "test_run_575_last_entry_per_key",
            None,
            "deepseek_v4_flash_list_price",
            None,
            None,
            "V4-flash list price unavailable locally (searched config/ and "
            "repo docs, 2026-07-04); only the labelled V3 scenario is "
            "computed.",
        ),
    ]

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
