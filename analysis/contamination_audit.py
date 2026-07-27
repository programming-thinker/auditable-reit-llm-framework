"""Pretraining-contamination robustness audit (free, no API).

The headline LLM result is threatened by the fact that DeepSeek/Qwen pretraining
likely covers the 2024-2025 test window: the model may RECALL outcomes rather than
REASON from the filing. We cannot fully exclude weight-level leakage, but we can
gather evidence on which way it leans, using only the existing audit log:

  (1) Input look-ahead integrity: every filing date cited by the disclosure agent
      must be <= the decision date. Any violation = input leakage bug.
  (2) Rationale grounding: for decisions PREDICTED 'reduce', do the cited facts
      reference SPECIFIC filing content (named risks, $ amounts, %, dates) -- which
      supports reasoning -- or only generic market talk -- which is consistent with
      recall / hand-waving?

Writes to outputs/llm_deepseek_test/ only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = REPO / "audit_log/decisions.jsonl"
OUT_DIR = REPO / "outputs/llm_deepseek_test"

# named forward-looking risk terms that would appear in genuine 10-K/8-K grounding
RISK_TERMS = [
    "covenant", "default", "litigation", "lawsuit", "impairment", "bankrupt",
    "downgrade", "tenant", "vacancy", "vacanc", "lease termination", "going concern",
    "restructur", "dilution", "redemption", "refinanc", "maturity", "delinquen",
    "occupancy", "guidance cut", "dividend cut", "suspend", "write-down", "writedown",
]

# A genuine FILING date appears next to "filed"/"8-K"/"(8-K)". In-text future
# references (notes "due 2029", "effective Jan 1, 2025") are NOT filing dates and
# must be excluded, or the look-ahead check produces false positives.
FILING_DATE_RE = re.compile(
    r"(?:filed|8-?K[^.]{0,12}?)\s*(?:on\s+)?"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"\$\s?\d")
PCT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")


def cited_filing_dates(facts: list[str]) -> list[pd.Timestamp]:
    """Extract only dates that are tagged as actual filing dates."""
    dates = []
    for fact in facts:
        for m in FILING_DATE_RE.finditer(fact):
            try:
                dates.append(pd.to_datetime(m.group(1)))
            except (ValueError, TypeError):
                pass
    return dates


def specificity(facts: list[str], rationale: str) -> dict:
    blob = " ".join(facts) + " " + (rationale or "")
    low = blob.lower()
    n_risk = sum(1 for t in RISK_TERMS if t in low)
    return {
        "n_facts_cited": len(facts),
        "has_money": bool(MONEY_RE.search(blob)),
        "has_pct": bool(PCT_RE.search(blob)),
        "n_risk_terms": n_risk,
        "grounded": len(facts) >= 1 and (n_risk >= 1 or bool(MONEY_RE.search(blob))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    lookahead_violations = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            d_date = pd.to_datetime(rec["decision_date_t"])
            disc = rec["agent_outputs"]["disclosure"]
            facts = disc.get("facts_cited") or []
            rationale = disc.get("rationale", "")
            probs = rec["final_probabilities"]
            pred = max(probs, key=probs.get)

            # input look-ahead: any actual FILING date after the decision date is a bug
            fdates = cited_filing_dates(facts)
            future_filings = [str(d.date()) for d in fdates if d > d_date]
            if future_filings:
                lookahead_violations.append(
                    {"date": rec["decision_date_t"], "ticker": rec["ticker"],
                     "cited_future_filing_dates": future_filings}
                )

            spec = specificity(facts, rationale)
            rows.append({
                "date": rec["decision_date_t"], "ticker": rec["ticker"],
                "pred": pred, "sentiment": disc.get("sentiment"), **spec,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "contamination_rationale_audit.csv", index=False)

    red = df[df["pred"] == "reduce"]
    summary = {
        "n_decisions": len(df),
        "n_reduce_predicted": len(red),
        "reduce_grounded_pct": float(red["grounded"].mean()) if len(red) else float("nan"),
        "reduce_mean_facts_cited": float(red["n_facts_cited"].mean()) if len(red) else float("nan"),
        "reduce_mean_risk_terms": float(red["n_risk_terms"].mean()) if len(red) else float("nan"),
        "reduce_with_money_or_pct_pct": float(((red["has_money"] | red["has_pct"]).mean())) if len(red) else float("nan"),
        "all_grounded_pct": float(df["grounded"].mean()) if len(df) else float("nan"),
        "input_lookahead_violations": len(lookahead_violations),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "contamination_summary.csv", index=False)
    if lookahead_violations:
        pd.DataFrame(lookahead_violations).to_csv(
            OUT_DIR / "contamination_lookahead_violations.csv", index=False)

    print("=== CONTAMINATION / GROUNDING AUDIT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n[interpretation]")
    print("  - input_lookahead_violations should be 0 (filing-date enforcement).")
    print("  - reduce_grounded_pct HIGH -> reduce calls cite specific filing facts")
    print("    (supports reasoning). LOW -> generic talk, consistent with recall.")
    print("\n10 sampled reduce rationales -> manual_sample_reduce.csv (eyeball these).")
    if len(red):
        red.head(10).to_csv(OUT_DIR / "manual_sample_reduce.csv", index=False)


if __name__ == "__main__":
    main()
