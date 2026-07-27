"""Decomposable-rationale case studies (the title's 'Beyond' demonstration).

Pulls illustrative decisions from the v2 5-agent audit log and renders the full
decomposable, grounded reasoning trace -- the auditable artifact the opaque
structured baseline cannot produce. Selects: (1) a correctly-flagged true-reduce
with grounded disclosure/fundamentals; (2) a high-disagreement case. Writes a
markdown table to outputs/llm_deepseek_test/case_studies.md.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DECISIONS = REPO / "audit_log/decisions.jsonl"
TEST = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
OUT = REPO / "outputs/llm_deepseek_test/case_studies.md"


def amax(p):
    return max(p, key=p.get)


def fmt(p):
    return f"inc {p['increase']:.2f} / hold {p['hold']:.2f} / red {p['reduce']:.2f}"


def render(d, truth):
    ao = d["agent_outputs"]
    lines = [f"### {d['ticker']} @ {d['decision_date_t']}  (true label: **{truth}**, "
             f"framework: **{amax(d['final_probabilities'])}**)", ""]
    for key, label, tagk in [("disclosure", "Disclosure", "sentiment"),
                             ("macro", "Macro", "regime_label"),
                             ("price", "Price", "momentum_state"),
                             ("fundamentals", "Fundamentals", "financial_health")]:
        a = ao.get(key)
        if not a:
            continue
        lines.append(f"- **{label}** [{fmt(a['probabilities'])} | {a.get(tagk)}]")
        lines.append(f"    - rationale: {a['rationale']}")
        facts = a.get("facts_cited") or []
        if facts:
            lines.append(f"    - cites: {'; '.join(str(x) for x in facts[:3])}")
    agg = ao["aggregator"]
    lines.append(f"- **Aggregator** [{fmt(agg['probabilities'])} | agreement {agg['agreement_score']}]")
    lines.append(f"    - {agg['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main():
    rows = []
    with DECISIONS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_key = {(r["decision_date_t"], r["ticker"]): r for r in rows}

    truth = pd.read_csv(TEST)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    tmap = {(r["date"], r["ticker"]): r["label"] for _, r in truth.iterrows()}

    def grounded(d):
        f = d["agent_outputs"]["disclosure"].get("facts_cited") or []
        ff = d["agent_outputs"].get("fundamentals", {}).get("facts_cited") or []
        return len(f) + len(ff)

    # case 1: true reduce, predicted reduce, most grounded
    c1 = [r for r in rows if tmap.get((r["decision_date_t"], r["ticker"])) == "reduce"
          and amax(r["final_probabilities"]) == "reduce"]
    c1.sort(key=grounded, reverse=True)
    # case 2: highest disagreement (lowest agreement_score)
    c2 = sorted(rows, key=lambda r: r["agent_outputs"]["aggregator"]["agreement_score"])

    out = ["# Decomposable-Rationale Case Studies (v2 5-agent framework)", "",
           "The framework's value: an auditable, decomposed, fact-grounded reasoning trace "
           "for each decision — beyond the opaque structured baseline. (Predictive accuracy "
           "is bounded by systematic risk; see economic backbone.)", "",
           "## Case 1 — correctly flagged downside, grounded rationale", ""]
    if c1:
        out.append(render(c1[0], tmap.get((c1[0]["decision_date_t"], c1[0]["ticker"]))))
    out += ["## Case 2 — high specialist disagreement (auditable dissent)", ""]
    if c2:
        out.append(render(c2[0], tmap.get((c2[0]["decision_date_t"], c2[0]["ticker"]))))

    OUT.write_text("\n".join(out))
    print(f"written -> {OUT}\n")
    print("\n".join(out[:4]))
    if c1:
        print("\n--- CASE 1 preview ---")
        print(render(c1[0], tmap.get((c1[0]["decision_date_t"], c1[0]["ticker"])))[:900])


if __name__ == "__main__":
    main()
