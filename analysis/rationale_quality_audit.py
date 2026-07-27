"""Rationale-quality audit (reviewer critique: factual *grounding* is necessary but
NOT sufficient for *diagnostic validity* — even true cited facts may not logically
support a 'reduce' view, be materially relevant, or be actionable).

The existing analysis/factuality_audit.py verifies that cited numbers are TRUE
(100% fundamentals match; 90.4% disclosure tokens verifiable). This script goes one
level up: it audits whether the reasoning is VALID, using an independent LLM judge
(DeepSeek V4-Pro) that scores each REDUCE rationale on three dimensions:

  * entailment   (0-2): do the cited facts logically support a downside / reduce view?
  * relevance    (0-2): are the cited facts materially relevant to near-term downside?
  * actionability(0-2): is the rationale specific and decision-useful (not boilerplate)?

This is an AUTOMATED audit (single LLM judge, distinct model family from the V4-Flash
generator to limit self-evaluation bias). It does NOT claim to replace a full two-rater
HUMAN audit with Cohen's kappa, which remains future work; a human spot-check sheet is
emitted alongside so the automated judge can itself be validated against a human subset.
The judge never sees the realized outcome — it grades reasoning quality, not correctness.

Census: ALL reduce-predicted decisions (argmax==reduce). Read-only on decisions/predictions;
calls the LLM judge with temperature=0 + diskcache (deterministic, idempotent). Writes
outputs/llm_deepseek_test/rationale_quality_audit.csv and audit_log/rationale_spotcheck_sheet.csv.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # allow `from llm...` when run as a script

from llm.llm_client import LLMClient  # noqa: E402
DEC = REPO / "audit_log/decisions.jsonl"
PRED = REPO / "audit_log/predictions.csv"
OUT = REPO / "outputs/llm_deepseek_test/rationale_quality_audit.csv"
SHEET = REPO / "audit_log/rationale_spotcheck_sheet.csv"
LEDGER = REPO / "audit_log/cost_ledger.jsonl"
JUDGE_PRIMARY = "deepseek_v4_pro"   # independent of the V4-Flash generator
JUDGE_FALLBACK = "deepseek_v4_flash"
SPOTCHECK_N = 60
SEED = 20260627

JUDGE_SYSTEM = (
    "You are an independent buy-side credit analyst auditing the QUALITY of a reasoning "
    "trace produced by an automated system that recommended REDUCING exposure to a U.S. "
    "REIT. Judge ONLY the reasoning quality, not whether the call later proved correct. "
    "You do not know the realized return. Respond with strict JSON."
)

RUBRIC = """Score the rationale on three 0-2 integer dimensions.

entailment (do the CITED FACTS logically support a downside / 'reduce' conclusion?):
  0 = facts do not support, or actively contradict, a downside view
  1 = facts weakly or only partially support a downside view (mixed signals)
  2 = facts clearly and directly support a downside view

relevance (are the cited facts MATERIALLY relevant to a REIT's near-term downside risk?):
  0 = irrelevant or generic boilerplate
  1 = tangential / second-order
  2 = directly material (leverage, coverage, tenant/credit, refinancing, guidance cut, impairment)

actionability (is the rationale SPECIFIC and decision-useful?):
  0 = vague, could apply to any firm
  1 = somewhat specific
  2 = specific, concrete, names the driver an analyst could act on

Return JSON: {"entailment": int, "relevance": int, "actionability": int, "justification": "one sentence"}"""


def compile_rationale(ao: dict) -> str:
    parts = []
    for agent in ("disclosure", "fundamentals", "macro", "price", "aggregator"):
        a = ao.get(agent)
        if not a:
            continue
        r = (a.get("rationale") or "").strip()
        facts = a.get("facts_cited") or []
        block = f"[{agent.upper()}] {r}"
        if facts:
            block += "\n  facts: " + " | ".join(str(x) for x in facts[:8])
        parts.append(block)
    return "\n".join(parts)


def load_reduce_decisions() -> list[dict]:
    pred = pd.read_csv(PRED)[["decision_date", "ticker", "true_label"]]
    truth = {(r.ticker, r.decision_date): r.true_label for r in pred.itertuples()}
    out = []
    with DEC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fp = d["final_probabilities"]
            if max(fp, key=fp.get) != "reduce":
                continue
            tk, dt = d["ticker"], d["decision_date_t"]
            out.append({
                "ticker": tk, "decision_date": dt,
                "true_label": truth.get((tk, dt), "?"),
                "p_reduce": round(fp["reduce"], 3),
                "rationale_text": compile_rationale(d["agent_outputs"]),
            })
    return out


def make_judge() -> tuple[LLMClient, str]:
    """Prefer the independent V4-Pro judge; fall back to V4-Flash if unavailable."""
    try:
        return LLMClient(JUDGE_PRIMARY), JUDGE_PRIMARY
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {JUDGE_PRIMARY} unavailable ({type(e).__name__}); using {JUDGE_FALLBACK}")
        return LLMClient(JUDGE_FALLBACK), JUDGE_FALLBACK


def _extract_json(text: str) -> dict:
    """Parse a JSON object, tolerating markdown fences or surrounding prose."""
    import re
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        t = m.group(0)
    return json.loads(t)


def judge_one(client: LLMClient, rec: dict) -> dict | None:
    prompt = (
        f"REIT: {rec['ticker']}   Decision date: {rec['decision_date']}\n\n"
        f"Automated reduce-recommendation reasoning trace:\n{rec['rationale_text']}\n\n"
        f"{RUBRIC}"
    )
    try:
        # V4-Pro is a reasoning model: it needs a generous token budget or it spends
        # the whole budget on hidden reasoning and returns empty content.
        resp = client.query(prompt, system_prompt=JUDGE_SYSTEM, temperature=0.0,
                            max_tokens=2048, json_mode=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] judge failed {rec['ticker']} {rec['decision_date']}: {type(e).__name__}")
        return None
    try:
        s = _extract_json(resp["content"])
        return {
            "entailment": int(s["entailment"]),
            "relevance": int(s["relevance"]),
            "actionability": int(s["actionability"]),
            "justification": str(s.get("justification", ""))[:300],
            "_usage": resp.get("usage", {}), "_cached": resp.get("cached", False),
        }
    except Exception:  # noqa: BLE001
        print(f"  [warn] unparseable judge JSON {rec['ticker']} {rec['decision_date']}")
        return None


def main() -> None:
    recs = load_reduce_decisions()
    print(f"[rationale-audit] {len(recs)} reduce-predicted decisions "
          f"({len({r['ticker'] for r in recs})} distinct REITs)\n")
    client, judge_model = make_judge()
    print(f"[judge] {judge_model} (generator was deepseek_v4_flash)\n")

    rows = []
    p_tok = c_tok = 0
    for i, rec in enumerate(recs, 1):
        v = judge_one(client, rec)
        if v is None:
            continue
        u = v.pop("_usage", {})
        v.pop("_cached", None)
        p_tok += u.get("prompt_tokens", 0)
        c_tok += u.get("completion_tokens", 0)
        rows.append({**{k: rec[k] for k in ("ticker", "decision_date", "true_label", "p_reduce")}, **v})
        if i % 25 == 0:
            print(f"  judged {i}/{len(recs)}")

    df = pd.DataFrame(rows)
    if df.empty or "entailment" not in df:
        raise SystemExit(f"[fatal] no judge scores parsed ({len(rows)} rows). "
                         "Check the judge model output format / token budget.")
    df["supported"] = df["entailment"] >= 1      # at least partial logical support
    df["material"] = df["relevance"] >= 1
    df["strong_entail"] = df["entailment"] == 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    n = len(df)
    summary = {
        "judge_model": judge_model,
        "reduce_decisions_judged": n,
        "mean_entailment": round(df["entailment"].mean(), 3),
        "mean_relevance": round(df["relevance"].mean(), 3),
        "mean_actionability": round(df["actionability"].mean(), 3),
        "pct_entailment_supported_ge1": round(df["supported"].mean(), 3),
        "pct_relevance_material_ge1": round(df["material"].mean(), 3),
        "pct_entailment_strong_eq2": round(df["strong_entail"].mean(), 3),
    }
    # break down by whether the reduce call was correct (judge never saw this)
    if "true_label" in df:
        for lab in ("reduce", "hold", "increase"):
            sub = df[df["true_label"] == lab]
            if len(sub):
                summary[f"mean_entailment_when_true_{lab}"] = round(sub["entailment"].mean(), 3)
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        OUT.with_name("rationale_quality_audit_summary.csv"))

    # ---- human spot-check sheet (stratified ~SPOTCHECK_N; NO auto scores, to avoid anchoring) ----
    rng = np.random.default_rng(SEED)
    pool = pd.DataFrame(recs)
    take = []
    per = max(1, round(SPOTCHECK_N / pool["ticker"].nunique()))
    for tk, g in pool.groupby("ticker"):
        idx = rng.permutation(len(g))[:per]
        take.append(g.iloc[idx])
    sheet = pd.concat(take).head(SPOTCHECK_N).reset_index(drop=True)
    sheet = sheet[["ticker", "decision_date", "rationale_text"]].copy()
    for col in ("entailment_0_2", "relevance_0_2", "actionability_0_2", "human_notes"):
        sheet[col] = ""
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET, index=False)

    # ---- cost ledger ----
    est_cost = round((p_tok / 1e6) * 0.28 + (c_tok / 1e6) * 0.42, 4)  # rough DeepSeek-class rate
    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": "rationale_quality_audit", "judge_model": judge_model,
            "prompt_tokens": p_tok, "completion_tokens": c_tok,
            "est_cost_usd": est_cost,
        }) + "\n")

    print("\n=== RATIONALE QUALITY AUDIT (automated, independent judge) ===\n")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  tokens: prompt={p_tok} completion={c_tok}  est_cost_usd~{est_cost}")
    print(f"\n[written] {OUT}")
    print(f"[written] {OUT.with_name('rationale_quality_audit_summary.csv')}")
    print(f"[written] {SHEET}  ({len(sheet)} rows for human spot-check)")
    print("\n[reading] Scores>=1 mean the cited facts at least partially support, and are")
    print("  materially relevant to, a downside view. This upgrades 'grounded' toward")
    print("  'grounded AND diagnostically valid'. A two-human-rater Cohen's-kappa study")
    print("  (validating this automated judge) remains the specified next step.")


if __name__ == "__main__":
    main()
