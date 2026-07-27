"""Is the final decision an artifact of the opaque LLM 'aggregator' step?

The pipeline's final_probabilities come from a 5th LLM call (the Aggregator) that
reads the FOUR specialists' outputs (disclosure, macro, price, fundamentals — the
v2 5-agent design) and synthesises a consensus. That step is a black box, in tension
with the 'auditable' selling point. Here we replace it with TRANSPARENT deterministic
rules applied to the SAME four specialist probability vectors (already in
decisions.jsonl) and check whether the conclusion changes:

  - mean         : simple average of the 4 specialists (forecast combination; Clemen 1989)
  - confidence   : weight each agent by its own max-probability (peakedness)
  - disclosure   : disclosure-only (the unique forward-looking text channel)

For each rule we save reduce recall / precision / fire rate / accuracy AND the
final-label agreement with the LLM aggregator, over the full v2 test log
(575 decisions; same filter as analysis/ablation_channels.py: dedupe on
(date, ticker) keep last, inner-merge with the enriched test split).
If the deterministic rules agree with the LLM aggregator and give similar metrics,
the headline conclusion does not depend on the black-box step.

Read-only on the audit log; writes outputs/llm_deepseek_test/aggregation_robustness.csv.
No API calls, no randomness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DECISIONS = REPO / "audit_log/decisions.jsonl"
TEST_SPLIT = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
OUT_CSV = REPO / "outputs/llm_deepseek_test/aggregation_robustness.csv"
LABELS = ["increase", "hold", "reduce"]
SPECIALISTS = ["disclosure", "macro", "price", "fundamentals"]


def vec(p: dict) -> np.ndarray:
    return np.array([p["increase"], p["hold"], p["reduce"]], dtype=float)


def main() -> None:
    # --- load the v2 5-agent audit log (same filter as ablation_channels.py) ---
    rows = []
    with DECISIONS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ao = d["agent_outputs"]
            rec = {
                "date": pd.to_datetime(d["decision_date_t"]).strftime("%Y-%m-%d"),
                "ticker": d["ticker"],
                "_llm": vec(d["final_probabilities"]),
            }
            for a in SPECIALISTS:
                rec[a] = vec(ao[a]["probabilities"]) if ao.get(a) else None
            rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "ticker"], keep="last")
    n_v2 = len(df)

    truth = pd.read_csv(TEST_SPLIT)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    df = df.merge(truth[["date", "ticker", "label"]], on=["date", "ticker"], how="inner")
    y = df["label"].values
    n = len(df)

    # --- deterministic aggregation rules over the SAME 4 specialist vectors ---
    def rule_mean(r) -> np.ndarray:
        return np.mean([r[a] for a in SPECIALISTS], axis=0)

    def rule_conf(r) -> np.ndarray:
        ws = np.array([r[a].max() for a in SPECIALISTS])
        s = ws.sum()
        if not s:
            return rule_mean(r)
        return np.sum([w * r[a] for w, a in zip(ws, SPECIALISTS)], axis=0) / s

    methods = {
        "LLM-aggregator": lambda r: r["_llm"],
        "mean-4-specialists": rule_mean,
        "confidence-weighted": rule_conf,
        "disclosure-only": lambda r: r["disclosure"],
    }

    preds_by_method = {
        name: df.apply(lambda r: LABELS[int(np.argmax(fn(r)))], axis=1).values
        for name, fn in methods.items()
    }
    llm_pred = preds_by_method["LLM-aggregator"]

    out = []
    for name, pred in preds_by_method.items():
        tp = int(((pred == "reduce") & (y == "reduce")).sum())
        fn_ = int(((pred != "reduce") & (y == "reduce")).sum())
        fp = int(((pred == "reduce") & (y != "reduce")).sum())
        n_fire = int((pred == "reduce").sum())
        out.append({
            "aggregation": name,
            "n_decisions": n,
            "reduce_recall": round(tp / (tp + fn_), 3) if (tp + fn_) else 0.0,
            "reduce_precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
            "reduce_npred": n_fire,
            "reduce_fire_rate": round(n_fire / n, 3),
            "accuracy": round(float((pred == y).mean()), 3),
            "label_agreement_with_llm_agg": round(float((pred == llm_pred).mean()), 3),
            "n_agree_with_llm_agg": int((pred == llm_pred).sum()),
        })
    res = pd.DataFrame(out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"v2 decisions in log = {n_v2}; joined with test split = {n}\n")
    print(res.to_string(index=False))
    mean_row = res.loc[res["aggregation"] == "mean-4-specialists"].iloc[0]
    print(
        f"\nLLM-aggregator vs deterministic mean (4 specialists): "
        f"final-label agreement = {mean_row['label_agreement_with_llm_agg']:.3f} "
        f"({mean_row['n_agree_with_llm_agg']}/{n})"
    )
    print("\n[reading] If the deterministic rules give similar reduce_recall/accuracy to the "
          "LLM-aggregator, the headline conclusion does NOT depend on the black-box step "
          "-> strengthens the auditability claim.")
    print(f"\nWrote {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
