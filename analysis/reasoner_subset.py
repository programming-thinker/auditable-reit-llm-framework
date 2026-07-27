"""Stronger-model robustness check: does DeepSeek V4-Pro (a stronger model) beat
V4-Flash on the 5-agent framework? The systematic-risk decomposition predicts NO for
prediction (the ceiling is informational, not model-capacity), so this closes the
'you only used a weak model' escape route. We also check whether rationale grounding
improves (the auditability dimension, where there is genuine headroom).

Runs the 5-agent orchestrator with deepseek_v4_pro on a stratified ~70-decision test
subset (writes to a separate audit dir), then compares reduce recall and grounding to
the Flash v2 results on the SAME decisions.

Live API (background). Writes outputs/llm_deepseek_test/reasoner_subset.csv.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from llm.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parents[1]
TEST = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
FLASH = REPO / "audit_log/decisions.jsonl"           # v2 flash (575)
PRO_DIR = Path("/tmp/pro_audit")
OUT = REPO / "outputs/llm_deepseek_test/reasoner_subset.csv"
SEED = 20260626
N_RED, N_OTH = 40, 30
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek_v4_pro"

RISK = ["covenant", "default", "litigation", "impairment", "tenant", "vacanc",
        "lease termination", "going concern", "refinanc", "maturity", "dividend cut",
        "downgrade", "leverage", "occupancy", "guidance"]


def grounded(rec) -> bool:
    ao = rec["agent_outputs"]
    facts = (ao["disclosure"].get("facts_cited") or []) + \
            ((ao.get("fundamentals") or {}).get("facts_cited") or [])
    blob = " ".join(str(x) for x in facts) + " " + ao["disclosure"].get("rationale", "")
    has_money = bool(re.search(r"\$\s?\d|\d+(?:\.\d+)?\s?%", blob))
    has_risk = any(t in blob.lower() for t in RISK)
    return len(facts) >= 1 and (has_money or has_risk)


def amax(p):
    return max(p, key=p.get)


def main():
    truth = pd.read_csv(TEST)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    red = truth[truth["label"] == "reduce"].sample(N_RED, random_state=SEED)
    oth = truth[truth["label"] != "reduce"].sample(N_OTH, random_state=SEED)
    sample = pd.concat([red, oth])
    keys = list(zip(sample["ticker"], sample["date"], sample["label"]))

    # run Pro on the subset
    PRO_DIR.mkdir(parents=True, exist_ok=True)
    (PRO_DIR / "decisions.jsonl").write_text("")  # fresh
    orch = Orchestrator(audit_log_dir=PRO_DIR, model_config_name=MODEL,
                        panel_path=REPO / "data/processed/backtest_ready_panel_enriched.csv")
    pro = {}
    for tk, dt, lab in keys:
        try:
            rec = orch.run_single(tk, dt)
            pro[(tk, dt)] = json.loads(rec.model_dump_json())
        except Exception as e:  # noqa: BLE001
            print(f"  fail {tk} {dt}: {type(e).__name__}: {str(e)[:80]}")

    # load Flash v2 for the same keys
    flash = {}
    with FLASH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            flash[(d["ticker"], d["decision_date_t"])] = d

    rows = []
    for tk, dt, lab in keys:
        r = {"ticker": tk, "date": dt, "true": lab}
        if (tk, dt) in pro:
            r["pro_pred"] = amax(pro[(tk, dt)]["final_probabilities"])
            r["pro_grounded"] = grounded(pro[(tk, dt)])
        if (tk, dt) in flash:
            r["flash_pred"] = amax(flash[(tk, dt)]["final_probabilities"])
            r["flash_grounded"] = grounded(flash[(tk, dt)])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    def rr(col):
        sub = df[df["true"] == "reduce"].dropna(subset=[col])
        return round((sub[col] == "reduce").mean(), 3) if len(sub) else float("nan")

    redmask = df["true"] == "reduce"
    print(f"\n=== STRONGER-MODEL ROBUSTNESS ({MODEL} vs Flash, n={len(df)}, reduce={int(redmask.sum())}) ===")
    print(f"  reduce recall   Pro={rr('pro_pred')}   Flash={rr('flash_pred')}")
    if "pro_grounded" in df:
        print(f"  grounding (all) Pro={df['pro_grounded'].mean():.2f}   Flash={df['flash_grounded'].mean():.2f}")
        gp = df[redmask]["pro_grounded"].mean() if "pro_grounded" in df else float("nan")
        gf = df[redmask]["flash_grounded"].mean() if "flash_grounded" in df else float("nan")
        print(f"  grounding (reduce) Pro={gp:.2f}   Flash={gf:.2f}")
    print(f"  pred agreement Pro vs Flash: "
          f"{(df.dropna(subset=['pro_pred','flash_pred']).eval('pro_pred==flash_pred')).mean():.2f}")
    print("\n  VERDICT: see whether Pro reduce recall materially exceeds Flash / random (~0.20).")


if __name__ == "__main__":
    main()
