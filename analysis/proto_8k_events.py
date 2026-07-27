"""Last keystone test: do discrete ADVERSE EVENTS in 8-Ks (the channel conceptually
closest to discrete downside) precede 'reduce' outcomes?

LLM extracts adverse-event flags from the provided 8-K text (filing-date enforced);
the label is taken externally. We compare the extracted event_risk_score for
true-reduce vs non-reduce decisions. If reduce decisions are NOT preceded by more
adverse events, the 8-K event channel also lacks signal -> the negative result is
comprehensive.

~50 decisions x 1 extractor call. Writes outputs/llm_deepseek_test/proto_8k_events.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm.edgar_client import EdgarClient
from llm.llm_client import LLMClient

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/llm_deepseek_test/proto_8k_events.csv"
SEED = 20260626
N_PER = 25

PROMPT = """You extract DISCRETE ADVERSE corporate events from the recent 8-K filings of a REIT. Base every flag strictly on the provided text; do not infer or predict. If the text does not state it, the flag is false.

Return ONLY JSON:
{{
  "litigation": <bool>, "covenant_breach_or_amendment": <bool>, "tenant_default_or_major_loss": <bool>,
  "guidance_cut": <bool>, "dividend_cut_or_suspension": <bool>, "unplanned_exec_departure": <bool>,
  "credit_downgrade": <bool>, "impairment_or_writedown": <bool>, "distressed_asset_sale": <bool>,
  "dilutive_equity_raise": <bool>,
  "event_risk_score": <int 0..5; 0=routine/positive only, 5=multiple serious adverse events>,
  "evidence": "<=1 sentence quoting the most material adverse item, or 'no adverse events'"
}}

Recent 8-K filings:
{eightk}
"""


def main() -> None:
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    lab = panel.dropna(subset=["label"])
    lab = lab[(lab["date"] >= "2022-01-01") & (lab["date"] <= "2025-11-30")]
    red = lab[lab["label"] == "reduce"].sample(N_PER, random_state=SEED)
    oth = lab[lab["label"] != "reduce"].sample(N_PER, random_state=SEED)
    sample = pd.concat([red, oth]).reset_index(drop=True)

    ec = EdgarClient()
    client = LLMClient(model_config_name="deepseek_v4_flash")
    rows = []
    for _, r in sample.iterrows():
        tk, t = r["ticker"], r["date"]
        start = str((t - pd.DateOffset(months=6)).date())
        ek = ec.get_filings_in_window(tk, "8-K", start, str(t.date()))
        blob = "\n\n---\n\n".join((f["text"] or "")[:2500] for f in ek[:6]) or "No 8-K filings."
        resp = client.query_messages([
            {"role": "system", "content": "You output only strict JSON."},
            {"role": "user", "content": PROMPT.format(eightk=blob)},
        ])
        try:
            txt = resp["content"]
            ext = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            continue
        adverse = sum(bool(ext.get(k)) for k in
                      ["litigation", "covenant_breach_or_amendment", "tenant_default_or_major_loss",
                       "guidance_cut", "dividend_cut_or_suspension", "unplanned_exec_departure",
                       "credit_downgrade", "impairment_or_writedown", "distressed_asset_sale",
                       "dilutive_equity_raise"])
        rows.append({"ticker": tk, "date": str(t.date()), "label": r["label"],
                     "n_8k": len(ek), "adverse_flags": adverse,
                     "event_risk_score": ext.get("event_risk_score", 0),
                     "evidence": str(ext.get("evidence", ""))[:100]})
        print(f"  {tk} {t.date()} | {r['label']:8} | adverse={adverse} score={ext.get('event_risk_score')} | {str(ext.get('evidence',''))[:60]}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    rr = df[df["label"] == "reduce"]
    nn = df[df["label"] != "reduce"]
    print(f"\n{'='*66}\nSIGNAL CHECK (n={len(df)}: reduce={len(rr)}, other={len(nn)})")
    print(f"  mean event_risk_score   reduce={rr['event_risk_score'].mean():.2f}  other={nn['event_risk_score'].mean():.2f}")
    print(f"  mean adverse_flags      reduce={rr['adverse_flags'].mean():.2f}  other={nn['adverse_flags'].mean():.2f}")
    print(f"  any-adverse rate        reduce={(rr['adverse_flags']>0).mean():.2f}  other={(nn['adverse_flags']>0).mean():.2f}")
    diff = rr["event_risk_score"].mean() - nn["event_risk_score"].mean()
    print("\n  VERDICT: " + ("ADVERSE EVENTS precede reduce (reduce score materially higher) -> 8-K event channel has signal -> worth a focused event-extraction model"
                             if diff > 0.5 else
                             "No separation -> reduce is NOT preceded by more 8-K adverse events -> event channel also lacks signal -> comprehensive negative result"))


if __name__ == "__main__":
    main()
