"""Subset validation of the disclosure-text fix.

Bug: the disclosure agent was fed the first 15k chars of the raw filing, which are
pure XBRL metadata -- the actual Item 1A Risk Factors prose (chars ~44k-307k) was
never seen. Here we re-extract the real Item 1A section, re-run disclosure +
aggregator (macro/price unchanged) on a stratified subset that includes true-reduce
cases, and compare reduce detection OLD (XBRL) vs NEW (real risk factors).

Live API calls (disclosure+aggregator only) on ~25 decisions. Writes to
outputs/llm_deepseek_test/disclosure_fix_subset.csv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from llm.agents.aggregator import run_aggregator_agent
from llm.agents.disclosure import run_disclosure_agent
from llm.agents.macro import run_macro_agent
from llm.agents.price import run_price_agent
from llm.edgar_client import EdgarClient
from llm.llm_client import LLMClient

REPO = Path(__file__).resolve().parents[1]
DECISIONS = REPO / "audit_log/decisions.jsonl"
TEST_SPLIT = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/llm_deepseek_test/disclosure_fix_subset.csv"
SEED = 20260626
N_REDUCE, N_OTHER = 13, 12


def extract_item1a(raw: str | None, maxlen: int = 15000) -> str | None:
    """Real Item 1A Risk Factors prose: first 'Item 1A' before first 'Item 1B'."""
    if not raw:
        return None
    starts = [m.start() for m in re.finditer(r"item\s*1a", raw, re.I)]
    ends = [m.start() for m in re.finditer(r"item\s*1b", raw, re.I)]
    if not starts:
        m = re.search(r"risk factors", raw, re.I)
        return raw[m.start():m.start() + maxlen] if m else raw[:maxlen]
    s = starts[0]
    e = next((x for x in ends if x > s), s + maxlen)
    return raw[s:min(e, s + maxlen)]


def clean_8k(txt: str, maxlen: int = 3000) -> str:
    m = re.search(r"(Item\s+\d|entered into|resign|appoint|declared|results|completion|pricing)", txt, re.I)
    return txt[(m.start() if m else 0):][:maxlen]


def main() -> None:
    rng = np.random.default_rng(SEED)
    truth = pd.read_csv(TEST_SPLIT)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    truth = truth[["date", "ticker", "label"]]

    # old recorded decisions (test period)
    old = {}
    with DECISIONS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["decision_date_t"].startswith(("2024", "2025")):
                old[(d["decision_date_t"], d["ticker"])] = d
    odf = pd.DataFrame([{"date": k[0], "ticker": k[1],
                         "old_final": max(v["final_probabilities"], key=v["final_probabilities"].get)}
                        for k, v in old.items()])
    df = odf.merge(truth, on=["date", "ticker"], how="inner")

    red = df[df["label"] == "reduce"].sample(min(N_REDUCE, (df["label"] == "reduce").sum()), random_state=SEED)
    oth = df[df["label"] != "reduce"].sample(N_OTHER, random_state=SEED)
    sample = pd.concat([red, oth]).reset_index(drop=True)

    panel = pd.read_csv(PANEL)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    ec = EdgarClient()
    client = LLMClient(model_config_name="deepseek_v4_flash")

    rows = []
    for _, r in sample.iterrows():
        tk, t = r["ticker"], r["date"]
        prow = panel[(panel["ticker"] == tk) & (panel["date"] == t)]
        if prow.empty:
            continue
        prow = prow.iloc[0]
        fil = ec.get_latest_annual_and_quarterly(tk, t)
        ek = ec.get_filings_in_window(tk, "8-K", str((pd.Timestamp(t) - pd.DateOffset(months=6)).date()), t)
        disc_in = {
            "ticker": tk, "company": str(prow["company"]), "sector": str(prow["sector"]),
            "formation_date": t,
            "filing_10k_text": extract_item1a(fil.get("10-K")),
            "filing_10q_text": extract_item1a(fil.get("10-Q"), 10000),
            # ek is ASCENDING by filing_date -> [-5:] keeps the 5 NEWEST (fixed
            # 2026-07-04; [:5] kept the 5 oldest, as in the reported run).
            "filing_8k_texts": [clean_8k(f["text"]) for f in ek[-5:]],
        }
        # macro / price inputs from panel (same as orchestrator)
        macro_in = {"ticker": tk, "company": str(prow["company"]), "sector": str(prow["sector"]),
                    "formation_date": t, **{c: float(prow[c]) for c in
                    ["FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1", "term_spread_10y_2y_lag1", "cpi_yoy_lag1", "UNRATE_lag1"]}}
        price_in = {"ticker": tk, "company": str(prow["company"]), "sector": str(prow["sector"]),
                    "formation_date": t, **{c: float(prow[c]) for c in
                    ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown"]}}
        d_out, _ = run_disclosure_agent(disc_in, client)
        m_out, _ = run_macro_agent(macro_in, client)
        p_out, _ = run_price_agent(price_in, client)
        agg_in = {"ticker": tk, "company": str(prow["company"]), "sector": str(prow["sector"]), "formation_date": t}
        a_out, _ = run_aggregator_agent(agg_in, d_out, m_out, p_out, client)
        newp = a_out.probabilities
        new_final = max({"increase": newp.increase, "hold": newp.hold, "reduce": newp.reduce},
                        key=lambda k: getattr(newp, k))
        rows.append({"ticker": tk, "date": t, "true": r["label"],
                     "old_final": r["old_final"], "new_final": new_final,
                     "new_disc_sentiment": d_out.sentiment,
                     "new_disc_reduce_p": round(d_out.probabilities.reduce, 2)})
        print(f"  {tk} {t} | true={r['label']:8} old={r['old_final']:8} new={new_final:8} | disc={d_out.sentiment}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    def reduce_recall(col):
        red = res[res["true"] == "reduce"]
        return round((red[col] == "reduce").mean(), 3) if len(red) else float("nan")

    print(f"\n{'='*60}\nSUBSET n={len(res)}  (reduce={ (res['true']=='reduce').sum() })")
    print(f"  reduce recall  OLD(XBRL): {reduce_recall('old_final')}   NEW(real text): {reduce_recall('new_final')}")
    print(f"  decisions flipped: {(res['old_final'] != res['new_final']).sum()}/{len(res)}")
    print(f"  disclosure now non-neutral: {(res['new_disc_sentiment'] != 'neutral').mean():.0%}")


if __name__ == "__main__":
    main()
