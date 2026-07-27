"""KEYSTONE PROTOTYPE: does the YEAR-OVER-YEAR CHANGE in 10-K Item 1A risk
factors carry signal about future REIT downside?

This de-risks the V2 redesign before committing 10 days. The LLM ONLY extracts
the change between two consecutive Item 1A sections (a reading/comparison task,
contamination-resistant). The forward outcome is computed EXTERNALLY by us from
price data -- the LLM never predicts.

If a higher extracted risk_change_score does NOT separate high- vs low- forward
reduce-rate, the redesign won't help -> stop. If it does -> commit.

~6 REITs x 3 consecutive 10-K pairs = ~18 LLM calls. Writes
outputs/llm_deepseek_test/proto_change_signal.csv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from llm.edgar_client import EdgarClient
from llm.llm_client import LLMClient

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
META = REPO / "data/interim/filing_metadata.csv"
OUT = REPO / "outputs/llm_deepseek_test/proto_change_signal.csv"
REITS = ["BXP", "ARE", "DLR", "PLD", "SPG", "VTR"]

PROMPT = """You compare two consecutive annual 10-K "Item 1A Risk Factors" sections of a REIT and extract ONLY what CHANGED. Do not predict stock returns. Base every item strictly on the provided text.

Return ONLY JSON:
{{
  "risk_change_score": <int -2..2; -2=materially milder risk language, 0=no real change/boilerplate, +2=materially worse/new serious risks>,
  "n_new_risk_topics": <int>,
  "n_intensified_topics": <int>,
  "salient_new_category": "litigation|covenant_leverage|tenant_occupancy|interest_rate|liquidity|impairment|guidance|none",
  "evidence": "<=1 sentence quoting the most material NEW or INTENSIFIED risk language, or 'no material change'"
}}

PRIOR YEAR Item 1A (excerpt):
{prior}

CURRENT YEAR Item 1A (excerpt):
{curr}
"""


def item1a(raw: str | None, maxlen: int = 7000) -> str:
    if not raw:
        return ""
    starts = [m.start() for m in re.finditer(r"item\s*1a", raw, re.I)]
    ends = [m.start() for m in re.finditer(r"item\s*1b", raw, re.I)]
    if not starts:
        m = re.search(r"risk factors", raw, re.I)
        return raw[m.start():m.start() + maxlen] if m else raw[:maxlen]
    s = starts[0]
    e = next((x for x in ends if x > s), s + maxlen)
    return raw[s:min(e, s + maxlen)]


def forward_outcome(panel: pd.DataFrame, tk: str, fdate: pd.Timestamp) -> dict:
    g = panel[(panel["ticker"] == tk) & (panel["date"] > fdate) &
              (panel["date"] <= fdate + pd.DateOffset(months=12))]
    if g.empty:
        return {"fwd_reduce_rate": float("nan"), "fwd_12m_ret": float("nan"), "n_months": 0}
    red = (g["label"] == "reduce").mean()
    ret = (1 + g["ret_1m"]).prod() - 1 if "ret_1m" in g else float("nan")
    return {"fwd_reduce_rate": round(red, 3), "fwd_12m_ret": round(float(ret), 3), "n_months": len(g)}


def main() -> None:
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    meta = pd.read_csv(META, parse_dates=["filing_date"])
    tenk = meta[meta["form"].str.contains("10-K", na=False)].sort_values(["ticker", "filing_date"])
    ec = EdgarClient()
    client = LLMClient(model_config_name="deepseek_v4_flash")

    rows = []
    for tk in REITS:
        dts = list(tenk[tenk["ticker"] == tk]["filing_date"])
        # 3 most recent pairs: compare each of the last 3 filings to its prior
        pairs = [(dts[i - 1], dts[i]) for i in range(len(dts) - 3, len(dts)) if i >= 1]
        for prior_d, curr_d in pairs:
            cur_raw = ec.get_latest_annual_and_quarterly(tk, str((curr_d + pd.Timedelta(days=3)).date())).get("10-K")
            pri_raw = ec.get_latest_annual_and_quarterly(tk, str((prior_d + pd.Timedelta(days=3)).date())).get("10-K")
            prompt = PROMPT.format(prior=item1a(pri_raw), curr=item1a(cur_raw))
            resp = client.query_messages([
                {"role": "system", "content": "You output only strict JSON."},
                {"role": "user", "content": prompt},
            ])
            try:
                txt = resp["content"]
                txt = txt[txt.find("{"):txt.rfind("}") + 1]
                ext = json.loads(txt)
            except Exception as e:  # noqa: BLE001
                print(f"  parse fail {tk} {curr_d.date()}: {e}")
                continue
            out = forward_outcome(panel, tk, curr_d)
            row = {"ticker": tk, "filing_date": str(curr_d.date()),
                   "risk_change_score": ext.get("risk_change_score"),
                   "n_new": ext.get("n_new_risk_topics"), "n_intens": ext.get("n_intensified_topics"),
                   "category": ext.get("salient_new_category"), **out,
                   "evidence": str(ext.get("evidence", ""))[:120]}
            rows.append(row)
            print(f"  {tk} {curr_d.date()} | Δrisk={row['risk_change_score']:+} new={row['n_new']} "
                  f"cat={row['category']:14} | fwd_reduce={out['fwd_reduce_rate']} fwd_ret={out['fwd_12m_ret']}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print(f"\n{'='*70}\nSIGNAL CHECK (n={len(df)})")
    if len(df) >= 6 and df["risk_change_score"].notna().any():
        hi = df[df["risk_change_score"] >= 1]
        lo = df[df["risk_change_score"] <= 0]
        print(f"  risk worsened (Δ>=+1): n={len(hi)}  mean fwd_reduce_rate={hi['fwd_reduce_rate'].mean():.3f}  mean fwd_12m_ret={hi['fwd_12m_ret'].mean():.3f}")
        print(f"  risk flat/milder (Δ<=0): n={len(lo)}  mean fwd_reduce_rate={lo['fwd_reduce_rate'].mean():.3f}  mean fwd_12m_ret={lo['fwd_12m_ret'].mean():.3f}")
        corr = df[["risk_change_score", "fwd_reduce_rate"]].corr().iloc[0, 1]
        corr_r = df[["risk_change_score", "fwd_12m_ret"]].corr().iloc[0, 1]
        print(f"  corr(Δrisk, fwd_reduce_rate) = {corr:+.3f}   corr(Δrisk, fwd_12m_ret) = {corr_r:+.3f}")
        print("\n  VERDICT: " + ("SIGNAL PRESENT -> worsening risk language precedes more reduce / lower returns -> COMMIT V2"
                                  if corr > 0.15 and corr_r < -0.10 else
                                  "WEAK/NO SIGNAL -> change detection alone may not rescue -> reconsider before committing"))


if __name__ == "__main__":
    main()
