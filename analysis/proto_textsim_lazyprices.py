"""Decisive (free) keystone test: the ACTUAL Lazy Prices (Cohen-Malloy-Nguyen 2020)
signal = MECHANICAL year-over-year text similarity of Item 1A, not a coarse LLM
judgment. We compute it across ALL 25 REITs x all consecutive 10-K pairs (~200+
points, real power) and correlate '% of risk-factor text that changed' with the
forward 12-month reduce-rate and return.

If even this full-sample mechanical signal is weak -> the text-change channel does
not carry exploitable downside signal for this task -> the V2 keystone is unfounded
-> do not commit 10 days (or commit knowing it is a negative result). If strong ->
the LLM coarse judgment was the bottleneck; V2 should use mechanical-diff + LLM
categorisation.

No API. Writes outputs/llm_deepseek_test/proto_textsim.csv.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm.edgar_client import EdgarClient

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
META = REPO / "data/interim/filing_metadata.csv"
OUT = REPO / "outputs/llm_deepseek_test/proto_textsim.csv"


def item1a(raw: str | None, maxlen: int = 40000) -> str:
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


def forward(panel, tk, fdate):
    g = panel[(panel["ticker"] == tk) & (panel["date"] > fdate) &
              (panel["date"] <= fdate + pd.DateOffset(months=12))]
    if g.empty:
        return np.nan, np.nan
    return (g["label"] == "reduce").mean(), (1 + g["ret_1m"]).prod() - 1


def main() -> None:
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    meta = pd.read_csv(META, parse_dates=["filing_date"])
    tenk = meta[meta["form"].str.contains("10-K", na=False)].sort_values(["ticker", "filing_date"])
    ec = EdgarClient()

    rows = []
    for tk in sorted(tenk["ticker"].unique()):
        dts = list(tenk[tenk["ticker"] == tk]["filing_date"])
        texts = {}
        for d in dts:
            raw = ec.get_latest_annual_and_quarterly(tk, str((d + pd.Timedelta(days=3)).date())).get("10-K")
            texts[d] = item1a(raw)
        for i in range(1, len(dts)):
            a, b = texts[dts[i - 1]], texts[dts[i]]
            if len(a) < 500 or len(b) < 500:
                continue
            try:
                tf = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform([a, b])
                sim = float(cosine_similarity(tf[0], tf[1])[0, 0])
            except Exception:  # noqa: BLE001
                continue
            pct_changed = 1 - sim
            fr, ret = forward(panel, tk, dts[i])
            rows.append({"ticker": tk, "filing_date": str(dts[i].date()),
                         "cos_sim": round(sim, 4), "pct_changed": round(pct_changed, 4),
                         "fwd_reduce_rate": round(fr, 3) if fr == fr else np.nan,
                         "fwd_12m_ret": round(ret, 3) if ret == ret else np.nan})
    df = pd.DataFrame(rows).dropna(subset=["fwd_reduce_rate"])
    df.to_csv(OUT, index=False)

    c1 = df[["pct_changed", "fwd_reduce_rate"]].corr().iloc[0, 1]
    c2 = df[["pct_changed", "fwd_12m_ret"]].corr().iloc[0, 1]
    # tercile split on pct_changed
    df["bucket"] = pd.qcut(df["pct_changed"], 3, labels=["least_changed", "mid", "most_changed"])
    g = df.groupby("bucket", observed=True)[["fwd_reduce_rate", "fwd_12m_ret"]].mean()

    print(f"SIGNAL CHECK (n={len(df)} REIT-year pairs)\n")
    print(g.round(3).to_string())
    print(f"\n  corr(% Item1A changed, fwd_reduce_rate) = {c1:+.3f}")
    print(f"  corr(% Item1A changed, fwd_12m_ret)     = {c2:+.3f}")
    strong = (c1 > 0.15 and c2 < -0.10)
    mod = (c1 > 0.08) or (c2 < -0.08)
    print("\n  VERDICT: " + ("STRONG signal -> commit V2 with mechanical-diff keystone"
                             if strong else
                             "MODERATE/borderline -> mechanical diff helps but weak; weigh effort"
                             if mod else
                             "NO signal even mechanically -> text-CHANGE channel lacks downside signal "
                             "for this task -> V2 keystone unfounded; treat LLM result as negative"))


if __name__ == "__main__":
    main()
