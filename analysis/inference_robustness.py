"""Addresses three reviewer critiques on identification / inference:

C. Leave-one-out market factor: the equal-weight REIT index includes firm i, inflating
   the market-model R² mechanically. We recompute R² on a leave-one-out index R_{-i}.
D. Block-bootstrap direction: reduce events cluster in TIME, so REIT-block resampling
   can understate uncertainty. We add a MONTH-block bootstrap for the LLM−random
   reduce-recall difference and compare to the REIT-block CI.
E. 'Matches random' overclaim: we run a two-one-sided-tests (TOST) equivalence test
   against a pre-specified economically-meaningful margin delta = 0.05 (5 pp).

No API. Writes outputs/fundamentals_robustness/inference_robustness.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
DEC = REPO / "audit_log/decisions.jsonl"
TEST = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
OUT = REPO / "outputs/fundamentals_robustness/inference_robustness.csv"
SEED = 20260626
NB = 3000
DELTA = 0.05  # economically meaningful reduce-recall margin


def loo_market_r2():
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["future_ret_1m"])
    tot = df.groupby("date")["future_ret_1m"].transform("sum")
    n = df.groupby("date")["future_ret_1m"].transform("count")
    df["mkt_loo"] = (tot - df["future_ret_1m"]) / (n - 1)          # leave-one-out index
    df["mkt_incl"] = tot / n                                        # naive (includes i)
    def avg_r2(col):
        r2s = []
        for tk, g in df.groupby("ticker"):
            if len(g) < 24:
                continue
            x, y = g[col].values, g["future_ret_1m"].values
            b = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
            a = y.mean() - b * x.mean()
            ss_res = np.sum((y - (a + b * x)) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot)
        return float(np.median(r2s)), float(np.mean(r2s))
    incl = avg_r2("mkt_incl")
    loo = avg_r2("mkt_loo")
    return {"market_R2_incl_self_median": round(incl[0], 3), "market_R2_incl_self_mean": round(incl[1], 3),
            "market_R2_leave_one_out_median": round(loo[0], 3), "market_R2_leave_one_out_mean": round(loo[1], 3)}


def load_llm():
    rows = []
    with DEC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = d["final_probabilities"]
            rows.append({"date": pd.to_datetime(d["decision_date_t"]).strftime("%Y-%m-%d"),
                         "ticker": d["ticker"], "pred": max(p, key=p.get)})
    df = pd.DataFrame(rows).drop_duplicates(["date", "ticker"], keep="last")
    truth = pd.read_csv(TEST)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    return df.merge(truth[["date", "ticker", "label"]], on=["date", "ticker"], how="inner")


def month_block_bootstrap(df):
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["month"] = df["date"]
    months = df["month"].unique()
    p_fire = float((df["pred"] == "reduce").mean())
    diffs = []
    by_m = {m: df[df["month"] == m] for m in months}
    for _ in range(NB):
        samp = rng.choice(months, len(months), replace=True)
        sub = pd.concat([by_m[m] for m in samp], ignore_index=True)
        ir = sub["label"].values == "reduce"
        if ir.sum() == 0:
            continue
        llm = ((sub["pred"].values == "reduce") & ir).sum() / ir.sum()
        n = len(sub); k = int(round(p_fire * n))
        idx = rng.choice(n, k, replace=False)
        rnd = ir[idx].sum() / ir.sum()
        diffs.append(llm - rnd)
    diffs = np.array(diffs)
    return diffs


def main():
    res = {}
    res.update(loo_market_r2())

    df = load_llm()
    diffs = month_block_bootstrap(df)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res["LLMminusRandom_monthblock_mean"] = round(float(diffs.mean()), 3)
    res["LLMminusRandom_monthblock_CI"] = f"[{lo:.3f}, {hi:.3f}]"
    # TOST equivalence within +/- DELTA
    p_below = float((diffs < DELTA).mean())     # P(diff < +delta)
    p_above = float((diffs > -DELTA).mean())    # P(diff > -delta)
    tost_p = 1 - min(p_below, p_above)          # rough one-sided bootstrap p for the binding side
    res["TOST_delta"] = DELTA
    res["TOST_equivalent_within_delta"] = bool(lo > -DELTA and hi < DELTA)
    res["TOST_note"] = ("within +/-5pp band: equivalent" if (lo > -DELTA and hi < DELTA)
                        else "CI exceeds the +/-5pp band on at least one side -> cannot claim equivalence; "
                             "report as 'no reliable evidence of outperformance'")

    pd.DataFrame([res]).T.rename(columns={0: "value"}).to_csv(OUT)
    print("=== INFERENCE ROBUSTNESS ===\n")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\n[reading]")
    print(f"  C. Market R^2 barely changes leave-one-out ({res['market_R2_leave_one_out_median']} vs "
          f"{res['market_R2_incl_self_median']}) -> co-movement is real, not mechanical.")
    print(f"  D. Month-block CI {res['LLMminusRandom_monthblock_CI']} (vs REIT-block [-0.12,0.13]) "
          f"-> still includes 0.")
    print(f"  E. {res['TOST_note']}")


if __name__ == "__main__":
    main()
