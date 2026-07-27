"""Zero-cost evidence set A, part 1: firing-budget sweep.

At each budget b in {2.5%, 5%, ..., 30%} flag the top-b firm-months by
(a) LLM final reduce probability (ties broken by a fixed seeded jitter),
(b) baseline logistic reduce probability (threshold rule: top-b by prob).
Report reduce recall + precision for both at every budget, plus the
random-at-budget expectation (= k/n ~ budget).

Month-block bootstrap 95% band for the LLM-minus-threshold-logistic recall
difference at each budget (B=1000, seeded; resampling pattern reused from
analysis/inference_robustness.py).

No API. Deterministic. Writes outputs/fundamentals_robustness/budget_sweep.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEC = REPO / "audit_log/decisions.jsonl"
BASE = REPO / "outputs/tables/quant_only_test_predictions.csv"
OUT = REPO / "outputs/fundamentals_robustness/budget_sweep.csv"
SEED = 20260626
NB = 1000
BUDGETS = [round(0.025 * i, 4) for i in range(1, 13)]  # 2.5% .. 30%


def load_merged() -> pd.DataFrame:
    rows = []
    with DEC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                {
                    "date": pd.to_datetime(d["decision_date_t"]).strftime("%Y-%m-%d"),
                    "ticker": d["ticker"],
                    "llm_p_reduce": float(d["final_probabilities"]["reduce"]),
                }
            )
    llm = pd.DataFrame(rows).drop_duplicates(["date", "ticker"], keep="last")
    base = pd.read_csv(BASE)
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    df = llm.merge(
        base[["date", "ticker", "true_label", "pred_proba_reduce"]],
        on=["date", "ticker"],
        how="inner",
    )
    assert len(df) == 575, f"expected 575 merged firm-months, got {len(df)}"
    df["is_reduce"] = (df["true_label"] == "reduce").astype(int)
    return df


def topk_recall_precision(
    p: np.ndarray, y: np.ndarray, k: int, tie_rand: np.ndarray
) -> tuple[float, float]:
    """Flag top-k by p (descending), ties broken by tie_rand (ascending)."""
    order = np.lexsort((tie_rand, -p))  # primary: -p, secondary: jitter
    flagged = order[:k]
    hits = int(y[flagged].sum())
    total = int(y.sum())
    recall = hits / total if total else np.nan
    precision = hits / k if k else np.nan
    return recall, precision


def recall_curve(p: np.ndarray, y: np.ndarray, tie_rand: np.ndarray) -> np.ndarray:
    """Cumulative reduce-hit count after flagging the top-j items, j=1..n."""
    order = np.lexsort((tie_rand, -p))
    return np.cumsum(y[order])


def main() -> None:
    df = load_merged()
    n = len(df)
    y = df["is_reduce"].to_numpy()
    p_llm = df["llm_p_reduce"].to_numpy()
    p_log = df["pred_proba_reduce"].to_numpy()

    rng = np.random.default_rng(SEED)
    tie_llm = rng.random(n)  # fixed-seed tie-break for the LLM ranking
    tie_log = rng.random(n)  # same machinery for the (near-tie-free) logistic

    # ---- point estimates -------------------------------------------------
    point = []
    for b in BUDGETS:
        k = int(round(b * n))
        r_llm, pr_llm = topk_recall_precision(p_llm, y, k, tie_llm)
        r_log, pr_log = topk_recall_precision(p_log, y, k, tie_log)
        point.append(
            {
                "budget": b,
                "k_flagged": k,
                "n": n,
                "llm_recall": r_llm,
                "llm_precision": pr_llm,
                "logit_recall": r_log,
                "logit_precision": pr_log,
                "random_expected_recall": k / n,
                "recall_diff_llm_minus_logit": r_llm - r_log,
            }
        )

    # ---- month-block bootstrap for the recall difference -----------------
    months = df["date"].unique()
    by_m = {
        m: (
            df.loc[df["date"] == m, "llm_p_reduce"].to_numpy(),
            df.loc[df["date"] == m, "pred_proba_reduce"].to_numpy(),
            df.loc[df["date"] == m, "is_reduce"].to_numpy(),
        )
        for m in months
    }
    diffs = {b: [] for b in BUDGETS}
    for _ in range(NB):
        samp = rng.choice(months, len(months), replace=True)
        pl = np.concatenate([by_m[m][0] for m in samp])
        pg = np.concatenate([by_m[m][1] for m in samp])
        yy = np.concatenate([by_m[m][2] for m in samp])
        tot = yy.sum()
        if tot == 0:
            continue
        nn = len(yy)
        jit_l = rng.random(nn)
        jit_g = rng.random(nn)
        cum_l = recall_curve(pl, yy, jit_l)
        cum_g = recall_curve(pg, yy, jit_g)
        for b in BUDGETS:
            k = int(round(b * nn))
            if k == 0:
                continue
            diffs[b].append((cum_l[k - 1] - cum_g[k - 1]) / tot)

    for row in point:
        d = np.asarray(diffs[row["budget"]])
        lo, hi = np.percentile(d, [2.5, 97.5])
        row["boot_mean_diff"] = float(d.mean())
        row["boot_ci_lo"] = float(lo)
        row["boot_ci_hi"] = float(hi)
        row["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
        row["n_boot_used"] = int(len(d))

    res = pd.DataFrame(point)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("=== BUDGET SWEEP (reduce class, n=575, 165 reduce events) ===\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(4).to_string(index=False))
    any_sig = res["ci_excludes_zero"].any()
    print(
        f"\n[reading] LLM-minus-logistic recall CI excludes 0 at "
        f"{int(res['ci_excludes_zero'].sum())}/{len(res)} budgets "
        f"({'some' if any_sig else 'no'} budget shows a reliable difference)."
    )
    beats_rand = (res["llm_recall"] > res["random_expected_recall"]).sum()
    print(
        f"          LLM recall exceeds the random-at-budget expectation at "
        f"{beats_rand}/{len(res)} budgets (point estimates)."
    )


if __name__ == "__main__":
    main()
