"""Zero-cost evidence set A, parts 2+3: reduce-vs-not calibration + tie analysis.

Part 2 — binary (reduce vs not) probability calibration for three forecasters:
  * LLM final reduce probability (audit_log/decisions.jsonl)
  * baseline logistic reduce probability (outputs/tables/quant_only_test_predictions.csv)
  * climatology: constant 0.287 (test-window reduce base rate, rounded)
Metrics: Brier score, Murphy decomposition (reliability / resolution /
uncertainty, 10 equal-width bins), ECE, and a per-bin reliability-curve table.
Notes the 0.05-grid quantisation of LLM probabilities.

Part 3 — exact top-2 ties in the LLM final probability vectors: count them,
recompute argmax reduce recall re-resolving every top tie TOWARD reduce
(upper bound), and report the delta.

No API. Deterministic. Writes outputs/fundamentals_robustness/calibration_llm.csv
(long format: section / model / metric rows, plus reliability-curve rows).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEC = REPO / "audit_log/decisions.jsonl"
BASE = REPO / "outputs/tables/quant_only_test_predictions.csv"
OUT = REPO / "outputs/fundamentals_robustness/calibration_llm.csv"
CLIM = 0.287
NBINS = 10
CLASSES = ("increase", "hold", "reduce")


def load() -> pd.DataFrame:
    rows = []
    with DEC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fp = d["final_probabilities"]
            rows.append(
                {
                    "date": pd.to_datetime(d["decision_date_t"]).strftime("%Y-%m-%d"),
                    "ticker": d["ticker"],
                    "p_increase": float(fp["increase"]),
                    "p_hold": float(fp["hold"]),
                    "p_reduce": float(fp["reduce"]),
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
    df["y"] = (df["true_label"] == "reduce").astype(int)
    return df


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def murphy_ece(p: np.ndarray, y: np.ndarray, nbins: int = NBINS):
    """Murphy decomposition + ECE over equal-width bins on [0, 1].

    REL = (1/N) sum n_k (pbar_k - ybar_k)^2
    RES = (1/N) sum n_k (ybar_k - ybar)^2
    UNC = ybar (1 - ybar)
    Brier = REL - RES + UNC + within-bin variance term (reported as residual).
    ECE = (1/N) sum n_k |pbar_k - ybar_k|
    """
    n = len(p)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nbins - 1)
    ybar = y.mean()
    rel = res = ece = 0.0
    curve = []
    for k in range(nbins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            curve.append((edges[k], edges[k + 1], 0, np.nan, np.nan))
            continue
        pk, yk = p[m].mean(), y[m].mean()
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - ybar) ** 2
        ece += nk * abs(pk - yk)
        curve.append((edges[k], edges[k + 1], nk, pk, yk))
    rel, res, ece = rel / n, res / n, ece / n
    unc = float(ybar * (1 - ybar))
    return {
        "brier": brier(p, y),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": unc,
        "decomp_residual": float(brier(p, y) - (rel - res + unc)),
        "ece": float(ece),
    }, curve


def tie_analysis(df: pd.DataFrame) -> dict:
    P = df[["p_increase", "p_hold", "p_reduce"]].to_numpy()
    y = df["y"].to_numpy()
    top = P.max(axis=1)
    n_at_top = (P == top[:, None]).sum(axis=1)
    tied = n_at_top >= 2
    tied_reduce_top = tied & (P[:, 2] == top)

    # as-logged convention: max(p, key=p.get) -> first key in (increase, hold,
    # reduce) order wins a tie, so reduce never wins a tie it is part of.
    pred_asis = np.array(CLASSES)[P.argmax(axis=1)]
    # upper bound: any top tie involving reduce resolves to reduce.
    pred_up = pred_asis.copy()
    pred_up[tied_reduce_top] = "reduce"

    n_red = int(y.sum())
    rec_asis = float(((pred_asis == "reduce") & (y == 1)).sum() / n_red)
    rec_up = float(((pred_up == "reduce") & (y == 1)).sum() / n_red)
    fired_asis = int((pred_asis == "reduce").sum())
    fired_up = int((pred_up == "reduce").sum())
    return {
        "n_decisions": len(df),
        "n_top2_ties": int(tied.sum()),
        "n_ties_reduce_at_top": int(tied_reduce_top.sum()),
        "n_reduce_events": n_red,
        "reduce_fired_argmax_asis": fired_asis,
        "reduce_recall_argmax_asis": rec_asis,
        "reduce_fired_ties_to_reduce": fired_up,
        "reduce_recall_ties_to_reduce_upper": rec_up,
        "recall_delta_ties_to_reduce": rec_up - rec_asis,
    }


def main() -> None:
    df = load()
    y = df["y"].to_numpy()
    forecasts = {
        "llm_final": df["p_reduce"].to_numpy(),
        "baseline_logit": df["pred_proba_reduce"].to_numpy(),
        "climatology_0.287": np.full(len(df), CLIM),
    }

    rows = []
    for name, p in forecasts.items():
        met, curve = murphy_ece(p, y)
        for metric, val in met.items():
            rows.append({"section": "summary", "model": name, "metric": metric, "value": val})
        for lo, hi, nk, pk, yk in curve:
            rows.append(
                {
                    "section": "reliability_curve",
                    "model": name,
                    "metric": "bin",
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": nk,
                    "mean_forecast": pk,
                    "observed_freq": yk,
                    "value": (abs(pk - yk) if nk > 0 else np.nan),
                }
            )

    # quantisation note: LLM final reduce probabilities sit (mostly) on a
    # 0.05 grid, which coarsens the reliability curve.
    p_llm = forecasts["llm_final"]
    on_grid = np.isclose(np.round(p_llm / 0.05) * 0.05, p_llm, atol=1e-9)
    rows += [
        {"section": "quantisation", "model": "llm_final", "metric": "n_distinct_reduce_probs",
         "value": float(len(np.unique(p_llm)))},
        {"section": "quantisation", "model": "llm_final", "metric": "share_on_0.05_grid",
         "value": float(on_grid.mean())},
        {"section": "quantisation", "model": "llm_final", "metric": "note",
         "value": "LLM reduce probs are quantised (~0.05 grid); bin-level reliability is coarse and REL/ECE partly reflect the grid, not smooth miscalibration"},
    ]

    ties = tie_analysis(df)
    for metric, val in ties.items():
        rows.append({"section": "ties", "model": "llm_final", "metric": metric, "value": val})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=== CALIBRATION (reduce vs not, n=575, base rate 0.28696) ===\n")
    summ = out[out["section"] == "summary"].pivot(index="metric", columns="model", values="value")
    print(summ.round(5).to_string())
    print("\n--- reliability curves (n, mean forecast, observed freq) ---")
    rc = out[out["section"] == "reliability_curve"]
    for name in forecasts:
        sub = rc[rc["model"] == name]
        print(f"\n{name}:")
        print(
            sub[["bin_lo", "bin_hi", "n", "mean_forecast", "observed_freq"]]
            .round(4)
            .to_string(index=False)
        )
    print("\n--- ties ---")
    for k, v in ties.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
