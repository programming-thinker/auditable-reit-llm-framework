"""Timing and agreement-score analysis for the v2 LLM run.

Evidence set B, part 2 (zero cost: deterministic local replay, no API calls).

Joins ``audit_log/decisions.jsonl`` to
``data/processed/splits/enriched_test_2024_2025.csv`` (read-only) and reports:

(a) Timing: monthly LLM reduce fire rate vs monthly true reduce rate across
    the 23 test months (Pearson and Spearman correlation; n is small, so
    these are descriptive only).
(b) Agreement score: correlation between the aggregator's verbalised
    agreement_score and decision correctness (predicted == true label);
    accuracy in the top vs bottom halves of the agreement distribution
    (both an exact half split on a stable sort and a median-threshold split).
(c) Validation of the verbalised agreement score against a deterministic
    pairwise-disagreement measure: mean pairwise total-variation distance
    (0.5 * L1) across the 6 pairs of the four specialist probability vectors
    (disclosure, macro, price, fundamentals). Spearman rho between the
    verbalised agreement and this deterministic disagreement.
(d) Count of decisions where the disclosure agent returned exactly the
    uniform fallback vector (increase, hold, reduce) = (0.34, 0.33, 0.33).

Outputs:
  outputs/fundamentals_robustness/timing_agreement.csv          (metrics)
  outputs/fundamentals_robustness/timing_agreement_monthly.csv  (monthly series)
Run:
  python3 analysis/timing_and_agreement.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DECISIONS_JSONL = REPO / "audit_log" / "decisions.jsonl"
TEST_CSV = REPO / "data" / "processed" / "splits" / "enriched_test_2024_2025.csv"
OUT_CSV = REPO / "outputs" / "fundamentals_robustness" / "timing_agreement.csv"
OUT_MONTHLY_CSV = (
    REPO / "outputs" / "fundamentals_robustness" / "timing_agreement_monthly.csv"
)

SPECIALISTS = ["disclosure", "macro", "price", "fundamentals"]
CLASSES = ["increase", "hold", "reduce"]
UNIFORM_FALLBACK = (0.34, 0.33, 0.33)


def _argmax_label(probs: dict[str, float]) -> str:
    """Same tie-break rule as llm/postprocess.py (dict order: increase, hold, reduce)."""
    mapping = {c: probs[c] for c in CLASSES}
    return max(mapping, key=mapping.get)  # type: ignore[arg-type]


def _pairwise_disagreement(vectors: list[np.ndarray]) -> float:
    """Mean pairwise total-variation distance (0.5 * L1) over all 6 pairs."""
    dists = [0.5 * float(np.abs(a - b).sum()) for a, b in combinations(vectors, 2)]
    return float(np.mean(dists))


def load_joined() -> pd.DataFrame:
    rows = []
    with DECISIONS_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            spec_vecs = [
                np.array(
                    [r["agent_outputs"][a]["probabilities"][c] for c in CLASSES]
                )
                for a in SPECIALISTS
            ]
            disc = r["agent_outputs"]["disclosure"]["probabilities"]
            rows.append(
                {
                    "date": r["decision_date_t"],
                    "ticker": r["ticker"],
                    "predicted_label": _argmax_label(r["final_probabilities"]),
                    "agreement_score": float(
                        r["agent_outputs"]["aggregator"]["agreement_score"]
                    ),
                    "det_disagreement": _pairwise_disagreement(spec_vecs),
                    "disclosure_uniform_fallback": int(
                        (disc["increase"], disc["hold"], disc["reduce"])
                        == UNIFORM_FALLBACK
                    ),
                }
            )
    dec = pd.DataFrame(rows)
    panel = pd.read_csv(TEST_CSV, usecols=["ticker", "date", "sector", "label"])
    df = dec.merge(panel, on=["date", "ticker"], how="inner", validate="1:1")
    if len(df) != len(dec):
        raise RuntimeError(f"Join lost rows: {len(dec)} -> {len(df)}")
    df["correct"] = (df["predicted_label"] == df["label"]).astype(int)
    df["fired"] = (df["predicted_label"] == "reduce").astype(int)
    df["is_reduce"] = (df["label"] == "reduce").astype(int)
    return df


def timing_block(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    monthly = (
        df.groupby("date", sort=True)
        .agg(
            n_obs=("ticker", "size"),
            n_llm_reduce_fired=("fired", "sum"),
            n_true_reduce=("is_reduce", "sum"),
        )
        .reset_index()
    )
    monthly["llm_fire_rate"] = monthly["n_llm_reduce_fired"] / monthly["n_obs"]
    monthly["true_reduce_rate"] = monthly["n_true_reduce"] / monthly["n_obs"]

    pearson_r, pearson_p = stats.pearsonr(
        monthly["llm_fire_rate"], monthly["true_reduce_rate"]
    )
    rho, rho_p = stats.spearmanr(
        monthly["llm_fire_rate"], monthly["true_reduce_rate"]
    )
    metrics = [
        ("timing_n_months", float(len(monthly))),
        ("timing_pearson_r_fire_vs_true", float(pearson_r)),
        ("timing_pearson_p", float(pearson_p)),
        ("timing_spearman_rho_fire_vs_true", float(rho)),
        ("timing_spearman_p", float(rho_p)),
        ("timing_mean_monthly_fire_rate", float(monthly["llm_fire_rate"].mean())),
        ("timing_mean_monthly_true_rate", float(monthly["true_reduce_rate"].mean())),
    ]
    return monthly, metrics


def agreement_block(df: pd.DataFrame) -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []

    # (b) agreement vs correctness
    pearson_r, pearson_p = stats.pearsonr(df["agreement_score"], df["correct"])
    rho, rho_p = stats.spearmanr(df["agreement_score"], df["correct"])
    metrics += [
        ("agree_n_decisions", float(len(df))),
        ("agree_pearson_r_agreement_vs_correct", float(pearson_r)),
        ("agree_pearson_p", float(pearson_p)),
        ("agree_spearman_rho_agreement_vs_correct", float(rho)),
        ("agree_spearman_p", float(rho_p)),
        ("agree_overall_accuracy", float(df["correct"].mean())),
    ]

    # Exact half split on a stable sort (ties broken deterministically by
    # date then ticker so the split is reproducible).
    s = df.sort_values(
        ["agreement_score", "date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    half = len(s) // 2
    bottom, top = s.iloc[:half], s.iloc[len(s) - half :]
    metrics += [
        ("agree_halfsplit_n_per_half", float(half)),
        ("agree_halfsplit_bottom_accuracy", float(bottom["correct"].mean())),
        ("agree_halfsplit_top_accuracy", float(top["correct"].mean())),
        (
            "agree_halfsplit_top_minus_bottom",
            float(top["correct"].mean() - bottom["correct"].mean()),
        ),
    ]

    # Median-threshold split (ties at the median go to the bottom group).
    med = float(df["agreement_score"].median())
    lo = df[df["agreement_score"] <= med]
    hi = df[df["agreement_score"] > med]
    metrics += [
        ("agree_median_agreement_score", med),
        ("agree_medsplit_n_le_median", float(len(lo))),
        ("agree_medsplit_n_gt_median", float(len(hi))),
        ("agree_medsplit_accuracy_le_median", float(lo["correct"].mean())),
        ("agree_medsplit_accuracy_gt_median", float(hi["correct"].mean())),
        (
            "agree_medsplit_gt_minus_le",
            float(hi["correct"].mean() - lo["correct"].mean()),
        ),
    ]

    # (c) verbalised agreement vs deterministic pairwise disagreement
    rho_d, rho_d_p = stats.spearmanr(df["agreement_score"], df["det_disagreement"])
    metrics += [
        ("valid_spearman_rho_agreement_vs_det_disagreement", float(rho_d)),
        ("valid_spearman_p", float(rho_d_p)),
        ("valid_mean_det_disagreement", float(df["det_disagreement"].mean())),
        ("valid_mean_agreement_score", float(df["agreement_score"].mean())),
    ]

    # (d) disclosure uniform fallback count
    metrics += [
        (
            "disclosure_uniform_fallback_count",
            float(df["disclosure_uniform_fallback"].sum()),
        ),
        (
            "disclosure_uniform_fallback_share",
            float(df["disclosure_uniform_fallback"].mean()),
        ),
    ]
    return metrics


def main() -> None:
    df = load_joined()
    monthly, timing_metrics = timing_block(df)
    agree_metrics = agreement_block(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        timing_metrics + agree_metrics, columns=["metric", "value"]
    ).to_csv(OUT_CSV, index=False)
    monthly.to_csv(OUT_MONTHLY_CSV, index=False)

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MONTHLY_CSV}")
    for m, v in timing_metrics + agree_metrics:
        print(f"{m}: {v}")


if __name__ == "__main__":
    main()
