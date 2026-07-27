"""Sector decomposition of the v2 LLM run's reduce-call performance.

Evidence set B, part 1 (zero cost: deterministic local replay, no API calls).

Joins ``audit_log/decisions.jsonl`` (v2 five-agent DeepSeek test run,
575 decisions = 25 REITs x 23 months, 2024-01 .. 2025-11) to
``data/processed/splits/enriched_test_2024_2025.csv`` (read-only Zone 1;
provides ``sector`` and true ``label``).

Per sector:
  * n decisions, LLM reduce fire rate, true reduce rate
  * within-sector recall and precision for the reduce class

Decomposition of overall reduce recall into
  * between-sector ALLOCATION skill: counterfactual where the model keeps
    its per-sector reduce budgets (number of reduce calls per sector) but
    allocates them uniformly at random *within* each sector
    (seeded Monte Carlo, 1000 draws; analytic expectation also reported), and
  * within-sector SELECTION skill: actual recall minus the allocation
    counterfactual.
  A fully random baseline (same overall budget, allocated uniformly across
  all 575 observations) anchors the decomposition:
     recall_actual = recall_random + allocation_effect + selection_effect.

Output: outputs/fundamentals_robustness/sector_decomposition.csv
Run:    python3 analysis/sector_decomposition.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DECISIONS_JSONL = REPO / "audit_log" / "decisions.jsonl"
TEST_CSV = REPO / "data" / "processed" / "splits" / "enriched_test_2024_2025.csv"
OUT_CSV = REPO / "outputs" / "fundamentals_robustness" / "sector_decomposition.csv"

MC_SEED = 42
MC_DRAWS = 1000


def _argmax_label(probs: dict[str, float]) -> str:
    """Same tie-break rule as llm/postprocess.py (dict order: increase, hold, reduce)."""
    mapping = {
        "increase": probs["increase"],
        "hold": probs["hold"],
        "reduce": probs["reduce"],
    }
    return max(mapping, key=mapping.get)  # type: ignore[arg-type]


def load_joined() -> pd.DataFrame:
    """Join v2 decisions to the enriched test panel on (date, ticker)."""
    rows = []
    with DECISIONS_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "date": r["decision_date_t"],
                    "ticker": r["ticker"],
                    "predicted_label": _argmax_label(r["final_probabilities"]),
                }
            )
    dec = pd.DataFrame(rows)

    panel = pd.read_csv(TEST_CSV, usecols=["ticker", "date", "sector", "label"])
    df = dec.merge(panel, on=["date", "ticker"], how="inner", validate="1:1")
    if len(df) != len(dec):
        raise RuntimeError(
            f"Join lost rows: {len(dec)} decisions -> {len(df)} joined"
        )
    df["fired"] = (df["predicted_label"] == "reduce").astype(int)
    df["is_reduce"] = (df["label"] == "reduce").astype(int)
    return df


def per_sector_table(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for sector, g in df.groupby("sector", sort=True):
        n = len(g)
        fired = int(g["fired"].sum())
        true_red = int(g["is_reduce"].sum())
        tp = int((g["fired"] & g["is_reduce"]).sum())
        recs.append(
            {
                "row_type": "sector",
                "sector": sector,
                "n_obs": n,
                "n_llm_reduce_fired": fired,
                "llm_reduce_fire_rate": fired / n,
                "n_true_reduce": true_red,
                "true_reduce_rate": true_red / n,
                "n_true_positive": tp,
                "within_sector_recall": (tp / true_red) if true_red else np.nan,
                "within_sector_precision": (tp / fired) if fired else np.nan,
            }
        )
    return pd.DataFrame(recs)


def decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Allocation-vs-selection decomposition of overall reduce recall."""
    rng = np.random.default_rng(MC_SEED)

    n_total = len(df)
    m_total = int(df["is_reduce"].sum())  # total true reduces
    k_total = int(df["fired"].sum())  # total reduce calls (budget)
    tp_total = int((df["fired"] & df["is_reduce"]).sum())

    recall_actual = tp_total / m_total
    # Fully random baseline: K picks among N, E[TP] = K*M/N -> recall = K/N.
    recall_random_overall = k_total / n_total

    # Analytic allocation counterfactual: keep per-sector budgets k_s, pick
    # uniformly within sector -> E[TP] = sum_s k_s * m_s / n_s.
    exp_tp_alloc = 0.0
    for _, g in df.groupby("sector", sort=True):
        n_s = len(g)
        k_s = int(g["fired"].sum())
        m_s = int(g["is_reduce"].sum())
        exp_tp_alloc += k_s * m_s / n_s
    recall_alloc_analytic = exp_tp_alloc / m_total

    # Seeded Monte Carlo of the same counterfactual (1000 draws).
    draws = np.empty(MC_DRAWS)
    sector_groups = [
        (int(g["fired"].sum()), g["is_reduce"].to_numpy())
        for _, g in df.groupby("sector", sort=True)
    ]
    for d in range(MC_DRAWS):
        tp = 0
        for k_s, is_red in sector_groups:
            if k_s == 0:
                continue
            picks = rng.choice(len(is_red), size=k_s, replace=False)
            tp += int(is_red[picks].sum())
        draws[d] = tp / m_total
    recall_alloc_mc_mean = float(draws.mean())
    recall_alloc_mc_std = float(draws.std(ddof=1))
    # Two-sided-ish exceedance: share of random-within-sector draws >= actual.
    p_mc_geq_actual = float((draws >= recall_actual).mean())

    allocation_effect = recall_alloc_analytic - recall_random_overall
    selection_effect = recall_actual - recall_alloc_analytic

    summary = [
        ("n_obs_total", n_total),
        ("n_true_reduce_total", m_total),
        ("n_llm_reduce_fired_total", k_total),
        ("n_true_positive_total", tp_total),
        ("recall_actual", recall_actual),
        ("precision_actual", tp_total / k_total if k_total else np.nan),
        ("recall_random_overall_budget", recall_random_overall),
        ("recall_alloc_counterfactual_analytic", recall_alloc_analytic),
        ("recall_alloc_counterfactual_mc_mean", recall_alloc_mc_mean),
        ("recall_alloc_counterfactual_mc_std", recall_alloc_mc_std),
        ("mc_share_draws_geq_actual_recall", p_mc_geq_actual),
        ("allocation_effect_between_sector", allocation_effect),
        ("selection_effect_within_sector", selection_effect),
        ("mc_seed", MC_SEED),
        ("mc_draws", MC_DRAWS),
    ]
    return pd.DataFrame(
        [
            {"row_type": "decomposition", "sector": metric, "value": val}
            for metric, val in summary
        ]
    )


def main() -> None:
    df = load_joined()
    sector_tbl = per_sector_table(df)
    decomp_tbl = decomposition(df)
    out = pd.concat([sector_tbl, decomp_tbl], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(sector_tbl.to_string(index=False))
    print(decomp_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
