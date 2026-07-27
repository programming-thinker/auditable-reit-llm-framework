"""Exposure audit for the 8-K recency bug in the REPORTED test run (no API).

Bug: llm/orchestrator.py built the disclosure inputs with
``eight_k_filings[:5]`` but EdgarClient.get_filings_in_window returns filings
ASCENDING by filing_date -- so whenever more than 5 8-Ks fell in the 6-month
window, the run silently kept the 5 OLDEST and dropped the NEWEST ones.
(Fixed to ``[-5:]`` on 2026-07-04 for future runs.)

This script deterministically replays the EdgarClient windows for the 575
test decisions in audit_log/decisions.jsonl (local metadata + filings/ only):
  - per decision, the number of 8-Ks in the 6-month window,
  - how many decisions had >5 (i.e. the newest 8-Ks were dropped) and the
    distribution of dropped counts,
  - reduce recall and grounding rate (grounding = the
    analysis/contamination_audit.py rule) for affected vs unaffected
    decisions.

Reads (read-only): audit_log/decisions.jsonl, data/interim/filing_metadata.csv,
filings/clean_text/, data/processed/splits/enriched_test_2024_2025.csv.
Writes: outputs/llm_deepseek_test/audit_8k_recency.csv (per decision)
        outputs/llm_deepseek_test/audit_8k_recency_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.contamination_audit import specificity  # noqa: E402  (grounding rule)
from llm.edgar_client import EdgarClient  # noqa: E402

DECISIONS = REPO / "audit_log/decisions.jsonl"
TEST_SPLIT = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
OUT_DIR = REPO / "outputs/llm_deepseek_test"
OUT_CSV = OUT_DIR / "audit_8k_recency.csv"
OUT_SUMMARY = OUT_DIR / "audit_8k_recency_summary.csv"
N_KEPT = 5  # the as-run cap


def load_decisions() -> pd.DataFrame:
    rows = []
    with DECISIONS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec["decision_date_t"].startswith(("2024", "2025")):
                continue  # test window only
            probs = rec["final_probabilities"]
            disc = rec["agent_outputs"]["disclosure"]
            spec = specificity(disc.get("facts_cited") or [], disc.get("rationale", ""))
            rows.append({
                "date": rec["decision_date_t"],
                "ticker": rec["ticker"],
                "pred": max(probs, key=probs.get),
                "grounded": bool(spec["grounded"]),
            })
    return pd.DataFrame(rows)


def group_metrics(g: pd.DataFrame) -> dict:
    red = g[g["true_label"] == "reduce"]
    pred_red = g[g["pred"] == "reduce"]
    return {
        "n": int(len(g)),
        "n_true_reduce": int(len(red)),
        "reduce_recall": round(float((red["pred"] == "reduce").mean()), 4) if len(red) else float("nan"),
        "grounding_rate": round(float(g["grounded"].mean()), 4) if len(g) else float("nan"),
        "n_pred_reduce": int(len(pred_red)),
        "pred_reduce_grounded_pct": round(float(pred_red["grounded"].mean()), 4) if len(pred_red) else float("nan"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_decisions()
    assert len(df) == 575, f"expected 575 test decisions, got {len(df)}"

    truth = pd.read_csv(TEST_SPLIT)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    df = df.merge(truth[["date", "ticker", "label"]].rename(columns={"label": "true_label"}),
                  on=["date", "ticker"], how="left", validate="one_to_one")
    assert df["true_label"].notna().all(), "unmatched decisions vs test split"

    # deterministic replay of the as-run EdgarClient 8-K windows
    ec = EdgarClient(
        filings_dir=REPO / "filings",
        metadata_path=REPO / "data/interim/filing_metadata.csv",
    )
    n_8k, dropped_dates = [], []
    for _, r in df.iterrows():
        start = str((pd.Timestamp(r["date"]) - pd.DateOffset(months=6)).date())
        filings = ec.get_filings_in_window(r["ticker"], "8-K", start, r["date"])
        n_8k.append(len(filings))
        # as-run kept filings[:5] -> anything beyond index 4 (the NEWEST) was dropped
        dropped_dates.append(";".join(f["filing_date"] for f in filings[N_KEPT:]))

    df["n_8k_in_window"] = n_8k
    df["n_kept_as_run"] = df["n_8k_in_window"].clip(upper=N_KEPT)
    df["n_dropped_newest"] = (df["n_8k_in_window"] - N_KEPT).clip(lower=0)
    df["affected"] = df["n_dropped_newest"] > 0
    df["dropped_filing_dates"] = dropped_dates
    df["correct"] = df["pred"] == df["true_label"]
    df.to_csv(OUT_CSV, index=False)

    aff = df[df["affected"]]
    unaff = df[~df["affected"]]
    summary_rows = [
        {"group": "affected (>5 8-Ks, newest dropped)", **group_metrics(aff)},
        {"group": "unaffected (<=5 8-Ks)", **group_metrics(unaff)},
        {"group": "all", **group_metrics(df)},
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    print("=== 8-K RECENCY BUG EXPOSURE AUDIT (as-run replay, n=575) ===")
    print(f"  decisions with >5 8-Ks in window (affected): {len(aff)} "
          f"({len(aff) / len(df):.1%})")
    print(f"  8-Ks in window: mean={df['n_8k_in_window'].mean():.2f} "
          f"median={df['n_8k_in_window'].median():.0f} max={df['n_8k_in_window'].max()}")
    dist = df.loc[df["affected"], "n_dropped_newest"].value_counts().sort_index()
    print("  distribution of dropped (newest) 8-K counts among affected:")
    for k, v in dist.items():
        print(f"    dropped {k}: {v} decisions")
    if len(aff):
        print(f"  dropped per affected decision: mean={aff['n_dropped_newest'].mean():.2f} "
              f"max={aff['n_dropped_newest'].max()}")
    print("\n  group comparison (reduce recall / grounding):")
    for row in summary_rows:
        print(f"    {row['group']:38} n={row['n']:3} true_reduce={row['n_true_reduce']:3} "
              f"reduce_recall={row['reduce_recall']} grounding_rate={row['grounding_rate']} "
              f"pred_reduce_grounded={row['pred_reduce_grounded_pct']}")
    print(f"\n  wrote {OUT_CSV.relative_to(REPO)} and {OUT_SUMMARY.relative_to(REPO)}")


if __name__ == "__main__":
    main()
