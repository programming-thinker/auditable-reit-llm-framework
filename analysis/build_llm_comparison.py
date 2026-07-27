"""Legitimate same-period comparison of LLM multi-agent vs structured baseline,
plus trivial baselines (random-at-budget, threshold-tuned logistic) and
REIT-block bootstrap CIs.

This script REPLACES the invalid cross-period 0% -> 26.2% headline. It compares
all methods on the EXACT SAME samples (the Test 2024-2025 rows that the LLM has
decided), so base-rate differences cannot confound the result.

Read-only w.r.t. Zone 1: it READS outputs/tables/quant_only_test_predictions.csv
and data/processed/splits/enriched_test_2024_2025.csv, and WRITES only to the new
outputs/llm_deepseek_test/ directory. No API calls.

Usage:
    python analysis/build_llm_comparison.py            # uses audit_log/decisions.jsonl
    python analysis/build_llm_comparison.py --jsonl <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASELINE_PRED = REPO / "outputs/tables/quant_only_test_predictions.csv"
TEST_SPLIT = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
DEFAULT_JSONL = REPO / "audit_log/decisions.jsonl"
OUT_DIR = REPO / "outputs/llm_deepseek_test"

LABELS = ["increase", "hold", "reduce"]
SEED = 20260626
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_llm_predictions(jsonl_path: Path) -> pd.DataFrame:
    """Parse decisions.jsonl -> df[date, ticker, llm_pred, p_inc, p_hold, p_red]."""
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec["final_probabilities"]
            pred = max(p, key=p.get)
            rows.append(
                {
                    "date": pd.to_datetime(rec["decision_date_t"]).strftime("%Y-%m-%d"),
                    "ticker": rec["ticker"],
                    "llm_pred": pred,
                    "llm_p_increase": p["increase"],
                    "llm_p_hold": p["hold"],
                    "llm_p_reduce": p["reduce"],
                }
            )
    df = pd.DataFrame(rows)
    # de-dup (diskcache replays can append duplicates); keep last
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last").reset_index(drop=True)
    return df


def load_baseline_and_truth() -> pd.DataFrame:
    """df[date, ticker, true_label, base_pred, base_p_reduce] on the Test set."""
    base = pd.read_csv(BASELINE_PRED)
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    base = base.rename(
        columns={"pred_label": "base_pred", "pred_proba_reduce": "base_p_reduce"}
    )
    # authoritative truth from the split (cross-check against baseline's own true_label)
    split = pd.read_csv(TEST_SPLIT)
    split["date"] = pd.to_datetime(split["date"]).dt.strftime("%Y-%m-%d")
    split = split[["date", "ticker", "label"]].rename(columns={"label": "true_label"})
    merged = base.merge(split, on=["date", "ticker"], how="inner", suffixes=("_base", ""))
    return merged[["date", "ticker", "true_label", "base_pred", "base_p_reduce"]]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for lab in LABELS:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        out[lab] = {"recall": recall, "precision": prec, "f1": f1, "support": tp + fn}
    out["accuracy"] = float((y_pred == y_true).mean())
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = pd.DataFrame(
        0, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]
    )
    for t in LABELS:
        for p in LABELS:
            cm.loc[f"true_{t}", f"pred_{p}"] = int(((y_true == t) & (y_pred == p)).sum())
    return cm


# --------------------------------------------------------------------------- #
# Trivial baselines
# --------------------------------------------------------------------------- #
def random_at_budget(y_true: np.ndarray, p_fire: float, rng: np.random.Generator,
                     n_draws: int = 2000) -> dict:
    """Predict 'reduce' on a random fraction p_fire of samples; rest 'hold'.
    Monte-Carlo expectation of reduce recall/precision."""
    n = len(y_true)
    k = int(round(p_fire * n))
    is_red = y_true == "reduce"
    recalls, precs = [], []
    for _ in range(n_draws):
        idx = rng.choice(n, size=k, replace=False)
        pred_red = np.zeros(n, dtype=bool)
        pred_red[idx] = True
        tp = int((pred_red & is_red).sum())
        recalls.append(tp / is_red.sum() if is_red.sum() else 0.0)
        precs.append(tp / k if k else 0.0)
    return {
        "reduce_recall": float(np.mean(recalls)),
        "reduce_recall_lo": float(np.percentile(recalls, 2.5)),
        "reduce_recall_hi": float(np.percentile(recalls, 97.5)),
        "reduce_precision": float(np.mean(precs)),
        "fire_rate": p_fire,
    }


def threshold_logistic(df: pd.DataFrame, p_fire: float) -> dict:
    """Flag the top p_fire fraction by baseline reduce-probability as 'reduce'.
    Demonstrates the argmax-0% is not a true floor."""
    n = len(df)
    k = int(round(p_fire * n))
    order = df["base_p_reduce"].rank(ascending=False, method="first")
    pred_red = order <= k
    is_red = df["true_label"].values == "reduce"
    tp = int((pred_red.values & is_red).sum())
    return {
        "reduce_recall": tp / is_red.sum() if is_red.sum() else 0.0,
        "reduce_precision": tp / k if k else 0.0,
        "fire_rate": p_fire,
        "threshold_k": k,
    }


# --------------------------------------------------------------------------- #
# Bootstrap (REIT-block) for reduce recall + LLM-vs-random difference
# --------------------------------------------------------------------------- #
def block_bootstrap(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample REIT tickers with replacement; recompute reduce recall per method
    and the LLM - random difference. Return summary with 95% CIs."""
    tickers = df["ticker"].unique()
    is_red_all = df["true_label"].values == "reduce"
    p_fire = float((df["llm_pred"] == "reduce").mean())

    def reduce_recall(sub: pd.DataFrame, pred_col: str) -> float:
        ir = sub["true_label"].values == "reduce"
        if ir.sum() == 0:
            return np.nan
        tp = int(((sub[pred_col].values == "reduce") & ir).sum())
        return tp / ir.sum()

    llm_r, base_r, rand_r, diff = [], [], [], []
    by_tkr = {t: df[df["ticker"] == t] for t in tickers}
    for _ in range(N_BOOT):
        samp = rng.choice(tickers, size=len(tickers), replace=True)
        sub = pd.concat([by_tkr[t] for t in samp], ignore_index=True)
        lr = reduce_recall(sub, "llm_pred")
        br = reduce_recall(sub, "base_pred")
        # random-at-budget on this resample (single draw, analytic-ish)
        ir = sub["true_label"].values == "reduce"
        n = len(sub)
        k = int(round(p_fire * n))
        idx = rng.choice(n, size=k, replace=False)
        rr = int(ir[idx].sum()) / ir.sum() if ir.sum() else np.nan
        llm_r.append(lr); base_r.append(br); rand_r.append(rr)
        if not (np.isnan(lr) or np.isnan(rr)):
            diff.append(lr - rr)

    def ci(arr):
        arr = np.array([a for a in arr if not np.isnan(a)])
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    rows = []
    for name, arr in [("LLM", llm_r), ("Logistic-argmax", base_r),
                      ("Random-at-budget", rand_r), ("LLM_minus_Random", diff)]:
        m, lo, hi = ci(arr)
        rows.append({"quantity": name, "reduce_recall_mean": m,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    llm = load_llm_predictions(args.jsonl)
    base = load_baseline_and_truth()
    df = llm.merge(base, on=["date", "ticker"], how="inner")
    n = len(df)
    partial = n < 575
    print(f"[info] joined samples: {n} (LLM={len(llm)}, baseline={len(base)}) "
          f"{'== PARTIAL DRY-RUN ==' if partial else '== FULL =='}")
    print(f"[info] true label dist:\n{df['true_label'].value_counts().to_string()}")

    y_true = df["true_label"].values
    p_fire = float((df["llm_pred"] == "reduce").mean())
    print(f"[info] LLM reduce fire-rate p = {p_fire:.3f}")

    # --- confusion matrices ---
    confusion(y_true, df["llm_pred"].values).to_csv(OUT_DIR / "llm_confusion.csv")
    confusion(y_true, df["base_pred"].values).to_csv(OUT_DIR / "baseline_confusion.csv")

    # --- per-method metrics ---
    llm_m = per_class_metrics(y_true, df["llm_pred"].values)
    base_m = per_class_metrics(y_true, df["base_pred"].values)
    rand_m = random_at_budget(y_true, p_fire, rng)
    thr_m = threshold_logistic(df, p_fire)

    headline = pd.DataFrame([
        {"method": "LLM-DeepSeek", "reduce_recall": llm_m["reduce"]["recall"],
         "reduce_precision": llm_m["reduce"]["precision"], "hold_recall": llm_m["hold"]["recall"],
         "accuracy": llm_m["accuracy"], "reduce_fire_rate": p_fire},
        {"method": "Logistic-argmax", "reduce_recall": base_m["reduce"]["recall"],
         "reduce_precision": base_m["reduce"]["precision"], "hold_recall": base_m["hold"]["recall"],
         "accuracy": base_m["accuracy"],
         "reduce_fire_rate": float((df["base_pred"] == "reduce").mean())},
        {"method": "Random-at-budget", "reduce_recall": rand_m["reduce_recall"],
         "reduce_precision": rand_m["reduce_precision"], "hold_recall": np.nan,
         "accuracy": np.nan, "reduce_fire_rate": p_fire},
        {"method": "Threshold-logistic", "reduce_recall": thr_m["reduce_recall"],
         "reduce_precision": thr_m["reduce_precision"], "hold_recall": np.nan,
         "accuracy": np.nan, "reduce_fire_rate": p_fire},
    ])
    headline.to_csv(OUT_DIR / "headline_comparison.csv", index=False)

    # --- full per-class classification report (LLM) ---
    rep = []
    for lab in LABELS:
        rep.append({"class": lab, **llm_m[lab]})
    pd.DataFrame(rep).to_csv(OUT_DIR / "llm_classification_report.csv", index=False)

    # --- bootstrap CIs ---
    boot = block_bootstrap(df, rng)
    boot.to_csv(OUT_DIR / "significance_bootstrap.csv", index=False)

    # --- verdict ---
    diff_row = boot[boot["quantity"] == "LLM_minus_Random"].iloc[0]
    verdict = ("LLM significantly > random (CI excludes 0)"
               if diff_row["ci_lo"] > 0 else
               "LLM NOT significantly > random (CI includes 0) -- rewrite claims")

    print("\n=== HEADLINE (same-period, same-samples) ===")
    print(headline.to_string(index=False))
    print("\n=== BOOTSTRAP reduce-recall (REIT-block, 95% CI) ===")
    print(boot.to_string(index=False))
    print(f"\n=== VERDICT: {verdict} ===")
    if partial:
        print("\n[warn] PARTIAL data -- numbers provisional until 575 complete.")

    (OUT_DIR / "VERDICT.txt").write_text(
        f"n={n} partial={partial}\nLLM reduce fire-rate={p_fire:.3f}\n"
        f"LLM-minus-Random reduce recall: mean={diff_row['reduce_recall_mean']:.3f} "
        f"CI=[{diff_row['ci_lo']:.3f},{diff_row['ci_hi']:.3f}]\nVERDICT: {verdict}\n"
    )


if __name__ == "__main__":
    main()
