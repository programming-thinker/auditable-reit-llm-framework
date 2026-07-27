"""Rigorous validation of the bad-month regime signal (OOS AUC ~0.73 in the quick
prototype). We guard against overfit / luck / regime-shift with:

  1. PARSIMONY: 3 theory-chosen macro drivers only (Chan-Erickson-Wang 2003: rates;
     Lewellen 2015: avoid data-mining) -- not 16 features.
       - dgs10_chg_3m_lag1   : recent 10Y move (the direct REIT driver)
       - term_spread_10y_2y_lag1 : curve / recession signal
       - baa_spread_10y_lag1 : credit / risk-premium
  2. WALK-FORWARD: expanding-window one-step-ahead OOS (no look-ahead), pooled AUC.
  3. PERMUTATION TEST: shuffle labels, redo walk-forward -> null AUC distribution -> p.
  4. BOOTSTRAP CI on the OOS AUC.

Target: bad_month = 1 if that month's reduce-rate > 2x the full-sample base rate
(a systematic-downside crash month). No API. Writes regime_validation.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/fundamentals_robustness/regime_validation.csv"
FEATS = ["dgs10_chg_3m_lag1", "term_spread_10y_2y_lag1", "baa_spread_10y_lag1"]
INIT = 48          # initial training months
N_PERM = 1000
SEED = 20260626


def build_monthly(df):
    g = df.groupby("date")
    m = pd.DataFrame({"reduce_rate": df.assign(r=(df["label"] == "reduce")).groupby("date")["r"].mean()})
    for c in FEATS:
        m[c] = g[c].first()
    return m.dropna().sort_index()


def walk_forward_auc(m, y, rng=None, shuffle=False):
    """Expanding-window one-step-ahead; return pooled OOS AUC (or nan)."""
    yv = y.values.copy()
    if shuffle:
        yv = rng.permutation(yv)
    X = m[FEATS].values
    preds, trues = [], []
    for t in range(INIT, len(m)):
        ytr = yv[:t]
        if len(np.unique(ytr)) < 2:
            continue
        sc = StandardScaler().fit(X[:t])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        clf.fit(sc.transform(X[:t]), ytr)
        p = clf.predict_proba(sc.transform(X[t:t + 1]))[0, 1]
        preds.append(p); trues.append(yv[t])
    if len(set(trues)) < 2:
        return np.nan, 0
    return roc_auc_score(trues, preds), len(trues)


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["future_ret_1m", "label"])
    base = (df["label"] == "reduce").mean()
    m = build_monthly(df)
    y = (m["reduce_rate"] > 2 * base).astype(int)
    n_bad = int(y.sum())

    obs_auc, n_oos = walk_forward_auc(m, y)

    # permutation null
    null = []
    for _ in range(N_PERM):
        a, _ = walk_forward_auc(m, y, rng=rng, shuffle=True)
        if not np.isnan(a):
            null.append(a)
    null = np.array(null)
    pval = float((null >= obs_auc).mean())

    # bootstrap CI on OOS AUC (resample the OOS month set)
    # recompute pooled OOS preds once, then bootstrap those
    yv = y.values; X = m[FEATS].values
    preds, trues = [], []
    for t in range(INIT, len(m)):
        if len(np.unique(yv[:t])) < 2:
            continue
        sc = StandardScaler().fit(X[:t])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[:t]), yv[:t])
        preds.append(clf.predict_proba(sc.transform(X[t:t + 1]))[0, 1]); trues.append(yv[t])
    preds, trues = np.array(preds), np.array(trues)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(preds), len(preds))
        if len(set(trues[idx])) < 2:
            continue
        boot.append(roc_auc_score(trues[idx], preds[idx]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # full-sample economic signs
    sc = StandardScaler().fit(m[FEATS]); clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(m[FEATS]), y)
    coefs = dict(zip(FEATS, np.round(clf.coef_[0], 3)))

    res = {"n_months": len(m), "n_bad_months": n_bad, "n_oos_predictions": n_oos,
           "OOS_AUC": round(obs_auc, 3), "perm_null_mean_AUC": round(float(null.mean()), 3),
           "perm_p_value": round(pval, 4), "boot_CI_lo": round(ci[0], 3), "boot_CI_hi": round(ci[1], 3)}
    pd.DataFrame([res]).to_csv(OUT, index=False)

    print("=== REGIME SIGNAL VALIDATION (parsimonious, walk-forward) ===\n")
    for k, v in res.items():
        print(f"  {k:24} {v}")
    print(f"\n  economic coefficients (standardized, +→more likely bad month):")
    for k, v in coefs.items():
        print(f"    {k:28} {v:+.3f}")
    print("\n  VERDICT:", end=" ")
    if pval < 0.05 and ci[0] > 0.5:
        print("REAL signal -- OOS AUC significant vs permutation null AND bootstrap CI excludes 0.5.")
        print("  -> The bad-month regime is genuinely (if modestly) predictable from rates/credit.")
        print("  -> CONSTRUCTIVE positive result stands; write the regime chapter.")
    elif pval < 0.10:
        print("Borderline -- suggestive but not conclusive given small sample; report honestly with caveats.")
    else:
        print("Not significant vs permutation null -- the 0.73 was likely overfit/regime-shift luck;")
        print("  -> downside is systematic AND not reliably timable -> stronger efficient-markets conclusion.")


if __name__ == "__main__":
    main()
