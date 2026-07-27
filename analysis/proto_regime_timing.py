"""CONSTRUCTIVE pivot prototype: the variance decomposition says REIT downside is
SYSTEMATIC (which-month, not which-firm). So the RIGHT prediction problem is a
time-series REGIME task, not a cross-sectional firm task. We test whether the
market-level monthly REIT downside is predictable from LAGGED MACRO variables --
including the 64 'dead' macro columns that had zero cross-sectional variation
(useless for classification) but DO vary over time (the right input here).

Targets (monthly, market-level):
  - mkt_fwd_ret  : equal-weight REIT index next-month return (the systematic component)
  - bad_month    : 1 if reduce-rate that month > 2x base rate

Features: lagged macro levels + rate CHANGES + market momentum/vol (all known at t).
Honest OOS: train <=2021, test 2022-2025 (matches splits). Compare to naive base rate.
Welch-Goyal (2008) warns macro return prediction is weak OOS -- we report it honestly.

No API. Writes outputs/fundamentals_robustness/regime_timing.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/fundamentals_robustness/regime_timing.csv"

# market/macro features that vary over TIME (lagged, known at t)
MACRO = ["FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1", "term_spread_10y_2y_lag1",
         "cpi_yoy_lag1", "UNRATE_lag1", "fedfunds_chg_1m_lag1", "fedfunds_chg_3m_lag1",
         "dgs10_chg_1m_lag1", "dgs10_chg_3m_lag1", "mortgage30_lag1",
         "baa_spread_10y_lag1", "baa_spread_chg_1m_lag1", "home_price_yoy_lag1",
         "indpro_yoy_lag1", "real_income_yoy_lag1"]


def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate firm-month panel to a market-level monthly time series."""
    g = df.groupby("date")
    m = pd.DataFrame({
        "mkt_fwd_ret": g["future_ret_1m"].mean(),               # target: systematic next-month return
        "reduce_rate": (df.assign(r=(df["label"] == "reduce")).groupby("date")["r"].mean()),
        "mkt_mom_1m": g["ret_1m"].mean(),                       # market momentum (known at t)
        "mkt_mom_3m": g["ret_3m"].mean(),
        "mkt_vol": g["vol_annualized"].mean(),
    })
    # macro features: constant across firms within a month -> take first
    for c in MACRO:
        if c in df.columns:
            m[c] = g[c].first()
    return m.dropna(subset=["mkt_fwd_ret"]).sort_index()


def main() -> None:
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["future_ret_1m", "label"])
    m = build_monthly(df)
    base = (m["reduce_rate"] > 2 * (df["label"] == "reduce").mean()).astype(int)
    m["bad_month"] = base.values

    feats = [c for c in MACRO if c in m.columns] + ["mkt_mom_1m", "mkt_mom_3m", "mkt_vol"]
    m = m.dropna(subset=feats)
    train = m[m.index <= "2021-12-31"]
    test = m[m.index >= "2022-01-01"]

    sc = StandardScaler().fit(train[feats])
    Xtr, Xte = sc.transform(train[feats]), sc.transform(test[feats])

    rows = []
    # --- regression: predict next-month market return ---
    reg = LinearRegression().fit(Xtr, train["mkt_fwd_ret"])
    pred = reg.predict(Xte)
    ss_res = np.sum((test["mkt_fwd_ret"] - pred) ** 2)
    ss_tot = np.sum((test["mkt_fwd_ret"] - train["mkt_fwd_ret"].mean()) ** 2)  # naive = train mean
    oos_r2 = 1 - ss_res / ss_tot
    in_r2 = reg.score(Xtr, train["mkt_fwd_ret"])
    rows.append({"task": "predict mkt next-month return", "metric": "R2",
                 "in_sample": round(in_r2, 3), "OOS": round(float(oos_r2), 3),
                 "naive": 0.0})

    # --- classification: bad month ---
    if train["bad_month"].nunique() > 1 and test["bad_month"].nunique() > 1:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, train["bad_month"])
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(test["bad_month"], proba)
        auc_in = roc_auc_score(train["bad_month"], clf.predict_proba(Xtr)[:, 1])
        rows.append({"task": "predict bad-month (reduce-rate>2x)", "metric": "AUC",
                     "in_sample": round(auc_in, 3), "OOS": round(float(auc), 3), "naive": 0.5})

    # --- univariate signal: corr of each feature with future market return ---
    corrs = {c: round(float(np.corrcoef(m[c], m["mkt_fwd_ret"])[0, 1]), 3) for c in feats}
    top = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:6]

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(f"REGIME-TIMING TEST  (train n={len(train)} months, test n={len(test)} months)\n")
    print(res.to_string(index=False))
    print(f"\nbad-month base rate: train={train['bad_month'].mean():.2f} test={test['bad_month'].mean():.2f}")
    print("\nTop |corr| of lagged feature with next-month market return:")
    for c, v in top:
        print(f"  {c:28} {v:+.3f}")

    reg_oos = rows[0]["OOS"]
    auc_oos = rows[1]["OOS"] if len(rows) > 1 else np.nan
    print("\nVERDICT:")
    if (reg_oos > 0.03) or (not np.isnan(auc_oos) and auc_oos > 0.60):
        print("  REGIME SIGNAL OOS -> the systematic/timing framing IS partly predictable")
        print("  -> constructive positive result: predict the REGIME, not the firm.")
    elif (rows[0]["in_sample"] > 0.10) or (len(rows) > 1 and rows[1]["in_sample"] > 0.65):
        print("  In-sample regime structure exists (macro relates to downside) but OOS weak")
        print("  (consistent with Welch-Goyal 2008) -> downside is systematic AND hard to time")
        print("  -> richer conclusion: manage via hedging/diversification, not selection or timing.")
    else:
        print("  No regime predictability either -> downside is systematic and essentially")
        print("  unpredictable at both firm and market level -> strongest efficient-markets read.")


if __name__ == "__main__":
    main()
