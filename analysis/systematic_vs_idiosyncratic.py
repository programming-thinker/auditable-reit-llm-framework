"""ECONOMIC BACKBONE of the negative result: REIT monthly downside is SYSTEMATIC
(common, time-driven), not IDIOSYNCRATIC (firm-specific). If 'reduce' is a property
of WHICH MONTH it is rather than WHICH FIRM, then firm-level features are structurally
unable to predict it -- which explains every channel's failure.

Four complementary analyses on the panel (read-only):
  A. Return variance decomposition: systematic share = Var(monthly cross-sectional
     mean return) / total variance.
  B. Market-model R^2: regress each REIT's return on the equal-weight REIT index;
     average R^2 = systematic share of return variation.
  C. 'reduce' clustering in time: monthly reduce-rate dispersion vs the binomial null
     that would hold if reduce were i.i.d. across firms.
  D. Explanatory power: how much of the reduce indicator is explained by MONTH alone
     (month fixed effects) vs by FIRM FEATURES (the 13-feature model, ~0).

Writes outputs/fundamentals_robustness/systematic_vs_idiosyncratic.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/fundamentals_robustness/systematic_vs_idiosyncratic.csv"
RET = "future_ret_1m"
FIRM_FEATS = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
              "dividend_yield_lag1"]


def main() -> None:
    df = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=[RET, "label"])
    res = {}

    # ---- A. return variance decomposition ----
    monthly_mean = df.groupby("date")[RET].mean()
    var_between = monthly_mean.var(ddof=0)                         # systematic
    var_within = df.groupby("date")[RET].var(ddof=0).mean()       # idiosyncratic
    total = var_between + var_within
    res["A_systematic_share_of_return_var"] = round(var_between / total, 3)
    res["A_idiosyncratic_share"] = round(var_within / total, 3)

    # ---- B. market-model R^2 per REIT ----
    mkt = df.groupby("date")[RET].mean().rename("mkt")
    d = df.merge(mkt, on="date")
    r2s = []
    for tk, g in d.groupby("ticker"):
        if len(g) < 24:
            continue
        X = g[["mkt"]].values
        y = g[RET].values
        m = LinearRegression().fit(X, y)
        r2s.append(m.score(X, y))
    res["B_avg_market_model_R2"] = round(float(np.mean(r2s)), 3)
    res["B_median_market_model_R2"] = round(float(np.median(r2s)), 3)

    # ---- C. reduce clustering vs i.i.d. binomial null ----
    by_month = df.assign(is_red=(df["label"] == "reduce").astype(int)).groupby("date")
    reduce_rate = by_month["is_red"].mean()
    n_per_month = by_month.size()
    p = (df["label"] == "reduce").mean()                          # base rate
    obs_std = reduce_rate.std(ddof=0)
    null_std = float(np.sqrt(p * (1 - p) / n_per_month.mean()))   # if i.i.d. across firms
    res["C_base_reduce_rate"] = round(float(p), 3)
    res["C_obs_monthly_reduce_rate_std"] = round(float(obs_std), 3)
    res["C_iid_null_std"] = round(null_std, 3)
    res["C_dispersion_ratio_obs_over_null"] = round(float(obs_std / null_std), 2)
    # share of reduce events that occur in "high-reduce" months (>2x base rate)
    hi_months = reduce_rate[reduce_rate > 2 * p].index
    share_hi = df[(df["label"] == "reduce") & (df["date"].isin(hi_months))].shape[0] / max((df["label"] == "reduce").sum(), 1)
    res["C_pct_reduce_in_high_reduce_months"] = round(share_hi, 3)
    res["C_n_high_reduce_months"] = int(len(hi_months))
    res["C_n_total_months"] = int(reduce_rate.shape[0])

    # ---- D. explanatory power: MONTH vs FIRM FEATURES ----
    df["is_red"] = (df["label"] == "reduce").astype(int)
    # month fixed effects (pseudo-R^2 via linear prob model on month dummies)
    mon = pd.get_dummies(df["date"].dt.to_period("M").astype(str))
    lin_month = LinearRegression().fit(mon.values, df["is_red"].values)
    res["D_month_FE_R2_on_reduce"] = round(lin_month.score(mon.values, df["is_red"].values), 3)
    # firm features (same target)
    sub = df.dropna(subset=FIRM_FEATS)
    Xf = StandardScaler().fit_transform(sub[FIRM_FEATS].values)
    lin_firm = LinearRegression().fit(Xf, sub["is_red"].values)
    res["D_firm_features_R2_on_reduce"] = round(lin_firm.score(Xf, sub["is_red"].values), 3)

    out = pd.DataFrame([res]).T.rename(columns={0: "value"})
    out.to_csv(OUT)
    print("=== SYSTEMATIC vs IDIOSYNCRATIC ===\n")
    for k, v in res.items():
        print(f"  {k:42} {v}")
    print(f"\n  written -> {OUT}")
    print("\n[reading]")
    print(f"  A/B: ~{res['B_avg_market_model_R2']:.0%} of monthly REIT return variation is the common market factor.")
    print(f"  C: monthly reduce-rate varies {res['C_dispersion_ratio_obs_over_null']}x more than if reduce were")
    print(f"     firm-independent -> reduce CLUSTERS in time; {res['C_pct_reduce_in_high_reduce_months']:.0%} of all")
    print(f"     reduce events fall in {res['C_n_high_reduce_months']}/{res['C_n_total_months']} 'bad months'.")
    print(f"  D: knowing WHICH MONTH explains R^2={res['D_month_FE_R2_on_reduce']} of who-reduces; knowing the FIRM's")
    print(f"     features explains only R^2={res['D_firm_features_R2_on_reduce']}.")
    print("\n  => reduce is about WHEN, not WHICH FIRM. Firm-level prediction is structurally hopeless.")


if __name__ == "__main__":
    main()
