"""Fama-MacBeth Newey-West robustness (reviewer critique: serial correlation in the
time-series of cross-sectional coefficients inflates classical FM t-statistics).

The thesis's headline Fama-MacBeth result (Table 5, §6.1.4) is the EXTENDED panel
(13 + 9 reconstructed fundamentals + sector), where `leverage`, `debt_to_equity` and
`ret_6m` are nominally significant — a "pricing, not prediction" finding. Classical FM
t-stats (t = mean_coef / (std_coef / sqrt(T))) assume the monthly slope series is i.i.d.
over time. If that series is autocorrelated, the standard fix is a HAC correction
(Newey & West 1987). This script reproduces the SAME per-month cross-sectional OLS as
analysis/extended_ols_fmb.py (which itself replicates src/11b WITHOUT importing Zone 1),
then reports, for every feature, both:

  * classical FM t-stat  (validated against extended_fama_macbeth.csv)
  * Newey-West HAC t-stat (Bartlett kernel, automatic lag floor(4*(T/100)^(2/9)))

for the full sample (T=118) and the test window (T=23). The point is whether the
nominally significant pricing terms survive serial-correlation-robust inference. HAC
generally *widens* SEs, so this can only weaken — never manufacture — significance.

Read-only on Zone 1. Writes outputs/llm_deepseek_test/fama_macbeth_nw.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.extended_ols_fmb import FUND_FEATS, ORIG, load  # noqa: E402

PUBLISHED = REPO / "outputs/fundamentals_robustness/extended_fama_macbeth.csv"
OUT = REPO / "outputs/llm_deepseek_test/fama_macbeth_nw.csv"
TARGET = "future_ret_1m"
ZERO = 1e-12
WATCH = ["leverage", "debt_to_equity", "ret_6m"]  # the Table 5 significant terms


def monthly_coefs(df: pd.DataFrame, feats: list[str], secs: list[str], period: str):
    """Reproduce extended_ols_fmb.fama_macbeth: per-month cross-sectional OLS -> T x K."""
    d = df.copy()
    med = {c: d[d["date"] <= "2021-12-31"][c].median() for c in FUND_FEATS}
    for c in FUND_FEATS:
        if c in feats:
            d[c] = d[c].fillna(med[c])
    fm_feats = [f for f in feats if "12m" not in f] + secs
    if period == "test":
        d = d[d["date"] >= "2024-01-01"]
    d = d.dropna(subset=[TARGET] + fm_feats)
    coefs = []
    for _, g in d.groupby("date"):
        if len(g) < 10:
            continue
        Xs = StandardScaler().fit_transform(g[fm_feats].astype(float).values)
        coefs.append(LinearRegression().fit(Xs, g[TARGET].values).coef_)
    return np.vstack(coefs), fm_feats


def nw_t(series: np.ndarray, maxlags: int) -> float:
    if np.std(series, ddof=1) < ZERO:
        return float("nan")
    res = sm.OLS(series, np.ones(len(series))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})
    return float(res.tvalues[0])


def analyse(df, secs, feats, label, period) -> pd.DataFrame:
    M, fm_feats = monthly_coefs(df, feats, secs, period)
    T = M.shape[0]
    maxlags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = mean / (std / np.sqrt(T))
    tnw = np.array([nw_t(M[:, k], maxlags) for k in range(M.shape[1])])
    out = pd.DataFrame({
        "model": label, "period": period, "n_months": T, "nw_maxlags": maxlags,
        "feature": fm_feats, "mean_coef": np.round(mean, 5),
        "t_classical": np.round(tc, 2), "t_newey_west": np.round(tnw, 2),
        "sig_classical": np.abs(tc) > 1.96, "sig_nw": np.abs(tnw) > 1.96,
    })
    return out


def main() -> None:
    df, secs = load()
    all_out = []
    for period in ("full", "test"):
        all_out.append(analyse(df, secs, ORIG + FUND_FEATS, "Extended", period))
    res = pd.concat(all_out, ignore_index=True)

    # ---- validate classical reproduction against the published CSV ----
    pub = pd.read_csv(PUBLISHED).rename(columns={"fm_t": "t_pub"})
    chk = res.merge(pub[["feature", "model", "period", "t_pub"]],
                    on=["feature", "model", "period"], how="left")
    real = chk[chk["mean_coef"].abs() > 1e-5].dropna(subset=["t_pub"])
    print(f"[validation] max |reproduced - published| classical t (Extended, real feats): "
          f"{(real['t_classical'] - real['t_pub']).abs().max():.3f}\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    for period in ("full", "test"):
        sub = res[res["period"] == period]
        T = sub["n_months"].iloc[0]
        L = sub["nw_maxlags"].iloc[0]
        sig_c = sub[sub["sig_classical"]]["feature"].tolist()
        sig_n = sub[sub["sig_nw"].fillna(False)]["feature"].tolist()
        print(f"=== Extended FM, {period} (T={T}, Newey-West maxlags={L}) ===")
        print(f"  significant @5% classical:   {sig_c or 'NONE'}")
        print(f"  significant @5% Newey-West:  {sig_n or 'NONE'}")
        for w in WATCH:
            r = sub[sub["feature"] == w]
            if len(r):
                r = r.iloc[0]
                print(f"    {w:<16} t_classical={r['t_classical']:+.2f}  "
                      f"t_NW={r['t_newey_west']:+.2f}  "
                      f"{'(survives HAC)' if abs(r['t_newey_west'])>1.96 else '(NOT sig under HAC)'}")
        print()

    print(f"[written] {OUT}")
    print("[reading] Where classical significance survives HAC, the pricing result is robust; "
          "where it does not, the i.i.d. assumption was flattering it. Either way the "
          "out-of-sample PREDICTION conclusion (Tables 4, 7) is unaffected: pricing != prediction.")


if __name__ == "__main__":
    main()
