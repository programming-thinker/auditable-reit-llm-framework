"""Robustness for the OTHER two structured methods: OLS and Fama-MacBeth.

Parallels analysis/extended_features_robustness.py (which covered classification).
Question: does adding the 9 free firm-level fundamentals (incl. NAV proxy) rescue
EITHER continuous-return prediction (OLS R^2) OR cross-sectional pricing (Fama-MacBeth
significance)?

Replicates the methodology of src/11b_extended_baselines.py (StandardScaler + OLS;
monthly cross-sectional OLS with FM t-stats) WITHOUT importing/modifying it.
Read-only on Zone 1; writes outputs/fundamentals_robustness/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
FUND = REPO / "outputs/fundamentals_robustness/firm_fundamentals_panel.csv"
NAV = REPO / "outputs/fundamentals_robustness/nav_proxy_panel.csv"
OUT_DIR = REPO / "outputs/fundamentals_robustness"
TARGET = "future_ret_1m"

ORIG = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
        "dividend_yield_lag1", "FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1",
        "term_spread_10y_2y_lag1", "cpi_yoy_lag1", "UNRATE_lag1"]
FUND_FEATS = ["amihud_illiq", "idio_vol", "leverage", "debt_to_equity",
              "interest_cover", "book_to_market", "ln_mktcap", "ffo_yield_proxy",
              "navprem_book_adj"]


def load() -> pd.DataFrame:
    df = pd.read_csv(PANEL, parse_dates=["date"])
    df["dkey"] = df["date"].dt.strftime("%Y-%m-%d")
    fund = pd.read_csv(FUND)[["ticker", "date"] + [c for c in FUND_FEATS if c != "navprem_book_adj"]].rename(columns={"date": "dkey"})
    nav = pd.read_csv(NAV)[["ticker", "date", "navprem_book_adj"]].rename(columns={"date": "dkey"})
    df = df.merge(fund, on=["ticker", "dkey"], how="left").merge(nav, on=["ticker", "dkey"], how="left")
    # add sector dummies
    dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=True)
    return pd.concat([df, dummies], axis=1), list(dummies.columns)


def splits(df):
    tr = df[df["date"] <= "2021-12-31"]
    va = df[(df["date"] >= "2022-01-01") & (df["date"] <= "2023-12-31")]
    te = df[df["date"] >= "2024-01-01"]
    return tr, va, te


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def impute_fit(train, cols):
    return {c: train[c].median() for c in cols}


def ols_block(df, feats, secs, label):
    d = df.copy()
    med = impute_fit(d[d["date"] <= "2021-12-31"], FUND_FEATS)
    for c in FUND_FEATS:
        if c in feats:
            d[c] = d[c].fillna(med[c])
    cols = feats + secs
    d = d.dropna(subset=[TARGET] + cols)
    tr, va, te = splits(d)
    sc = StandardScaler().fit(tr[cols].astype(float))
    m = LinearRegression().fit(sc.transform(tr[cols].astype(float)), tr[TARGET])
    out = {}
    for nm, s in [("train", tr), ("val", va), ("test", te)]:
        out[nm] = round(r2(s[TARGET].values, m.predict(sc.transform(s[cols].astype(float)))), 4)
    out["model"] = label
    out["n_test"] = len(te)
    return out


def fama_macbeth(df, feats, secs, label, period):
    d = df.copy()
    med = impute_fit(d[d["date"] <= "2021-12-31"], FUND_FEATS)
    for c in FUND_FEATS:
        if c in feats:
            d[c] = d[c].fillna(med[c])
    fm_feats = [f for f in feats if "12m" not in f] + secs  # exclude ret_12m per src/11b
    if period == "test":
        d = d[d["date"] >= "2024-01-01"]
    d = d.dropna(subset=[TARGET] + fm_feats)
    coefs = []
    for _, g in d.groupby("date"):
        if len(g) < 10:
            continue
        X = g[fm_feats].astype(float).values
        y = g[TARGET].values
        Xs = StandardScaler().fit_transform(X)
        coefs.append(LinearRegression().fit(Xs, y).coef_)
    if not coefs:
        return pd.DataFrame()
    M = np.vstack(coefs)
    mean = M.mean(0)
    se = M.std(0, ddof=1) / np.sqrt(len(M))
    t = mean / se
    rep = pd.DataFrame({"feature": fm_feats, "mean_coef": mean.round(5),
                        "fm_t": t.round(2), "sig_5pct": np.abs(t) > 1.96})
    rep["model"] = label
    rep["period"] = period
    rep["n_months"] = len(M)
    return rep.sort_values("fm_t", key=np.abs, ascending=False)


def main() -> None:
    df, secs = load()

    print("=== OLS  (predict future_ret_1m, R^2) ===")
    ols_rows = [ols_block(df, ORIG, secs, "Original 13+sector"),
                ols_block(df, ORIG + FUND_FEATS, secs, "Extended +9 fundamentals")]
    ols = pd.DataFrame(ols_rows)[["model", "train", "val", "test", "n_test"]]
    ols.to_csv(OUT_DIR / "extended_ols.csv", index=False)
    print(ols.to_string(index=False))

    print("\n=== FAMA-MACBETH (significant features, |t|>1.96) ===")
    all_fm = []
    for feats, lbl in [(ORIG, "Original"), (ORIG + FUND_FEATS, "Extended")]:
        for period in ["full", "test"]:
            r = fama_macbeth(df, feats, secs, lbl, period)
            if not r.empty:
                all_fm.append(r)
                sig = r[r["sig_5pct"]]
                print(f"\n[{lbl} | {period}] {r['n_months'].iloc[0]} months | "
                      f"significant: {list(sig['feature']) or 'NONE'}")
                print(r.head(6)[["feature", "mean_coef", "fm_t", "sig_5pct"]].to_string(index=False))
    pd.concat(all_fm).to_csv(OUT_DIR / "extended_fama_macbeth.csv", index=False)

    print("\n[note] OLS test R^2 < 0 => worse than mean. FM significance = in-sample "
          "cross-sectional pricing, NOT out-of-sample predictability. Three distinct "
          "questions; fundamentals can be priced (FM) yet useless for OOS / reduce.")


if __name__ == "__main__":
    main()
