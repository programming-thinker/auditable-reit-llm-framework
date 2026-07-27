"""Market-adjusted label robustness check (no LLM calls, pure local compute).

Thesis labels are cut on RAW total return (future_ret_1m, +/-2%), and the common
market factor explains >50% of return variance (see systematic_vs_idiosyncratic.csv),
so 'downside is systematic' is partly baked into the label itself. This script tests
whether ANY firm-level signal appears once the common factor is removed:

  1. Relabel every firm-month on its EXCESS return over the leave-one-out (LOO)
     equal-weight REIT index: idx_i = (sum(future_ret_1m) - own) / (n - 1);
     excess_i = future_ret_1m - idx_i;  excess > +2% -> increase, < -2% -> reduce,
     else hold. Months with < 10 names are dropped (none in practice; reported).
  2. Refit the tuned multinomial logistic EXACTLY as analysis/tuned_robustness.py
     (elastic-net saga, class_weight='balanced', TimeSeriesSplit(4), same C/l1 grids,
     macro-F1 selection) on relabeled train, evaluate on relabeled test. Feature sets:
     Original 13+sector and Extended +9 fundamentals (incl NAV proxy).
  3. Repeat the variance decomposition of analysis/systematic_vs_idiosyncratic.py on
     the NEW labels: pooled LPM R^2 of the reduce indicator on month dummies vs on the
     13 firm features; monthly reduce-rate SD vs the binomial-independence null; share
     of reduce events in high-reduce months (> 2x base rate).

Read-only on Zone 1; writes outputs/fundamentals_robustness/market_adjusted_labels.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / "data/processed/splits"
FUND = REPO / "outputs/fundamentals_robustness/firm_fundamentals_panel.csv"
NAV = REPO / "outputs/fundamentals_robustness/nav_proxy_panel.csv"
OUT = REPO / "outputs/fundamentals_robustness/market_adjusted_labels.csv"
LABELS = ["increase", "hold", "reduce"]
SEED = 42
BAND = 0.02          # +/-2% band, same as the raw-return labels
MIN_NAMES = 10       # months with fewer names than this are dropped

ORIG = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
        "dividend_yield_lag1", "FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1",
        "term_spread_10y_2y_lag1", "cpi_yoy_lag1", "UNRATE_lag1"]
FUND_FEATS = ["amihud_illiq", "idio_vol", "leverage", "debt_to_equity",
              "interest_cover", "book_to_market", "ln_mktcap", "ffo_yield_proxy",
              "navprem_book_adj"]

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def load(split: str) -> pd.DataFrame:
    """Same merge as analysis/tuned_robustness.py::load."""
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    fund = pd.read_csv(FUND)[["ticker", "date"] + [c for c in FUND_FEATS if c != "navprem_book_adj"]]
    nav = pd.read_csv(NAV)[["ticker", "date", "navprem_book_adj"]]
    return df.merge(fund, on=["ticker", "date"], how="left").merge(nav, on=["ticker", "date"], how="left")


def relabel(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Leave-one-out equal-weight index excess-return labels. Returns (df, n_dropped)."""
    g = df.groupby("date")["future_ret_1m"]
    n = g.transform("count")
    tot = g.transform("sum")
    keep = n >= MIN_NAMES
    n_dropped = int((~keep).sum())
    df = df.loc[keep].copy()
    loo_idx = (tot[keep] - df["future_ret_1m"]) / (n[keep] - 1)
    df["excess_ret_1m"] = df["future_ret_1m"] - loo_idx
    df["label"] = np.where(df["excess_ret_1m"] > BAND, "increase",
                           np.where(df["excess_ret_1m"] < -BAND, "reduce", "hold"))
    return df, n_dropped


def metrics(y: np.ndarray, pred: np.ndarray, tag: str) -> dict:
    out = {"model": tag}
    for lab in LABELS:
        tp = int(((pred == lab) & (y == lab)).sum())
        fp = int(((pred == lab) & (y != lab)).sum())
        fn = int(((pred != lab) & (y == lab)).sum())
        out[f"{lab}_recall"] = round(tp / (tp + fn), 3) if (tp + fn) else 0.0
        out[f"{lab}_prec"] = round(tp / (tp + fp), 3) if (tp + fp) else 0.0
    out["reduce_npred"] = int((pred == "reduce").sum())
    out["firing_rate"] = round(float((pred == "reduce").mean()), 3)
    out["random_at_budget_recall"] = out["firing_rate"]  # = fire-rate for a random ranker
    out["accuracy"] = round(float((pred == y).mean()), 3)
    return out


def make_pipe(numeric: list[str], C: float, l1: float) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    lr = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1, C=C,
                            class_weight="balanced", multi_class="multinomial",
                            max_iter=4000, tol=1e-3, random_state=SEED)
    return Pipeline([("pre", pre), ("lr", lr)])


def tune(train: pd.DataFrame, numeric: list[str]) -> tuple[tuple[float, float], float]:
    """Time-series CV grid search; select by mean macro-F1 (same as tuned_robustness.py)."""
    tr = train.sort_values("date").reset_index(drop=True)
    X, y = tr[numeric + ["sector"]], tr["label"].values
    tscv = TimeSeriesSplit(n_splits=4)
    best, best_score = None, -1.0
    for C in C_GRID:
        for l1 in L1_GRID:
            scores = []
            for tri, vai in tscv.split(X):
                pipe = make_pipe(numeric, C, l1)
                pipe.fit(X.iloc[tri], y[tri])
                pred = pipe.predict(X.iloc[vai])
                scores.append(f1_score(y[vai], pred, average="macro", labels=LABELS, zero_division=0))
            m = float(np.mean(scores))
            if m > best_score:
                best_score, best = m, (C, l1)
    return best, best_score


def variance_decomposition(df: pd.DataFrame) -> dict:
    """Adapted from analysis/systematic_vs_idiosyncratic.py, on market-adjusted labels
    (all splits pooled, 2015-2025) with the 13 ORIG firm/macro features."""
    res: dict = {}
    df = df.copy()
    df["is_red"] = (df["label"] == "reduce").astype(int)

    # -- reduce clustering vs i.i.d. binomial null --
    by_month = df.groupby("date")
    reduce_rate = by_month["is_red"].mean()
    n_per_month = by_month.size()
    p = float(df["is_red"].mean())
    obs_std = float(reduce_rate.std(ddof=0))
    null_std = float(np.sqrt(p * (1 - p) / n_per_month.mean()))
    res["base_reduce_rate"] = round(p, 3)
    res["obs_monthly_reduce_rate_std"] = round(obs_std, 3)
    res["iid_null_std"] = round(null_std, 3)
    res["dispersion_ratio_obs_over_null"] = round(obs_std / null_std, 2)
    hi_months = reduce_rate[reduce_rate > 2 * p].index
    n_red = max(int(df["is_red"].sum()), 1)
    share_hi = df[(df["is_red"] == 1) & (df["date"].isin(hi_months))].shape[0] / n_red
    res["pct_reduce_in_high_reduce_months"] = round(share_hi, 3)
    res["n_high_reduce_months"] = int(len(hi_months))
    res["n_total_months"] = int(reduce_rate.shape[0])

    # -- explanatory power: MONTH FE vs the 13 FIRM/MACRO FEATURES (pooled LPM R^2) --
    mon = pd.get_dummies(df["date"])
    lin_month = LinearRegression().fit(mon.values, df["is_red"].values)
    res["month_FE_R2_on_reduce"] = round(float(lin_month.score(mon.values, df["is_red"].values)), 3)
    sub = df.dropna(subset=ORIG)
    Xf = StandardScaler().fit_transform(sub[ORIG].values)
    lin_firm = LinearRegression().fit(Xf, sub["is_red"].values)
    res["firm_features_R2_on_reduce"] = round(float(lin_firm.score(Xf, sub["is_red"].values)), 3)
    res["firm_features_n_obs"] = int(sub.shape[0])
    return res


def main() -> None:
    np.random.seed(SEED)
    train_raw = load("enriched_train_2015_2021")
    val_raw = load("enriched_validation_2022_2023")
    test_raw = load("enriched_test_2024_2025")

    rows: list[dict] = []

    # ---- 1. relabel on LOO excess return ----
    train, drop_tr = relabel(train_raw)
    val, drop_va = relabel(val_raw)
    test, drop_te = relabel(test_raw)
    print(f"[relabel] rows dropped (month n<{MIN_NAMES}): train={drop_tr}, "
          f"validation={drop_va}, test={drop_te}")
    for split_name, d, dropped in [("train", train, drop_tr), ("validation", val, drop_va),
                                   ("test", test, drop_te)]:
        counts = d["label"].value_counts()
        shares = d["label"].value_counts(normalize=True)
        for lab in LABELS:
            rows.append({"section": "label_distribution", "item": split_name,
                         "metric": f"{lab}_n", "value": int(counts.get(lab, 0))})
            rows.append({"section": "label_distribution", "item": split_name,
                         "metric": f"{lab}_share", "value": round(float(shares.get(lab, 0.0)), 3)})
        rows.append({"section": "label_distribution", "item": split_name,
                     "metric": "n_total", "value": int(d.shape[0])})
        rows.append({"section": "label_distribution", "item": split_name,
                     "metric": "rows_dropped_min_names", "value": dropped})
        print(f"[labels] {split_name}: n={d.shape[0]}  "
              + "  ".join(f"{lab}={shares.get(lab, 0.0):.3f}" for lab in LABELS))

    # ---- 2. tuned multinomial logistic, exactly as tuned_robustness.py ----
    for c in FUND_FEATS:
        lo, hi = train[c].quantile([0.01, 0.99])
        train[c] = train[c].clip(lo, hi)
        test[c] = test[c].clip(lo, hi)

    feature_sets = {"Original 13+sector": ORIG,
                    "Extended +9 fundamentals (incl NAV)": ORIG + FUND_FEATS}
    for name, num in feature_sets.items():
        (C, l1), cv = tune(train, num)
        pipe = make_pipe(num, C, l1).fit(train[num + ["sector"]], train["label"])
        pred = pipe.predict(test[num + ["sector"]])
        r = metrics(test["label"].values, pred, name)
        r["best_C"], r["best_l1"], r["cv_macroF1"] = C, l1, round(cv, 3)
        print(f"[tuned] {name}: best C={C}, l1_ratio={l1}, cv_macroF1={cv:.3f}")
        print(f"        reduce_recall={r['reduce_recall']}, reduce_prec={r['reduce_prec']}, "
              f"reduce_npred={r['reduce_npred']}, firing_rate={r['firing_rate']}, "
              f"hold_recall={r['hold_recall']}, accuracy={r['accuracy']}")
        for k, v in r.items():
            if k == "model":
                continue
            rows.append({"section": "tuned_logit_market_adjusted", "item": name,
                         "metric": k, "value": v})

    # ---- 3. variance decomposition on the NEW labels (all splits pooled) ----
    pooled = pd.concat([train, val, test], ignore_index=True)
    vd = variance_decomposition(pooled)
    print("\n[variance decomposition on market-adjusted labels, 2015-2025 pooled]")
    for k, v in vd.items():
        rows.append({"section": "variance_decomposition_market_adjusted",
                     "item": "pooled_2015_2025", "metric": k, "value": v})
        print(f"  {k:38} {v}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nwritten -> {OUT}")

    print("\n[reading]")
    print("  Raw-label reference (tuned_robustness.csv): reduce_recall=0.0 both sets;")
    print("  raw-label month FE R2=0.433 vs firm features R2=0.007 "
          "(systematic_vs_idiosyncratic.csv).")
    print("  If market-adjusted reduce recall stays ~0 AND the month-FE R2 collapses")
    print("  toward the firm-feature R2, the null is not an artifact of the raw-return")
    print("  label: no firm-level signal appears even after removing the common factor.")


if __name__ == "__main__":
    main()
