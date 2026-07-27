"""Properly TUNED robustness test (replaces the C=1.0 quick run).

(1) Tuned multinomial logistic (elastic-net, class_weight='balanced') with a
    time-series-CV grid search over C and l1_ratio, selected by macro-F1 on TRAIN
    (matching the selection metric of src/11_quant_only_model.py). Refit on full
    train, evaluated on the 2024-2025 test set.
(2) Ordered logit (proportional-odds, McCullagh 1980) on the same features, which
    respects the ordinal label order reduce < hold < increase that the multinomial
    model discards.

Feature sets compared: Original (13+sector) and Extended (13 + 9 free fundamentals
incl. NAV proxy + sector). Read-only on Zone 1; writes outputs/fundamentals_robustness/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / "data/processed/splits"
FUND = REPO / "outputs/fundamentals_robustness/firm_fundamentals_panel.csv"
NAV = REPO / "outputs/fundamentals_robustness/nav_proxy_panel.csv"
OUT_DIR = REPO / "outputs/fundamentals_robustness"
LABELS = ["increase", "hold", "reduce"]
ORDER = {"reduce": 0, "hold": 1, "increase": 2}

ORIG = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
        "dividend_yield_lag1", "FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1",
        "term_spread_10y_2y_lag1", "cpi_yoy_lag1", "UNRATE_lag1"]
FUND_FEATS = ["amihud_illiq", "idio_vol", "leverage", "debt_to_equity",
              "interest_cover", "book_to_market", "ln_mktcap", "ffo_yield_proxy",
              "navprem_book_adj"]

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    fund = pd.read_csv(FUND)[["ticker", "date"] + [c for c in FUND_FEATS if c != "navprem_book_adj"]]
    nav = pd.read_csv(NAV)[["ticker", "date", "navprem_book_adj"]]
    return df.merge(fund, on=["ticker", "date"], how="left").merge(nav, on=["ticker", "date"], how="left")


def metrics(y, pred, tag):
    out = {"model": tag}
    for lab in LABELS:
        tp = int(((pred == lab) & (y == lab)).sum())
        fp = int(((pred == lab) & (y != lab)).sum())
        fn = int(((pred != lab) & (y == lab)).sum())
        out[f"{lab}_recall"] = round(tp / (tp + fn), 3) if (tp + fn) else 0.0
        out[f"{lab}_prec"] = round(tp / (tp + fp), 3) if (tp + fp) else 0.0
    out["reduce_npred"] = int((pred == "reduce").sum())
    out["accuracy"] = round(float((pred == y).mean()), 3)
    return out


def make_pipe(numeric, C, l1):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    lr = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1, C=C,
                            class_weight="balanced", multi_class="multinomial",
                            max_iter=4000, tol=1e-3)
    return Pipeline([("pre", pre), ("lr", lr)])


def tune(train, numeric):
    """Time-series CV grid search; select by mean macro-F1."""
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


def ordered_logit(train, test, numeric):
    """Proportional-odds ordered logit (statsmodels). Unregularized MLE; no class
    weights exist for ordinal models, so this is the like-for-like ordinal analogue."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ["sector"]),
    ])
    Xtr = pre.fit_transform(train[numeric + ["sector"]])
    Xte = pre.transform(test[numeric + ["sector"]])
    Xtr = np.asarray(Xtr, dtype=float); Xte = np.asarray(Xte, dtype=float)
    ytr = train["label"].map(ORDER).astype(int).values
    mod = OrderedModel(ytr, Xtr, distr="logit")
    res = mod.fit(method="lbfgs", maxiter=2000, disp=False)
    probs = res.model.predict(res.params, exog=Xte)
    inv = {v: k for k, v in ORDER.items()}
    pred = np.array([inv[i] for i in probs.argmax(axis=1)])
    return pred


def main() -> None:
    train, test = load("enriched_train_2015_2021"), load("enriched_test_2024_2025")
    for c in FUND_FEATS:
        lo, hi = train[c].quantile([0.01, 0.99])
        train[c] = train[c].clip(lo, hi); test[c] = test[c].clip(lo, hi)

    feature_sets = {"Original 13+sector": ORIG,
                    "Extended +9 fundamentals (incl NAV)": ORIG + FUND_FEATS}
    rows = []
    for name, num in feature_sets.items():
        (C, l1), cv = tune(train, num)
        pipe = make_pipe(num, C, l1).fit(train[num + ["sector"]], train["label"])
        pred = pipe.predict(test[num + ["sector"]])
        r = metrics(test["label"].values, pred, f"TUNED logit | {name}")
        r["best_C"], r["best_l1"], r["cv_macroF1"] = C, l1, round(cv, 3)
        rows.append(r)
        print(f"[tuned] {name}: best C={C}, l1_ratio={l1}, cv_macroF1={cv:.3f}")

    # ordered logit on the extended set
    try:
        opred = ordered_logit(train, test, ORIG + FUND_FEATS)
        rows.append(metrics(test["label"].values, opred,
                            "ORDERED logit | Extended +9 fundamentals"))
        print("[ordered] fitted OK")
    except Exception as e:  # noqa: BLE001
        print(f"[ordered] failed: {type(e).__name__}: {e}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "tuned_robustness.csv", index=False)
    cols = ["model", "reduce_recall", "reduce_prec", "reduce_npred", "hold_recall",
            "accuracy"]
    print("\n" + res[cols].to_string(index=False))
    print("\nReference floors (same test, reduce base rate 28.7%):")
    print("  Random-at-budget reduce recall ~= fire-rate; precision ~= 0.287")


if __name__ == "__main__":
    main()
