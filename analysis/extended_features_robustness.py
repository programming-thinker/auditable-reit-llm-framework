"""THE crux robustness test for the thesis.

Question: is the structured baseline's 0% reduce-recall a FEATURE gap (we simply
lacked REIT fundamentals) or an INFORMATION gap (the predictive signal for discrete
downside is not in ANY backward-looking structured data, only in forward-looking
text)?

We re-fit the SAME multinomial logistic (class_weight='balanced') on the SAME
train/test split, but augment the original 13 features + sector with the 8 free
firm-level fundamentals built in build_fundamentals.py (FFO proxy, leverage,
interest coverage, book-to-market, size, Amihud illiquidity, idiosyncratic vol,
debt/equity) -- exactly the variables the REIT literature (Chan-Erickson-Wang 2003,
Campbell et al. 2008, Fama-French 1992, Amihud 2002, Ang et al. 2006) names as
first-order and that the 46->90 'enrichment' never added.

If reduce recall STILL ~ 0 with proper fundamentals -> strong evidence it is an
INFORMATION gap, which is the thesis's central claim and motivates the LLM text layer.

Read-only on Zone 1. Replicates the modelling logic of src/11_quant_only_model.py
(does NOT import/modify it). Writes to outputs/fundamentals_robustness/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / "data/processed/splits"
FUND = REPO / "outputs/fundamentals_robustness/firm_fundamentals_panel.csv"
OUT_DIR = REPO / "outputs/fundamentals_robustness"
LABELS = ["increase", "hold", "reduce"]

ORIG_NUMERIC = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
                "dividend_yield_lag1", "FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1",
                "term_spread_10y_2y_lag1", "cpi_yoy_lag1", "UNRATE_lag1"]
NEW_FUND = ["amihud_illiq", "idio_vol", "leverage", "debt_to_equity",
            "interest_cover", "book_to_market", "ln_mktcap", "ffo_yield_proxy"]


def load(split: str, fund: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.merge(fund, on=["ticker", "date"], how="left", suffixes=("", "_f"))


def winsorize(df: pd.DataFrame, cols: list[str], train: pd.DataFrame) -> None:
    """Clip to train 1/99 percentiles (fit on train, applied in place)."""
    for c in cols:
        lo, hi = train[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(lo, hi)


def fit_eval(train, test, numeric, label):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    clf = Pipeline([("pre", pre),
                    ("lr", LogisticRegression(class_weight="balanced", max_iter=5000,
                                              C=1.0, multi_class="multinomial"))])
    clf.fit(train[numeric + ["sector"]], train["label"])
    pred = clf.predict(test[numeric + ["sector"]])
    y = test["label"].values
    rep = {}
    for lab in LABELS:
        tp = int(((pred == lab) & (y == lab)).sum())
        fp = int(((pred == lab) & (y != lab)).sum())
        fn = int(((pred != lab) & (y == lab)).sum())
        rep[lab] = {
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "n_pred": int((pred == lab).sum()),
        }
    rep["accuracy"] = float((pred == y).mean())
    rep["_label"] = label
    return rep


def main() -> None:
    fund = pd.read_csv(FUND)[["ticker", "date"] + NEW_FUND]
    train = load("enriched_train_2015_2021", fund)
    test = load("enriched_test_2024_2025", fund)

    winsorize(test, NEW_FUND, train)
    winsorize(train, NEW_FUND, train)

    print(f"train={len(train)} test={len(test)}  "
          f"fundamentals coverage on test: "
          f"{test[NEW_FUND].notna().mean().mean():.1%}\n")

    orig = fit_eval(train, test, ORIG_NUMERIC, "Original 13+sector")
    ext = fit_eval(train, test, ORIG_NUMERIC + NEW_FUND, "Extended +8 fundamentals")
    firmonly = fit_eval(train, test,
                        ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized",
                         "drawdown", "dividend_yield_lag1"] + NEW_FUND,
                        "Firm-level only (7+8, no dead macro)")

    rows = []
    print(f"{'model':<34}{'reduce_recall':>14}{'reduce_prec':>13}"
          f"{'reduce_npred':>14}{'hold_recall':>13}{'acc':>8}")
    for r in [orig, ext, firmonly]:
        print(f"{r['_label']:<34}{r['reduce']['recall']:>14.3f}"
              f"{r['reduce']['precision']:>13.3f}{r['reduce']['n_pred']:>14d}"
              f"{r['hold']['recall']:>13.3f}{r['accuracy']:>8.3f}")
        rows.append({"model": r["_label"], "reduce_recall": r["reduce"]["recall"],
                     "reduce_precision": r["reduce"]["precision"],
                     "reduce_n_pred": r["reduce"]["n_pred"],
                     "hold_recall": r["hold"]["recall"], "accuracy": r["accuracy"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "extended_features_robustness.csv", index=False)

    verdict = ("INFORMATION gap confirmed: adding real REIT fundamentals does NOT "
               "rescue reduce recall (~0) -> signal is not in backward-looking "
               "structured data."
               if ext["reduce"]["recall"] < 0.05 else
               "Adding fundamentals MOVES reduce recall off 0 -> partly a FEATURE "
               "gap; re-examine the thesis claim.")
    print(f"\nVERDICT: {verdict}")
    (OUT_DIR / "extended_VERDICT.txt").write_text(verdict + "\n")


if __name__ == "__main__":
    main()
