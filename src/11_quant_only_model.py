import json
import math
import os
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "main_train_2015_2021.csv"
VAL_PATH = ROOT / "data" / "processed" / "splits" / "main_validation_2022_2023.csv"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "main_test_2024_2025.csv"
MACRO_PATH = ROOT / "data" / "processed" / "monthly_macro_signals.csv"
BASELINE_MONTHLY_GROSS_NET_PATH = ROOT / "outputs" / "tables" / "baseline_monthly_returns_gross_net.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREPROCESSING_AUDIT = OUT_TABLE_DIR / "quant_only_preprocessing_audit.csv"
OUT_CV_FOLD_METRICS = OUT_TABLE_DIR / "quant_only_cv_fold_metrics.csv"
OUT_CV_SEARCH = OUT_TABLE_DIR / "quant_only_hyperparameter_search_results_cv.csv"
OUT_CV_SELECTED = OUT_TABLE_DIR / "quant_only_selected_hyperparameters_cv.csv"
OUT_VALIDATION_METRICS = OUT_TABLE_DIR / "quant_only_validation_metrics.csv"
OUT_TEST_METRICS = OUT_TABLE_DIR / "quant_only_test_metrics.csv"
OUT_TEST_CONFUSION = OUT_TABLE_DIR / "quant_only_confusion_matrix_test.csv"
OUT_TEST_PREDICTIONS = OUT_TABLE_DIR / "quant_only_test_predictions.csv"
OUT_REDUCE_PROB_DIAGNOSTICS = OUT_TABLE_DIR / "quant_only_reduce_probability_diagnostics_test.csv"
OUT_STRAT_VAL_PERF = OUT_TABLE_DIR / "quant_only_strategy_validation_performance_gross_net.csv"
OUT_STRAT_TEST_PERF = OUT_TABLE_DIR / "quant_only_strategy_test_performance_gross_net.csv"
OUT_SELECTED_STRAT_TEST_PERF = OUT_TABLE_DIR / "quant_only_selected_strategy_test_performance_gross_net.csv"
OUT_MONTHLY_VAL = OUT_TABLE_DIR / "quant_only_monthly_returns_validation_gross_net.csv"
OUT_MONTHLY_TEST = OUT_TABLE_DIR / "quant_only_monthly_returns_test_gross_net.csv"

OUT_NAV_TEST = OUT_FIG_DIR / "quant_only_cumulative_nav_test_period_gross_net.png"
OUT_DD_TEST = OUT_FIG_DIR / "quant_only_drawdown_test_period_gross_net.png"
OUT_SELECTED_VS_BASELINES = OUT_FIG_DIR / "quant_only_selected_strategy_vs_baselines_test_gross_net.png"
OUT_REDUCE_PROB_FIG = OUT_FIG_DIR / "quant_only_p_reduce_distribution_test.png"

CLASS_ORDER = ["increase", "hold", "reduce"]

TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-11-30")

CV_FOLDS = [
    {
        "fold": 1,
        "train_start": "2015-01-01",
        "train_end": "2017-11-30",
        "purged_month": "2017-12",
        "cv_validation_start": "2018-01-01",
        "cv_validation_end": "2018-12-31",
    },
    {
        "fold": 2,
        "train_start": "2015-01-01",
        "train_end": "2018-11-30",
        "purged_month": "2018-12",
        "cv_validation_start": "2019-01-01",
        "cv_validation_end": "2019-12-31",
    },
    {
        "fold": 3,
        "train_start": "2015-01-01",
        "train_end": "2019-11-30",
        "purged_month": "2019-12",
        "cv_validation_start": "2020-01-01",
        "cv_validation_end": "2020-12-31",
    },
    {
        "fold": 4,
        "train_start": "2015-01-01",
        "train_end": "2020-11-30",
        "purged_month": "2020-12",
        "cv_validation_start": "2021-01-01",
        "cv_validation_end": "2021-12-31",
    },
]

# V6: Feature list is auto-detected from panel (supports both 46-col original and 90-col enriched)
# Baseline features (always present)
BASELINE_NUMERIC = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown",
    "FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1", "term_spread_10y_2y_lag1",
    "cpi_yoy_lag1", "UNRATE_lag1",
]

# Enriched features (only in 90-col panel, auto-detected)
# V6 Fix: Only include features with cross-sectional variation
# Most macro features (mortgage rates, credit spreads, housing) are identical
# across all REITs within each month → no discriminative power for classification
# Only dividend_yield_lag1 varies across REITs
ENRICHED_NUMERIC = [
    "dividend_yield_lag1",
]

# Runtime auto-detection (set by main() after loading data)
NUMERIC_FEATURES = BASELINE_NUMERIC  # default for imports
CATEGORICAL_FEATURES = ["sector"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL_COL = "label"
RET_COL = "future_ret_1m"
TC_RATE = 0.001

STRATEGY_HARD_INC = "Quant Hard Increase-Only"
STRATEGY_HARD_NOT_REDUCE = "Quant Hard Reduce-Avoidance"
STRATEGY_POSITIVE = "Quant Score Positive"
STRATEGY_TOP5 = "Quant Score Top-5"
STRATEGY_TOP10 = "Quant Score Top-10"
STRATEGY_TOP30 = "Quant Score Top-30pct"
STRATEGY_TOP10_DIVERSIFIED = "Quant Score Top-10 Diversified"
STRATEGY_TOP30_DIVERSIFIED = "Quant Score Top-30pct Diversified"


def require_columns(df, cols, source_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def hyperparameters_json(params):
    return json.dumps(params, sort_keys=True, default=str)


def load_macro_features():
    macro = pd.read_csv(MACRO_PATH, parse_dates=["date"]).sort_values("date").copy()
    lag_cols = [
        "FEDFUNDS_lag1",
        "DGS10_lag1",
        "DGS2_lag1",
        "term_spread_10y_2y_lag1",
        "cpi_yoy_lag1",
        "UNRATE_lag1",
    ]

    if "term_spread_10y_2y" not in macro.columns and {"DGS10", "DGS2"}.issubset(macro.columns):
        macro["term_spread_10y_2y"] = pd.to_numeric(macro["DGS10"], errors="coerce") - pd.to_numeric(
            macro["DGS2"], errors="coerce"
        )
    if "cpi_yoy" not in macro.columns and "CPIAUCSL" in macro.columns:
        macro["cpi_yoy"] = pd.to_numeric(macro["CPIAUCSL"], errors="coerce").pct_change(12)

    for col in lag_cols:
        if col not in macro.columns:
            base_col = col.replace("_lag1", "")
            if base_col not in macro.columns:
                raise ValueError(f"{MACRO_PATH.name} missing required columns for lag generation: {base_col}")
            macro[col] = pd.to_numeric(macro[base_col], errors="coerce").shift(1)

    if "monthly_rf" not in macro.columns:
        if "FEDFUNDS" not in macro.columns:
            raise ValueError("monthly_macro_signals.csv missing both monthly_rf and FEDFUNDS")
        macro["monthly_rf"] = pd.to_numeric(macro["FEDFUNDS"], errors="coerce") / 100.0 / 12.0

    require_columns(macro, ["date"] + lag_cols + ["monthly_rf"], MACRO_PATH.name)
    out = macro[["date"] + lag_cols + ["monthly_rf"]].drop_duplicates(subset=["date"]).sort_values("date").copy()
    out["monthly_rf"] = pd.to_numeric(out["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
    return out


def attach_macro_features(df, macro):
    """Attach macro features from monthly_macro_signals.csv (original features only)."""
    out = df.copy()
    macro_idx = macro.set_index("date")
    macro_cols = macro.columns.tolist()

    # Only attach columns that exist in macro DataFrame
    for col in NUMERIC_FEATURES:
        if col.endswith("_lag1") and col in macro_cols:
            mapped = out["date"].map(macro_idx[col])
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").combine_first(mapped)
            else:
                out[col] = mapped

    # Attach monthly_rf
    mapped_rf = out["date"].map(macro_idx["monthly_rf"])
    if "monthly_rf" in out.columns:
        out["monthly_rf"] = pd.to_numeric(out["monthly_rf"], errors="coerce").combine_first(mapped_rf)
    else:
        out["monthly_rf"] = mapped_rf
    return out


def make_preprocessor():
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def make_pipeline(estimator):
    return Pipeline(steps=[("preprocess", make_preprocessor()), ("model", estimator)])


def make_estimator(model_name, params):
    if model_name == "Multinomial Logistic Regression":
        kwargs = {
            "solver": "saga",
            "multi_class": "multinomial",
            "max_iter": 3000,
            "class_weight": "balanced",
            "random_state": 42,
            "C": params["C"],
            "penalty": params["penalty"],
            "n_jobs": -1,
        }
        if params["penalty"] == "elasticnet":
            kwargs["l1_ratio"] = params["l1_ratio"]
        return LogisticRegression(**kwargs)

    # V6: RF/GBM removed (see CLAUDE.md §1a). Logistic Regression only.
    raise ValueError(f"Unknown model: {model_name}")


def fit_model(pipe, X, y):
    # V6: Logistic Regression only (balanced class weights already in estimator)
    pipe.fit(X, y)
    return pipe


def evaluate_classification(y_true, y_pred):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
    for i, cls in enumerate(CLASS_ORDER):
        metrics[f"precision_{cls}"] = float(p[i])
        metrics[f"recall_{cls}"] = float(r[i])
    return metrics


def candidate_grid():
    # V6 Fix: Expanded C range (add weak regularization) + reduced L1 ratio
    logistic_params = []
    # L2 and L1
    for params in ParameterGrid({"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l2", "l1"]}):
        params["l1_ratio"] = None
        logistic_params.append(params)
    # Elastic-net with lower L1 ratio (was 0.15/0.5/0.85, now 0.1/0.3/0.5)
    for params in ParameterGrid({"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["elasticnet"], "l1_ratio": [0.1, 0.3, 0.5]}):
        logistic_params.append(params)

    candidates = []
    for model_name, params_list, prefix in [
        ("Multinomial Logistic Regression", logistic_params, "LR"),
    ]:
        for i, params in enumerate(params_list, start=1):
            clean_params = {k: (None if pd.isna(v) else v) for k, v in dict(params).items()}
            candidates.append(
                {
                    "model_name": model_name,
                    "candidate_id": f"{prefix}_{i:03d}",
                    "params": clean_params,
                }
            )
    return candidates


def slice_dates(df, start, end):
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()


def run_purged_walk_forward_cv(train_df):
    fold_rows = []
    search_rows = []

    candidates = candidate_grid()
    for candidate in candidates:
        model_name = candidate["model_name"]
        candidate_id = candidate["candidate_id"]
        params = candidate["params"]
        candidate_metrics = []

        for fold_cfg in CV_FOLDS:
            fold_train = slice_dates(train_df, fold_cfg["train_start"], fold_cfg["train_end"])
            fold_val = slice_dates(train_df, fold_cfg["cv_validation_start"], fold_cfg["cv_validation_end"])
            if fold_train.empty or fold_val.empty:
                raise ValueError(f"Empty CV fold for {candidate_id}, fold {fold_cfg['fold']}")

            X_train = fold_train[ALL_FEATURES].copy()
            y_train = fold_train[LABEL_COL].astype(str).copy()
            X_val = fold_val[ALL_FEATURES].copy()
            y_val = fold_val[LABEL_COL].astype(str).copy()

            pipe = make_pipeline(make_estimator(model_name, params))
            pipe = fit_model(pipe, X_train, y_train)
            y_pred = pipe.predict(X_val)
            metrics = evaluate_classification(y_val, y_pred)
            candidate_metrics.append(metrics)

            fold_rows.append(
                {
                    "model_name": model_name,
                    "candidate_id": candidate_id,
                    "fold": fold_cfg["fold"],
                    "train_start": fold_cfg["train_start"],
                    "train_end": fold_cfg["train_end"],
                    "purged_month": fold_cfg["purged_month"],
                    "cv_validation_start": fold_cfg["cv_validation_start"],
                    "cv_validation_end": fold_cfg["cv_validation_end"],
                    **metrics,
                }
            )

        metrics_df = pd.DataFrame(candidate_metrics)
        search_rows.append(
            {
                "model_name": model_name,
                "candidate_id": candidate_id,
                "hyperparameters_json": hyperparameters_json(params),
                "mean_cv_accuracy": float(metrics_df["accuracy"].mean()),
                "mean_cv_macro_f1": float(metrics_df["macro_f1"].mean()),
                "std_cv_macro_f1": float(metrics_df["macro_f1"].std(ddof=1)),
                "mean_cv_weighted_f1": float(metrics_df["weighted_f1"].mean()),
                "mean_cv_precision_increase": float(metrics_df["precision_increase"].mean()),
                "mean_cv_recall_increase": float(metrics_df["recall_increase"].mean()),
                "mean_cv_precision_hold": float(metrics_df["precision_hold"].mean()),
                "mean_cv_recall_hold": float(metrics_df["recall_hold"].mean()),
                "mean_cv_precision_reduce": float(metrics_df["precision_reduce"].mean()),
                "mean_cv_recall_reduce": float(metrics_df["recall_reduce"].mean()),
                "params": params,
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    search_results = pd.DataFrame(search_rows)
    search_results["selected_within_model"] = False

    selected_rows = []
    selected_by_model = {}
    for model_name, g in search_results.groupby("model_name", sort=False):
        ranked = g.sort_values(
            ["mean_cv_macro_f1", "mean_cv_recall_reduce", "mean_cv_weighted_f1", "std_cv_macro_f1"],
            ascending=[False, False, False, True],
        )
        selected = ranked.iloc[0].copy()
        selected_by_model[model_name] = selected.to_dict()
        search_results.loc[search_results["candidate_id"] == selected["candidate_id"], "selected_within_model"] = True
        selected_rows.append(
            {
                "model_name": model_name,
                "selected_candidate_id": selected["candidate_id"],
                "selected_hyperparameters_json": selected["hyperparameters_json"],
                "mean_cv_macro_f1": selected["mean_cv_macro_f1"],
                "std_cv_macro_f1": selected["std_cv_macro_f1"],
                "mean_cv_weighted_f1": selected["mean_cv_weighted_f1"],
                "mean_cv_reduce_recall": selected["mean_cv_recall_reduce"],
                "selection_rule": "Highest mean_cv_macro_f1; ties: mean_cv_reduce_recall, mean_cv_weighted_f1, lower std_cv_macro_f1.",
            }
        )

    cv_selected = pd.DataFrame(selected_rows)
    search_results_out = search_results.drop(columns=["params"])
    return fold_metrics, search_results_out, cv_selected, selected_by_model


def write_preprocessing_audit(train, val, test):
    rows = [
        {
            "preprocessing_step": "numeric_median_imputer",
            "fitted_on": "training period only: 2015-01 to 2021-12; fold training only during CV",
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "leakage_check_passed": True,
            "note": "SimpleImputer(strategy='median') is inside sklearn Pipeline/ColumnTransformer and is fit only on training data.",
        },
        {
            "preprocessing_step": "numeric_standard_scaler",
            "fitted_on": "training period only: 2015-01 to 2021-12; fold training only during CV",
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "leakage_check_passed": True,
            "note": "StandardScaler is inside sklearn Pipeline/ColumnTransformer and is fit only on training data.",
        },
        {
            "preprocessing_step": "sector_most_frequent_imputer",
            "fitted_on": "training period only: 2015-01 to 2021-12; fold training only during CV",
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "leakage_check_passed": True,
            "note": "Categorical imputer is fit only on training data.",
        },
        {
            "preprocessing_step": "sector_one_hot_encoder",
            "fitted_on": "training period only: 2015-01 to 2021-12; fold training only during CV",
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "leakage_check_passed": True,
            "note": "OneHotEncoder(handle_unknown='ignore') is fit only on training data.",
        },
        {
            "preprocessing_step": "validation_and_test_transform",
            "fitted_on": "not fitted on validation or test",
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "leakage_check_passed": True,
            "note": "Validation and test periods are transformed only by the training-fitted preprocessing pipeline.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_PREPROCESSING_AUDIT, index=False)


def enrich_predictions(df, model, features):
    out = df.copy()
    proba = model.predict_proba(features)
    classes = list(model.named_steps["model"].classes_)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for cls in CLASS_ORDER:
        out[f"pred_proba_{cls}"] = proba[:, class_to_idx[cls]] if cls in class_to_idx else 0.0
    out["pred_label"] = model.predict(features)
    out["score"] = out["pred_proba_increase"] - out["pred_proba_reduce"]
    return out


def _top_by_score(group, n):
    if len(group) == 0:
        return []
    return (
        group.sort_values(["score", "ticker"], ascending=[False, True])
        .head(min(n, len(group)))["ticker"]
        .tolist()
    )


def _top_pct(group, pct, min_holdings=1):
    if len(group) == 0:
        return []
    n = int(math.ceil(pct * len(group)))
    n = max(min_holdings, n)
    n = min(n, len(group))
    return _top_by_score(group, n)


STRATEGY_RULES = {
    STRATEGY_HARD_INC: lambda g: g.loc[g["pred_label"] == "increase", "ticker"].tolist(),
    STRATEGY_HARD_NOT_REDUCE: lambda g: g.loc[g["pred_label"] != "reduce", "ticker"].tolist(),
    STRATEGY_POSITIVE: lambda g: g.loc[g["score"] > 0, "ticker"].tolist(),
    STRATEGY_TOP5: lambda g: _top_by_score(g, 5),
    STRATEGY_TOP10: lambda g: _top_by_score(g, 10),
    STRATEGY_TOP30: lambda g: _top_pct(g, 0.30, min_holdings=1),
    STRATEGY_TOP10_DIVERSIFIED: lambda g: _top_by_score(g, 10),
    STRATEGY_TOP30_DIVERSIFIED: lambda g: _top_pct(g, 0.30, min_holdings=10),
}

STRATEGY_SELECTION_ELIGIBLE = [
    STRATEGY_HARD_INC,
    STRATEGY_HARD_NOT_REDUCE,
    STRATEGY_POSITIVE,
    STRATEGY_TOP10,
    STRATEGY_TOP30,
    STRATEGY_TOP10_DIVERSIFIED,
    STRATEGY_TOP30_DIVERSIFIED,
]


def finalize_series(df):
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["gross_return"] = out["monthly_return"]
    out["net_return"] = out["gross_return"] - TC_RATE * out["turnover"]
    out["excess_return_gross"] = out["gross_return"] - out["monthly_rf"]
    out["excess_return_net"] = out["net_return"] - out["monthly_rf"]
    out["nav_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["nav_net"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown_gross"] = out["nav_gross"] / out["nav_gross"].cummax() - 1.0
    out["drawdown_net"] = out["nav_net"] / out["nav_net"].cummax() - 1.0
    return out


def compute_perf_metrics(sdf):
    n_months = int(len(sdf))
    if n_months == 0:
        return {
            "final_nav_gross": np.nan,
            "final_nav_net": np.nan,
            "annualized_return_gross": np.nan,
            "annualized_return_net": np.nan,
            "annualized_excess_return_gross": np.nan,
            "annualized_excess_return_net": np.nan,
            "annualized_volatility_gross": np.nan,
            "annualized_volatility_net": np.nan,
            "excess_sharpe_gross": np.nan,
            "excess_sharpe_net": np.nan,
            "max_drawdown_gross": np.nan,
            "max_drawdown_net": np.nan,
            "n_months": 0,
            "avg_n_holdings": np.nan,
            "avg_monthly_turnover": np.nan,
        }

    final_nav_gross = float(sdf["nav_gross"].iloc[-1])
    final_nav_net = float(sdf["nav_net"].iloc[-1])
    annualized_return_gross = final_nav_gross ** (12.0 / n_months) - 1.0
    annualized_return_net = final_nav_net ** (12.0 / n_months) - 1.0
    annualized_excess_return_gross = float(sdf["excess_return_gross"].mean() * 12.0)
    annualized_excess_return_net = float(sdf["excess_return_net"].mean() * 12.0)
    vol_gross = sdf["gross_return"].std(ddof=1)
    vol_net = sdf["net_return"].std(ddof=1)
    annualized_volatility_gross = float(vol_gross * np.sqrt(12.0)) if pd.notna(vol_gross) else np.nan
    annualized_volatility_net = float(vol_net * np.sqrt(12.0)) if pd.notna(vol_net) else np.nan
    excess_sharpe_gross = (
        annualized_excess_return_gross / annualized_volatility_gross
        if pd.notna(annualized_volatility_gross) and annualized_volatility_gross > 0
        else np.nan
    )
    excess_sharpe_net = (
        annualized_excess_return_net / annualized_volatility_net
        if pd.notna(annualized_volatility_net) and annualized_volatility_net > 0
        else np.nan
    )
    return {
        "final_nav_gross": final_nav_gross,
        "final_nav_net": final_nav_net,
        "annualized_return_gross": annualized_return_gross,
        "annualized_return_net": annualized_return_net,
        "annualized_excess_return_gross": annualized_excess_return_gross,
        "annualized_excess_return_net": annualized_excess_return_net,
        "annualized_volatility_gross": annualized_volatility_gross,
        "annualized_volatility_net": annualized_volatility_net,
        "excess_sharpe_gross": excess_sharpe_gross,
        "excess_sharpe_net": excess_sharpe_net,
        "max_drawdown_gross": float(sdf["drawdown_gross"].min()),
        "max_drawdown_net": float(sdf["drawdown_net"].min()),
        "n_months": n_months,
        "avg_n_holdings": float(sdf["n_holdings"].mean()),
        "avg_monthly_turnover": float(sdf["turnover"].mean()),
    }


def backtest_strategies(pred_df, rf_df, strategy_group_label):
    all_tickers = sorted(pred_df["ticker"].unique().tolist())
    strategy_series = {}

    for strategy, selector in STRATEGY_RULES.items():
        prev_drifted_w = pd.Series(0.0, index=all_tickers, dtype=float)
        rows = []
        for dt, g in pred_df.groupby("date", sort=True):
            available = sorted(g["ticker"].unique().tolist())
            selected = sorted(set(selector(g)).intersection(available))
            w = pd.Series(0.0, index=all_tickers, dtype=float)
            if selected:
                w.loc[selected] = 1.0 / len(selected)

            returns = g.set_index("ticker")[RET_COL].reindex(all_tickers).fillna(0.0)
            monthly_return = float((w * returns).sum())
            turnover = float(0.5 * np.abs(w - prev_drifted_w).sum())
            n_holdings = int((w > 0).sum())
            rows.append(
                {
                    "date": dt,
                    "strategy": strategy,
                    "strategy_group": strategy_group_label,
                    "monthly_return": monthly_return,
                    "n_holdings": n_holdings,
                    "turnover": turnover,
                }
            )

            if n_holdings > 0 and monthly_return > -1.0:
                prev_drifted_w = (w * (1.0 + returns)) / (1.0 + monthly_return)
            else:
                prev_drifted_w = pd.Series(0.0, index=all_tickers, dtype=float)

        sdf = pd.DataFrame(rows).merge(rf_df, on="date", how="left")
        sdf["monthly_rf"] = pd.to_numeric(sdf["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
        strategy_series[strategy] = finalize_series(sdf)
    return strategy_series


def strategy_perf_table(strategy_dict):
    rows = []
    for strategy, sdf in strategy_dict.items():
        metrics = compute_perf_metrics(sdf)
        rows.append({"strategy": strategy, "strategy_group": sdf["strategy_group"].iloc[0], **metrics})
    out = pd.DataFrame(rows)
    cols = [
        "strategy",
        "strategy_group",
        "final_nav_gross",
        "final_nav_net",
        "annualized_return_gross",
        "annualized_return_net",
        "annualized_excess_return_gross",
        "annualized_excess_return_net",
        "annualized_volatility_gross",
        "annualized_volatility_net",
        "excess_sharpe_gross",
        "excess_sharpe_net",
        "max_drawdown_gross",
        "max_drawdown_net",
        "n_months",
        "avg_n_holdings",
        "avg_monthly_turnover",
    ]
    return out[cols].sort_values("strategy")


def load_baselines_for_period(start_date, end_date):
    if not BASELINE_MONTHLY_GROSS_NET_PATH.exists():
        return {}
    b = pd.read_csv(BASELINE_MONTHLY_GROSS_NET_PATH, parse_dates=["date"])
    required = ["date", "strategy", "strategy_group", "gross_return", "net_return", "monthly_rf", "n_holdings", "turnover"]
    require_columns(b, required, BASELINE_MONTHLY_GROSS_NET_PATH.name)
    keep = ["Benchmark SPY", "Benchmark VNQ", "Benchmark XLRE", "Equal-Weight REIT Portfolio"]
    b = b[
        (b["strategy_group"] == "implementable")
        & (b["strategy"].isin(keep))
        & (b["date"] >= start_date)
        & (b["date"] <= end_date)
    ].copy()
    out = {}
    for strategy, g in b.groupby("strategy", sort=False):
        core = g[["date", "strategy", "strategy_group", "gross_return", "net_return", "monthly_rf", "n_holdings", "turnover"]].copy()
        core["monthly_return"] = core["gross_return"]
        out[strategy] = finalize_series(core)
    return out


def plot_gross_net(series_dict, y_col_gross, y_col_net, title, y_label, out_path):
    plt.figure(figsize=(12, 7))
    for strategy, sdf in series_dict.items():
        plt.plot(sdf["date"], sdf[y_col_gross], label=f"{strategy} gross", linewidth=1.8)
        plt.plot(sdf["date"], sdf[y_col_net], label=f"{strategy} net", linewidth=1.4, linestyle="--")
    plt.title(title)
    plt.xlabel("Formation Date")
    plt.ylabel(y_label)
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_reduce_probability_diagnostics(test_pred_out):
    rows = []

    def summarize(frame, group_name):
        s = pd.to_numeric(frame["pred_proba_reduce"], errors="coerce")
        return {
            "group": group_name,
            "count": int(s.notna().sum()),
            "mean_p_reduce": float(s.mean()),
            "median_p_reduce": float(s.median()),
            "min_p_reduce": float(s.min()),
            "max_p_reduce": float(s.max()),
            "p_reduce_25pct": float(s.quantile(0.25)),
            "p_reduce_75pct": float(s.quantile(0.75)),
        }

    rows.append(summarize(test_pred_out, "all"))
    for label in CLASS_ORDER:
        rows.append(summarize(test_pred_out[test_pred_out["true_label"] == label], f"true_{label}"))

    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_REDUCE_PROB_DIAGNOSTICS, index=False)

    plt.figure(figsize=(10, 6))
    for label in CLASS_ORDER:
        subset = test_pred_out[test_pred_out["true_label"] == label]["pred_proba_reduce"]
        if not subset.empty:
            plt.hist(subset, bins=20, alpha=0.45, label=f"true {label}")
    plt.title("Test-Period Predicted Reduce Probability Distribution")
    plt.xlabel("P(reduce)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_REDUCE_PROB_FIG, dpi=180)
    plt.close()
    return diag


def main():
    global NUMERIC_FEATURES, ALL_FEATURES  # Allow runtime override

    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    val = pd.read_csv(VAL_PATH, parse_dates=["date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["date"])
    train = train[(train["date"] >= TRAIN_START) & (train["date"] <= TRAIN_END)].copy()
    val = val[(val["date"] >= VAL_START) & (val["date"] <= VAL_END)].copy()
    test = test[(test["date"] >= TEST_START) & (test["date"] <= TEST_END)].copy()

    # V6: Auto-detect enriched features
    available_cols = set(train.columns)
    detected_features = BASELINE_NUMERIC.copy()
    enriched_found = []
    for feat in ENRICHED_NUMERIC:
        if feat in available_cols:
            detected_features.append(feat)
            enriched_found.append(feat)

    NUMERIC_FEATURES = detected_features
    ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    print(f"\n{'='*70}")
    print(f"FEATURE AUTO-DETECTION")
    print(f"{'='*70}")
    print(f"Baseline features: {len(BASELINE_NUMERIC)}")
    print(f"Enriched features found: {len(enriched_found)}")
    print(f"Total numeric features: {len(NUMERIC_FEATURES)}")
    if enriched_found:
        print(f"Enriched: {', '.join(enriched_found[:5])}{'...' if len(enriched_found) > 5 else ''}")
    print(f"{'='*70}\n")

    macro = load_macro_features()
    train = attach_macro_features(train, macro)
    val = attach_macro_features(val, macro)
    test = attach_macro_features(test, macro)

    required = ALL_FEATURES + [LABEL_COL, "ticker", "date", RET_COL, "monthly_rf"]
    require_columns(train, required, TRAIN_PATH.name)
    require_columns(val, required, VAL_PATH.name)
    require_columns(test, required, TEST_PATH.name)

    train = train.dropna(subset=[LABEL_COL, RET_COL]).copy()
    val = val.dropna(subset=[LABEL_COL, RET_COL]).copy()
    test = test.dropna(subset=[LABEL_COL, RET_COL]).copy()
    write_preprocessing_audit(train, val, test)

    fold_metrics, search_results, cv_selected, selected_by_model = run_purged_walk_forward_cv(train)
    fold_metrics.to_csv(OUT_CV_FOLD_METRICS, index=False)
    search_results.to_csv(OUT_CV_SEARCH, index=False)
    cv_selected.to_csv(OUT_CV_SELECTED, index=False)

    X_train = train[ALL_FEATURES].copy()
    y_train = train[LABEL_COL].astype(str).copy()
    X_val = val[ALL_FEATURES].copy()
    y_val = val[LABEL_COL].astype(str).copy()
    X_test = test[ALL_FEATURES].copy()
    y_test = test[LABEL_COL].astype(str).copy()

    tuned_models = {}
    validation_rows = []
    for model_name, selected in selected_by_model.items():
        params = selected["params"]
        pipe = make_pipeline(make_estimator(model_name, params))
        pipe = fit_model(pipe, X_train, y_train)
        tuned_models[model_name] = pipe

        y_val_pred = pipe.predict(X_val)
        metrics = evaluate_classification(y_val, y_val_pred)
        validation_rows.append(
            {
                "model_name": model_name,
                "selected_hyperparameters_json": hyperparameters_json(params),
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_weighted_f1": metrics["weighted_f1"],
                "validation_precision_increase": metrics["precision_increase"],
                "validation_recall_increase": metrics["recall_increase"],
                "validation_precision_hold": metrics["precision_hold"],
                "validation_recall_hold": metrics["recall_hold"],
                "validation_precision_reduce": metrics["precision_reduce"],
                "validation_recall_reduce": metrics["recall_reduce"],
            }
        )

    validation_metrics = pd.DataFrame(validation_rows).sort_values(
        ["validation_macro_f1", "validation_weighted_f1", "validation_accuracy"],
        ascending=[False, False, False],
    )
    best_classifier_name = validation_metrics.iloc[0]["model_name"]
    validation_metrics["selected_best_model"] = validation_metrics["model_name"] == best_classifier_name
    validation_metrics.to_csv(OUT_VALIDATION_METRICS, index=False)
    final_model = tuned_models[best_classifier_name]

    val_pred_enriched = enrich_predictions(val.copy(), final_model, X_val)
    rf_val = val[["date", "monthly_rf"]].drop_duplicates(subset=["date"]).sort_values("date").copy()
    rf_test = test[["date", "monthly_rf"]].drop_duplicates(subset=["date"]).sort_values("date").copy()

    val_strategy_series = backtest_strategies(
        val_pred_enriched[["date", "ticker", RET_COL, "pred_label", "score"]].copy(),
        rf_val,
        "quant_only_validation",
    )
    val_strategy_perf = strategy_perf_table(val_strategy_series)
    val_strategy_perf.to_csv(OUT_STRAT_VAL_PERF, index=False)

    selection_pool = val_strategy_perf[val_strategy_perf["strategy"].isin(STRATEGY_SELECTION_ELIGIBLE)].copy()
    selection_pool = selection_pool.sort_values(
        ["excess_sharpe_net", "annualized_excess_return_net", "strategy"],
        ascending=[False, False, True],
    )
    selected_portfolio_rule = selection_pool.iloc[0]["strategy"]

    y_test_pred = final_model.predict(X_test)
    test_metrics_raw = evaluate_classification(y_test, y_test_pred)
    test_metrics = pd.DataFrame(
        [
            {
                "model_name": best_classifier_name,
                "selected_hyperparameters_json": validation_metrics.iloc[0]["selected_hyperparameters_json"],
                "test_accuracy": test_metrics_raw["accuracy"],
                "test_macro_f1": test_metrics_raw["macro_f1"],
                "test_weighted_f1": test_metrics_raw["weighted_f1"],
                "test_precision_increase": test_metrics_raw["precision_increase"],
                "test_recall_increase": test_metrics_raw["recall_increase"],
                "test_precision_hold": test_metrics_raw["precision_hold"],
                "test_recall_hold": test_metrics_raw["recall_hold"],
                "test_precision_reduce": test_metrics_raw["precision_reduce"],
                "test_recall_reduce": test_metrics_raw["recall_reduce"],
            }
        ]
    )
    test_metrics.to_csv(OUT_TEST_METRICS, index=False)

    cm = confusion_matrix(y_test, y_test_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_ORDER], columns=[f"pred_{c}" for c in CLASS_ORDER])
    cm_df.to_csv(OUT_TEST_CONFUSION)

    test_pred_enriched = enrich_predictions(test.copy(), final_model, X_test)
    test_pred_out = test_pred_enriched[
        [
            "date",
            "ticker",
            "sector",
            LABEL_COL,
            RET_COL,
            "pred_label",
            "pred_proba_increase",
            "pred_proba_hold",
            "pred_proba_reduce",
            "score",
        ]
    ].rename(columns={LABEL_COL: "true_label"})
    test_pred_out = test_pred_out.sort_values(["date", "ticker"]).reset_index(drop=True)
    test_pred_out.to_csv(OUT_TEST_PREDICTIONS, index=False)
    reduce_diag = write_reduce_probability_diagnostics(test_pred_out)

    test_strategy_series = backtest_strategies(
        test_pred_enriched[["date", "ticker", RET_COL, "pred_label", "score"]].copy(),
        rf_test,
        "quant_only_test",
    )
    test_strategy_perf_quant = strategy_perf_table(test_strategy_series)

    baseline_series = load_baselines_for_period(test["date"].min(), test["date"].max())
    baseline_perf = strategy_perf_table(baseline_series) if baseline_series else pd.DataFrame(columns=test_strategy_perf_quant.columns)
    test_strategy_perf_all = pd.concat([test_strategy_perf_quant, baseline_perf], ignore_index=True).sort_values(
        ["strategy_group", "strategy"]
    )
    test_strategy_perf_all.to_csv(OUT_STRAT_TEST_PERF, index=False)

    selected_rows = test_strategy_perf_all[test_strategy_perf_all["strategy"] == selected_portfolio_rule].copy()
    if not baseline_perf.empty:
        selected_rows = pd.concat([selected_rows, baseline_perf], ignore_index=True)
    selected_rows.to_csv(OUT_SELECTED_STRAT_TEST_PERF, index=False)

    monthly_val = pd.concat(val_strategy_series.values(), ignore_index=True)
    monthly_val = monthly_val[
        [
            "date",
            "strategy",
            "strategy_group",
            "gross_return",
            "net_return",
            "monthly_rf",
            "excess_return_gross",
            "excess_return_net",
            "nav_gross",
            "nav_net",
            "drawdown_gross",
            "drawdown_net",
            "n_holdings",
            "turnover",
        ]
    ].sort_values(["date", "strategy"])
    monthly_val.to_csv(OUT_MONTHLY_VAL, index=False)

    monthly_test_parts = list(test_strategy_series.values())
    if baseline_series:
        monthly_test_parts.extend(baseline_series.values())
    monthly_test = pd.concat(monthly_test_parts, ignore_index=True)
    monthly_test = monthly_test[
        [
            "date",
            "strategy",
            "strategy_group",
            "gross_return",
            "net_return",
            "monthly_rf",
            "excess_return_gross",
            "excess_return_net",
            "nav_gross",
            "nav_net",
            "drawdown_gross",
            "drawdown_net",
            "n_holdings",
            "turnover",
        ]
    ].sort_values(["date", "strategy"])
    monthly_test.to_csv(OUT_MONTHLY_TEST, index=False)

    plot_dict = {k: v for k, v in test_strategy_series.items()}
    if baseline_series:
        plot_dict.update(baseline_series)
    plot_gross_net(
        plot_dict,
        "nav_gross",
        "nav_net",
        "Quant-Only Strategies and Baselines: Cumulative NAV (Test Period, Gross vs Net)",
        "NAV",
        OUT_NAV_TEST,
    )
    plot_gross_net(
        plot_dict,
        "drawdown_gross",
        "drawdown_net",
        "Quant-Only Strategies and Baselines: Drawdown (Test Period, Gross vs Net)",
        "Drawdown",
        OUT_DD_TEST,
    )

    selected_plot = {selected_portfolio_rule: test_strategy_series[selected_portfolio_rule]}
    if baseline_series:
        selected_plot.update(baseline_series)
    plot_gross_net(
        selected_plot,
        "nav_gross",
        "nav_net",
        "Selected Tuned Quant Strategy vs Implementable Baselines (Test Period, Gross vs Net)",
        "NAV",
        OUT_SELECTED_VS_BASELINES,
    )

    print("\nSelected hyperparameters by CV:")
    print(cv_selected.to_string(index=False))
    print("\nValidation model metrics after CV tuning:")
    print(validation_metrics.to_string(index=False))
    print(f"\nSelected best classifier by validation macro F1: {best_classifier_name}")
    print("\nValidation strategy performance (gross/net):")
    print(val_strategy_perf.to_string(index=False))
    print(f"\nSelected best portfolio rule by validation net excess Sharpe: {selected_portfolio_rule}")
    print("\nTest metrics:")
    print(test_metrics.to_string(index=False))
    print("\nReduce probability diagnostics (test):")
    print(reduce_diag.to_string(index=False))
    print("\nTest strategy performance (gross/net):")
    print(test_strategy_perf_all.to_string(index=False))


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="V6 baseline (logistic regression)")
    _parser.add_argument("--panel", choices=["original", "enriched"], default="enriched",
                         help="original=46-col panel (V5), enriched=90-col panel (V6, default)")
    _cli_args = _parser.parse_args()
    if _cli_args.panel == "enriched":
        TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "enriched_train_2015_2021.csv"
        VAL_PATH = ROOT / "data" / "processed" / "splits" / "enriched_validation_2022_2023.csv"
        TEST_PATH = ROOT / "data" / "processed" / "splits" / "enriched_test_2024_2025.csv"
        print(f"[V6] Using ENRICHED panel (90 cols): {TRAIN_PATH.name}")
    else:
        print(f"[V5] Using ORIGINAL panel (46 cols): {TRAIN_PATH.name}")
    main()
