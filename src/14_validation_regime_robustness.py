import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

PANEL_SLIM_PATH = ROOT / "data" / "processed" / "backtest_ready_panel_slim.csv"
PANEL_FULL_PATH = ROOT / "data" / "processed" / "backtest_ready_panel.csv"
PANEL_REIT_PATH = ROOT / "data" / "processed" / "reit_monthly_panel.csv"
MACRO_PATH = ROOT / "data" / "processed" / "monthly_macro_signals.csv"

OUT_TABLE_DIR = ROOT / "outputs" / "tables"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODEL_SELECTION = OUT_TABLE_DIR / "validation_regime_robustness_model_selection.csv"
OUT_STRATEGY_SELECTION = OUT_TABLE_DIR / "validation_regime_robustness_strategy_selection.csv"
OUT_TEST_PERF = OUT_TABLE_DIR / "validation_regime_robustness_test_performance.csv"
OUT_CONSISTENCY = OUT_TABLE_DIR / "quant_only_main_vs_robustness_consistency_check.csv"
OUT_HURDLE_COMPARISON = OUT_TABLE_DIR / "validation_regime_hurdle_comparison.csv"
MAIN_VALIDATION_METRICS = OUT_TABLE_DIR / "quant_only_validation_metrics.csv"
MAIN_STRATEGY_VALIDATION = OUT_TABLE_DIR / "quant_only_strategy_validation_performance_gross_net.csv"
MAIN_SELECTED_TEST = OUT_TABLE_DIR / "quant_only_selected_strategy_test_performance_gross_net.csv"
BASELINE_TEST = OUT_TABLE_DIR / "baseline_performance_table_implementable_test_period_gross_net.csv"

CLASS_ORDER = ["increase", "hold", "reduce"]
TC_RATE = 0.001

PRICE_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown"]
LAGGED_MACRO_FEATURES = [
    "FEDFUNDS_lag1",
    "DGS10_lag1",
    "DGS2_lag1",
    "term_spread_10y_2y_lag1",
    "cpi_yoy_lag1",
    "UNRATE_lag1",
]
CATEGORICAL_FEATURES = ["sector"]
ALL_FEATURES = PRICE_FEATURES + LAGGED_MACRO_FEATURES + CATEGORICAL_FEATURES

LABEL_COL = "label"
RET_COL = "future_ret_1m"

REGIMES = [
    {
        "regime": "main_2022_2023",
        "train_start": "2015-01-01",
        "train_end": "2021-12-31",
        "val_start": "2022-01-01",
        "val_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2025-11-30",
    },
    {
        "regime": "alt1_2021",
        "train_start": "2015-01-01",
        "train_end": "2020-12-31",
        "val_start": "2021-01-01",
        "val_end": "2021-12-31",
        "test_start": "2024-01-01",
        "test_end": "2025-11-30",
    },
    {
        "regime": "alt2_2019",
        "train_start": "2015-01-01",
        "train_end": "2018-12-31",
        "val_start": "2019-01-01",
        "val_end": "2019-12-31",
        "test_start": "2024-01-01",
        "test_end": "2025-11-30",
    },
]

STRATEGY_HARD_INC = "Quant Hard Increase-Only"
STRATEGY_HARD_NOT_REDUCE = "Quant Hard Reduce-Avoidance"
STRATEGY_TOP5 = "Quant Score Top-5"
STRATEGY_TOP10 = "Quant Score Top-10"
STRATEGY_TOP30 = "Quant Score Top-30pct"
STRATEGY_POSITIVE = "Quant Score Positive"
STRATEGY_TOP10_DIVERSIFIED = "Quant Score Top-10 Diversified"
STRATEGY_TOP30_DIVERSIFIED = "Quant Score Top-30pct Diversified"


def _top_by_score(group, n):
    if len(group) == 0:
        return []
    return (
        group.sort_values(["score", "ticker"], ascending=[False, True])
        .head(min(n, len(group)))["ticker"]
        .tolist()
    )


def _top_30pct(group):
    if len(group) == 0:
        return []
    n = max(1, int(math.ceil(0.30 * len(group))))
    return _top_by_score(group, n)


def _top_pct(group, pct, min_holdings=1):
    if len(group) == 0:
        return []
    n = max(min_holdings, int(math.ceil(pct * len(group))))
    return _top_by_score(group, min(n, len(group)))


STRATEGY_RULES = {
    STRATEGY_HARD_INC: lambda g: g.loc[g["pred_label"] == "increase", "ticker"].tolist(),
    STRATEGY_HARD_NOT_REDUCE: lambda g: g.loc[g["pred_label"] != "reduce", "ticker"].tolist(),
    STRATEGY_TOP5: lambda g: _top_by_score(g, 5),
    STRATEGY_TOP10: lambda g: _top_by_score(g, 10),
    STRATEGY_TOP30: lambda g: _top_30pct(g),
    STRATEGY_POSITIVE: lambda g: g.loc[g["score"] > 0, "ticker"].tolist(),
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


def require_columns(df, cols, source_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def pick_panel_path():
    if PANEL_SLIM_PATH.exists():
        return PANEL_SLIM_PATH
    if PANEL_FULL_PATH.exists():
        return PANEL_FULL_PATH
    if PANEL_REIT_PATH.exists():
        return PANEL_REIT_PATH
    raise FileNotFoundError("No eligible panel file found.")


def load_macro_with_lags_and_rf():
    macro = pd.read_csv(MACRO_PATH, parse_dates=["date"]).sort_values("date").copy()
    require_columns(macro, ["date"], MACRO_PATH.name)

    if "term_spread_10y_2y" not in macro.columns and {"DGS10", "DGS2"}.issubset(macro.columns):
        macro["term_spread_10y_2y"] = pd.to_numeric(macro["DGS10"], errors="coerce") - pd.to_numeric(
            macro["DGS2"], errors="coerce"
        )
    if "cpi_yoy" not in macro.columns and "CPIAUCSL" in macro.columns:
        macro["cpi_yoy"] = pd.to_numeric(macro["CPIAUCSL"], errors="coerce").pct_change(12)

    for col in LAGGED_MACRO_FEATURES:
        if col not in macro.columns:
            base = col.replace("_lag1", "")
            if base not in macro.columns:
                raise ValueError(f"{MACRO_PATH.name} missing required base macro column: {base}")
            macro[col] = pd.to_numeric(macro[base], errors="coerce").shift(1)

    if "monthly_rf" not in macro.columns:
        if "FEDFUNDS" in macro.columns:
            macro["monthly_rf"] = pd.to_numeric(macro["FEDFUNDS"], errors="coerce") / 100.0 / 12.0
        else:
            macro["monthly_rf"] = 0.0

    cols = ["date"] + LAGGED_MACRO_FEATURES + ["monthly_rf"]
    out = macro[cols].drop_duplicates(subset=["date"]).sort_values("date").copy()
    out["monthly_rf"] = pd.to_numeric(out["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
    return out


def attach_macro_features(panel, macro):
    out = panel.copy()
    macro_idx = macro.set_index("date")
    for col in LAGGED_MACRO_FEATURES + ["monthly_rf"]:
        mapped = out["date"].map(macro_idx[col])
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").combine_first(mapped)
        else:
            out[col] = mapped
    return out


def make_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("num", num_pipe, PRICE_FEATURES + LAGGED_MACRO_FEATURES), ("cat", cat_pipe, CATEGORICAL_FEATURES)]
    )


def make_pipeline(model):
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


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


def backtest_strategies(pred_df, rf_df):
    all_tickers = sorted(pred_df["ticker"].unique().tolist())
    out = {}

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
                    "monthly_return": monthly_return,
                    "turnover": turnover,
                    "n_holdings": n_holdings,
                }
            )
            if n_holdings > 0 and monthly_return > -1.0:
                prev_drifted_w = (w * (1.0 + returns)) / (1.0 + monthly_return)
            else:
                prev_drifted_w = pd.Series(0.0, index=all_tickers, dtype=float)

        sdf = pd.DataFrame(rows).merge(rf_df, on="date", how="left")
        sdf["monthly_rf"] = pd.to_numeric(sdf["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
        out[strategy] = finalize_series(sdf)
    return out


def compute_metrics(sdf):
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


def run_regime(df, regime_cfg):
    regime = regime_cfg["regime"]
    train_start = pd.Timestamp(regime_cfg["train_start"])
    train_end = pd.Timestamp(regime_cfg["train_end"])
    val_start = pd.Timestamp(regime_cfg["val_start"])
    val_end = pd.Timestamp(regime_cfg["val_end"])
    test_start = pd.Timestamp(regime_cfg["test_start"])
    test_end = pd.Timestamp(regime_cfg["test_end"])

    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    train = train.dropna(subset=[LABEL_COL, RET_COL]).copy()
    val = val.dropna(subset=[LABEL_COL, RET_COL]).copy()
    test = test.dropna(subset=[LABEL_COL, RET_COL]).copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(f"{regime}: one of train/validation/test sets is empty.")

    required = ALL_FEATURES + [LABEL_COL, "ticker", "date", RET_COL, "monthly_rf"]
    require_columns(train, required, f"{regime} train")
    require_columns(val, required, f"{regime} val")
    require_columns(test, required, f"{regime} test")

    X_train, y_train = train[ALL_FEATURES].copy(), train[LABEL_COL].astype(str).copy()
    X_val, y_val = val[ALL_FEATURES].copy(), val[LABEL_COL].astype(str).copy()
    X_test, y_test = test[ALL_FEATURES].copy(), test[LABEL_COL].astype(str).copy()

    rf_val = val[["date", "monthly_rf"]].drop_duplicates(subset=["date"]).sort_values("date").copy()
    rf_test = test[["date", "monthly_rf"]].drop_duplicates(subset=["date"]).sort_values("date").copy()
    rf_val["monthly_rf"] = pd.to_numeric(rf_val["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)
    rf_test["monthly_rf"] = pd.to_numeric(rf_test["monthly_rf"], errors="coerce").ffill().bfill().fillna(0.0)

    model_specs = {
        "Multinomial Logistic Regression": LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    }

    model_rows = []
    fitted = {}
    for model_name, estimator in model_specs.items():
        pipe = make_pipeline(estimator)
        if model_name == "Gradient Boosting Classifier":
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            pipe.fit(X_train, y_train, model__sample_weight=sw)
        else:
            pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_val)
        metrics = evaluate_classification(y_val, y_val_pred)
        model_rows.append(
            {
                "regime": regime,
                "model_name": model_name,
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "val_start": val_start.date(),
                "val_end": val_end.date(),
                "n_train": int(len(train)),
                "n_validation": int(len(val)),
                **metrics,
            }
        )
        fitted[model_name] = pipe

    model_df = pd.DataFrame(model_rows).sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False)
    best_model = model_df.iloc[0]["model_name"]
    model_df["selected_best_model"] = model_df["model_name"] == best_model

    val_pred = enrich_predictions(val.copy(), fitted[best_model], X_val)
    val_strategy_series = backtest_strategies(val_pred[["date", "ticker", RET_COL, "pred_label", "score"]].copy(), rf_val)

    strategy_rows = []
    for strategy, sdf in val_strategy_series.items():
        strategy_rows.append({"regime": regime, "selected_model": best_model, "strategy": strategy, **compute_metrics(sdf)})
    strategy_df = pd.DataFrame(strategy_rows).sort_values(
        ["excess_sharpe_net", "annualized_excess_return_net", "strategy"], ascending=[False, False, True]
    )
    selection_pool = strategy_df[strategy_df["strategy"].isin(STRATEGY_SELECTION_ELIGIBLE)].copy()
    best_strategy = selection_pool.iloc[0]["strategy"]
    strategy_df["selected_best_strategy"] = strategy_df["strategy"] == best_strategy

    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = train_val[ALL_FEATURES].copy()
    y_train_val = train_val[LABEL_COL].astype(str).copy()

    final_model = make_pipeline(model_specs[best_model])
    if best_model == "Gradient Boosting Classifier":
        sw_tv = compute_sample_weight(class_weight="balanced", y=y_train_val)
        final_model.fit(X_train_val, y_train_val, model__sample_weight=sw_tv)
    else:
        final_model.fit(X_train_val, y_train_val)

    test_pred = enrich_predictions(test.copy(), final_model, X_test)
    test_strategy_series = backtest_strategies(test_pred[["date", "ticker", RET_COL, "pred_label", "score"]].copy(), rf_test)
    test_best = test_strategy_series[best_strategy]
    test_metrics = compute_metrics(test_best)

    test_row = {
        "regime": regime,
        "selected_model": best_model,
        "selected_strategy": best_strategy,
        "test_start": test_start.date(),
        "test_end": test_end.date(),
        "n_test": int(len(test)),
        **test_metrics,
    }
    return model_df, strategy_df, pd.DataFrame([test_row])


def load_main_pipeline_summary():
    validation = pd.read_csv(MAIN_VALIDATION_METRICS)
    strategy_val = pd.read_csv(MAIN_STRATEGY_VALIDATION)
    selected_test = pd.read_csv(MAIN_SELECTED_TEST)

    selected_model_row = validation[validation["selected_best_model"] == True].iloc[0]
    selected_quant_row = selected_test[selected_test["strategy_group"] == "quant_only_test"].iloc[0]
    selected_strategy = selected_quant_row["strategy"]
    selected_val_strategy_row = strategy_val[strategy_val["strategy"] == selected_strategy].iloc[0]

    return selected_model_row, selected_val_strategy_row, selected_quant_row


def override_main_regime_with_tuned_outputs(model_out, strategy_out, test_out):
    model_row, val_strategy_row, test_row = load_main_pipeline_summary()

    model_out = model_out[model_out["regime"] != "main_2022_2023"].copy()
    model_out = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "regime": "main_2022_2023",
                        "model_name": model_row["model_name"],
                        "selected_hyperparameters_json": model_row["selected_hyperparameters_json"],
                        "train_start": "2015-01-01",
                        "train_end": "2021-12-31",
                        "val_start": "2022-01-01",
                        "val_end": "2023-12-31",
                        "n_train": np.nan,
                        "n_validation": np.nan,
                        "accuracy": model_row["validation_accuracy"],
                        "macro_f1": model_row["validation_macro_f1"],
                        "weighted_f1": model_row["validation_weighted_f1"],
                        "precision_increase": model_row["validation_precision_increase"],
                        "recall_increase": model_row["validation_recall_increase"],
                        "precision_hold": model_row["validation_precision_hold"],
                        "recall_hold": model_row["validation_recall_hold"],
                        "precision_reduce": model_row["validation_precision_reduce"],
                        "recall_reduce": model_row["validation_recall_reduce"],
                        "selected_best_model": True,
                        "note": "Main regime imported from tuned src/11_quant_only_model.py output for exact consistency.",
                    }
                ]
            ),
            model_out,
        ],
        ignore_index=True,
    )

    strategy_out = strategy_out[strategy_out["regime"] != "main_2022_2023"].copy()
    strategy_row = val_strategy_row.to_dict()
    strategy_row.update(
        {
            "regime": "main_2022_2023",
            "selected_model": model_row["model_name"],
            "selected_hyperparameters_json": model_row["selected_hyperparameters_json"],
            "selected_best_strategy": True,
            "note": "Main regime imported from tuned src/11_quant_only_model.py validation strategy table.",
        }
    )
    strategy_out = pd.concat([pd.DataFrame([strategy_row]), strategy_out], ignore_index=True)

    test_out = test_out[test_out["regime"] != "main_2022_2023"].copy()
    test_main = test_row.to_dict()
    test_main.update(
        {
            "regime": "main_2022_2023",
            "selected_model": model_row["model_name"],
            "selected_hyperparameters_json": model_row["selected_hyperparameters_json"],
            "selected_strategy": test_row["strategy"],
            "test_start": "2024-01-01",
            "test_end": "2025-11-30",
            "n_test": np.nan,
            "note": "Main regime imported from tuned src/11_quant_only_model.py selected test table.",
        }
    )
    test_main.pop("strategy", None)
    test_main.pop("strategy_group", None)
    test_out = pd.concat([pd.DataFrame([test_main]), test_out], ignore_index=True)
    return model_out, strategy_out, test_out


def add_hurdle_columns(test_out):
    baseline = pd.read_csv(BASELINE_TEST)
    hurdle_map = {
        row["strategy"]: row["excess_sharpe_net"]
        for _, row in baseline.iterrows()
        if row["strategy"] in ["Equal-Weight REIT Portfolio", "Benchmark VNQ", "Benchmark XLRE", "Benchmark SPY"]
    }
    test_out = test_out.copy()
    test_out["equal_weight_reit_net_excess_sharpe"] = hurdle_map.get("Equal-Weight REIT Portfolio", np.nan)
    test_out["vnq_net_excess_sharpe"] = hurdle_map.get("Benchmark VNQ", np.nan)
    test_out["xlre_net_excess_sharpe"] = hurdle_map.get("Benchmark XLRE", np.nan)
    test_out["spy_net_excess_sharpe"] = hurdle_map.get("Benchmark SPY", np.nan)
    test_out["beats_equal_weight_reit_net_excess_sharpe"] = (
        test_out["excess_sharpe_net"] > test_out["equal_weight_reit_net_excess_sharpe"]
    )
    test_out["beats_vnq_net_excess_sharpe"] = test_out["excess_sharpe_net"] > test_out["vnq_net_excess_sharpe"]
    test_out["beats_xlre_net_excess_sharpe"] = test_out["excess_sharpe_net"] > test_out["xlre_net_excess_sharpe"]
    test_out["beats_all_reit_hurdles"] = (
        test_out["beats_equal_weight_reit_net_excess_sharpe"]
        & test_out["beats_vnq_net_excess_sharpe"]
        & test_out["beats_xlre_net_excess_sharpe"]
    )
    return test_out, hurdle_map


def write_hurdle_comparison(test_out, hurdle_map):
    rows = []
    for _, row in test_out.iterrows():
        rows.append(
            {
                "regime": row["regime"],
                "candidate": "selected_quant_only_strategy",
                "strategy": row["selected_strategy"],
                "net_excess_sharpe": row["excess_sharpe_net"],
                "equal_weight_reit_net_excess_sharpe": hurdle_map.get("Equal-Weight REIT Portfolio", np.nan),
                "vnq_net_excess_sharpe": hurdle_map.get("Benchmark VNQ", np.nan),
                "xlre_net_excess_sharpe": hurdle_map.get("Benchmark XLRE", np.nan),
                "spy_net_excess_sharpe": hurdle_map.get("Benchmark SPY", np.nan),
                "beats_equal_weight_reit": row["beats_equal_weight_reit_net_excess_sharpe"],
                "beats_vnq": row["beats_vnq_net_excess_sharpe"],
                "beats_xlre": row["beats_xlre_net_excess_sharpe"],
                "beats_all_reit_hurdles": row["beats_all_reit_hurdles"],
                "note": "alt2_2019 should be treated as evidence of validation-regime sensitivity rather than ignored.",
            }
        )
    for strategy, sharpe in hurdle_map.items():
        rows.append(
            {
                "regime": "benchmark",
                "candidate": strategy,
                "strategy": strategy,
                "net_excess_sharpe": sharpe,
                "equal_weight_reit_net_excess_sharpe": hurdle_map.get("Equal-Weight REIT Portfolio", np.nan),
                "vnq_net_excess_sharpe": hurdle_map.get("Benchmark VNQ", np.nan),
                "xlre_net_excess_sharpe": hurdle_map.get("Benchmark XLRE", np.nan),
                "spy_net_excess_sharpe": hurdle_map.get("Benchmark SPY", np.nan),
                "beats_equal_weight_reit": sharpe > hurdle_map.get("Equal-Weight REIT Portfolio", np.nan),
                "beats_vnq": sharpe > hurdle_map.get("Benchmark VNQ", np.nan),
                "beats_xlre": sharpe > hurdle_map.get("Benchmark XLRE", np.nan),
                "beats_all_reit_hurdles": False,
                "note": "Benchmark comparator; SPY is broad-equity opportunity-cost context.",
            }
        )
    pd.DataFrame(rows).to_csv(OUT_HURDLE_COMPARISON, index=False)


def write_consistency_check(model_out, strategy_out, test_out):
    main_model, main_val_strategy, main_test = load_main_pipeline_summary()
    robust_model = model_out[(model_out["regime"] == "main_2022_2023") & (model_out["selected_best_model"] == True)].iloc[0]
    robust_strategy = strategy_out[
        (strategy_out["regime"] == "main_2022_2023") & (strategy_out["selected_best_strategy"] == True)
    ].iloc[0]
    robust_test = test_out[test_out["regime"] == "main_2022_2023"].iloc[0]

    rows = [
        ("selected_model", main_model["model_name"], robust_model["model_name"], "Tuned model family selected by 2022-2023 validation macro F1."),
        (
            "selected_hyperparameters",
            main_model["selected_hyperparameters_json"],
            robust_model.get("selected_hyperparameters_json", ""),
            "Hyperparameters selected by purged walk-forward CV inside 2015-2021.",
        ),
        ("selected_strategy", main_test["strategy"], robust_test["selected_strategy"], "Rule selected by validation net excess Sharpe."),
        (
            "validation_net_excess_sharpe",
            main_val_strategy["excess_sharpe_net"],
            robust_strategy["excess_sharpe_net"],
            "Selected strategy validation-period net excess Sharpe.",
        ),
        ("test_net_excess_sharpe", main_test["excess_sharpe_net"], robust_test["excess_sharpe_net"], "Selected strategy test net excess Sharpe."),
        ("test_final_nav_net", main_test["final_nav_net"], robust_test["final_nav_net"], "Selected strategy test final net NAV."),
        ("test_max_drawdown_net", main_test["max_drawdown_net"], robust_test["max_drawdown_net"], "Selected strategy test net max drawdown."),
        (
            "avg_monthly_turnover",
            main_test["avg_monthly_turnover"],
            robust_test["avg_monthly_turnover"],
            "Selected strategy average monthly turnover.",
        ),
    ]

    out = []
    for item, main_val, robust_val, note in rows:
        if isinstance(main_val, str) or isinstance(robust_val, str):
            consistent = str(main_val) == str(robust_val)
        else:
            consistent = bool(np.isclose(float(main_val), float(robust_val), rtol=1e-10, atol=1e-10))
        out.append(
            {
                "item": item,
                "main_pipeline_value": main_val,
                "robustness_script_value": robust_val,
                "consistent": consistent,
                "note": note,
            }
        )
    pd.DataFrame(out).to_csv(OUT_CONSISTENCY, index=False)


def main():
    panel_path = pick_panel_path()
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    require_columns(panel, ["ticker", "date", "sector", LABEL_COL, RET_COL] + PRICE_FEATURES, panel_path.name)

    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel["date"] = pd.to_datetime(panel["date"])
    panel[RET_COL] = pd.to_numeric(panel[RET_COL], errors="coerce")
    panel = panel.dropna(subset=["ticker", "date", RET_COL]).sort_values(["date", "ticker"]).copy()

    macro = load_macro_with_lags_and_rf()
    panel = attach_macro_features(panel, macro)

    model_parts = []
    strategy_parts = []
    test_parts = []
    for regime_cfg in REGIMES:
        mdf, sdf, tdf = run_regime(panel, regime_cfg)
        model_parts.append(mdf)
        strategy_parts.append(sdf)
        test_parts.append(tdf)

    model_out = pd.concat(model_parts, ignore_index=True)
    strategy_out = pd.concat(strategy_parts, ignore_index=True)
    test_out = pd.concat(test_parts, ignore_index=True)

    model_out, strategy_out, test_out = override_main_regime_with_tuned_outputs(model_out, strategy_out, test_out)
    test_out, hurdle_map = add_hurdle_columns(test_out)
    write_hurdle_comparison(test_out, hurdle_map)
    write_consistency_check(model_out, strategy_out, test_out)

    model_out.to_csv(OUT_MODEL_SELECTION, index=False)
    strategy_out.to_csv(OUT_STRATEGY_SELECTION, index=False)
    test_out.to_csv(OUT_TEST_PERF, index=False)

    print("Saved:", OUT_MODEL_SELECTION)
    print("Saved:", OUT_STRATEGY_SELECTION)
    print("Saved:", OUT_TEST_PERF)
    print("Saved:", OUT_CONSISTENCY)
    print("Saved:", OUT_HURDLE_COMPARISON)
    print("\nSelected model by regime:")
    print(
        model_out[model_out["selected_best_model"]][["regime", "model_name", "macro_f1", "weighted_f1", "accuracy"]]
        .sort_values("regime")
        .to_string(index=False)
    )
    print("\nSelected strategy by regime:")
    print(
        strategy_out[strategy_out["selected_best_strategy"]][
            ["regime", "selected_model", "strategy", "excess_sharpe_net", "annualized_excess_return_net"]
        ]
        .sort_values("regime")
        .to_string(index=False)
    )
    print("\nSelected strategy test performance:")
    print(
        test_out[
            [
                "regime",
                "selected_model",
                "selected_strategy",
                "excess_sharpe_net",
                "annualized_excess_return_net",
                "max_drawdown_net",
                "n_months",
            ]
        ]
        .sort_values("regime")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
