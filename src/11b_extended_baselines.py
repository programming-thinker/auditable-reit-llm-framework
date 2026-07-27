"""
11b_extended_baselines.py
==========================
Extended baseline models: OLS regression and Fama-MacBeth cross-sectional regression.

This script complements 11_quant_only_model.py by adding:
1. OLS regression on continuous returns (Welch & Goyal 2008 style)
2. Fama-MacBeth two-pass cross-sectional regression (Fama & MacBeth 1973)

Both models predict future_ret_1m (continuous) rather than label (discrete).
Results are saved separately and can be compared with classification baselines.

Usage:
    python src/11b_extended_baselines.py --panel original  # 46 features
    python src/11b_extended_baselines.py --panel enriched  # 90 features
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]

PANEL_PATHS = {
    "original": ROOT / "data" / "processed" / "backtest_ready_panel.csv",
    "enriched": ROOT / "data" / "processed" / "backtest_ready_panel_enriched.csv",
}

OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-11-30")

# Original features (from 11_quant_only_model.py)
PRICE_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_annualized", "drawdown"]
MACRO_FEATURES = ["FEDFUNDS_lag1", "DGS10_lag1", "DGS2_lag1", "CPIAUCSL_lag1", "UNRATE_lag1"]
SECTOR_FEATURE = "sector"

ORIGINAL_FEATURES = PRICE_FEATURES + MACRO_FEATURES + [SECTOR_FEATURE]

# Fama-MacBeth features (exclude long-term momentum to avoid startup period)
# ret_12m requires 12 months of history, causing first year to have missing values
# Fama-MacBeth focuses on cross-sectional variation, not long-term time-series
FM_PRICE_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "vol_annualized", "drawdown"]
FM_FEATURES_BASE = FM_PRICE_FEATURES + MACRO_FEATURES + [SECTOR_FEATURE]

# Enriched features (new from 06e-06h)
ENRICHED_ADDITIONAL = [
    "mortgage30", "baa_spread_10y", "baa_yield",
    "home_price_index", "housing_starts", "building_permits",
    "industrial_production", "real_disp_income",
    "mortgage_spread", "mortgage30_chg_1m", "mortgage30_chg_3m",
    "baa_spread_chg_1m", "baa_spread_chg_3m",
    "baa_yield_chg_1m", "baa_yield_chg_3m",
    "home_price_yoy", "home_price_mom",
    "housing_starts_yoy", "permits_yoy",
    "indpro_yoy", "real_income_yoy",
    "dividend_yield",
    # lag1 versions
    "mortgage30_lag1", "baa_spread_10y_lag1", "baa_yield_lag1",
    "home_price_index_lag1", "housing_starts_lag1", "building_permits_lag1",
    "industrial_production_lag1", "real_disp_income_lag1",
    "mortgage_spread_lag1", "mortgage30_chg_1m_lag1", "mortgage30_chg_3m_lag1",
    "baa_spread_chg_1m_lag1", "baa_spread_chg_3m_lag1",
    "baa_yield_chg_1m_lag1", "baa_yield_chg_3m_lag1",
    "home_price_yoy_lag1", "home_price_mom_lag1",
    "housing_starts_yoy_lag1", "permits_yoy_lag1",
    "indpro_yoy_lag1", "real_income_yoy_lag1",
    "dividend_yield_lag1",
]

TARGET_COL = "future_ret_1m"


def load_panel(panel_type: str) -> pd.DataFrame:
    """Load and prepare panel data."""
    path = PANEL_PATHS[panel_type]
    df = pd.read_csv(path, parse_dates=["date"])

    # Filter dates
    df = df[
        (df["date"] >= TRAIN_START) & (df["date"] <= TEST_END)
    ].copy()

    # Check target exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in panel")

    print(f"Loaded {panel_type} panel: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def get_feature_list(panel_type: str, df: pd.DataFrame) -> list:
    """Return feature list based on panel type."""
    if panel_type == "original":
        base_features = [f for f in ORIGINAL_FEATURES if f in df.columns and f != "sector"]
    else:  # enriched
        base_features = [f for f in ORIGINAL_FEATURES + ENRICHED_ADDITIONAL
                        if f in df.columns and f != "sector"]

    # Add one-hot encoded sector columns if sector exists
    if "sector" in df.columns:
        sector_cols = []
        # Check if already one-hot encoded
        if any(c.startswith("sector_") for c in df.columns):
            sector_cols = [c for c in df.columns if c.startswith("sector_")]
        else:
            # Need to encode - will handle in prepare_splits
            pass
    else:
        sector_cols = [c for c in df.columns if c.startswith("sector_")]

    return base_features + sector_cols


def prepare_splits(df: pd.DataFrame, features: list):
    """Split data into train/val/test and one-hot encode sector if needed."""
    # One-hot encode sector if it's categorical
    if "sector" in df.columns and df["sector"].dtype == "object":
        print("One-hot encoding sector column...")
        sector_dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=True)
        df = pd.concat([df, sector_dummies], axis=1)

        # Update feature list
        features_updated = [f for f in features if f != "sector"]
        features_updated.extend(sector_dummies.columns.tolist())
    else:
        features_updated = features

    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()

    # Drop NaNs in target or features
    for split in [train, val, test]:
        split.dropna(subset=[TARGET_COL] + features_updated, inplace=True)

    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Features after encoding: {len(features_updated)}")

    return train, val, test, features_updated


def run_ols_regression(train, val, test, features):
    """
    OLS regression: predict future_ret_1m from features.

    Following Welch & Goyal (2008, RFS) methodology.
    """
    print("\n" + "="*60)
    print("OLS REGRESSION")
    print("="*60)

    # Prepare data
    X_train = train[features].values
    y_train = train[TARGET_COL].values
    X_val = val[features].values
    y_val = val[TARGET_COL].values
    X_test = test[features].values
    y_test = test[TARGET_COL].values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Fit OLS
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    def compute_metrics(y_true, y_pred, label):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        print(f"\n{label}:")
        print(f"  R² = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE = {mae:.4f}")

        return {"period": label, "r2": r2, "rmse": rmse, "mae": mae, "n_obs": len(y_true)}

    metrics_train = compute_metrics(y_train, y_train_pred, "Training (2015-2021)")
    metrics_val = compute_metrics(y_val, y_val_pred, "Validation (2022-2023)")
    metrics_test = compute_metrics(y_test, y_test_pred, "Test (2024-2025)")

    # Coefficients
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_)
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\nTop 10 coefficients by magnitude:")
    print(coef_df.head(10).to_string(index=False))

    # Statistical significance (approximate via bootstrap on training set)
    # Note: True significance requires heteroskedasticity-robust SEs (beyond scope)
    print(f"\nIntercept: {model.intercept_:.6f}")

    return {
        "model": model,
        "scaler": scaler,
        "coefficients": coef_df,
        "metrics": pd.DataFrame([metrics_train, metrics_val, metrics_test]),
        "predictions": {
            "train": pd.DataFrame({"date": train["date"], "ticker": train["ticker"],
                                   "actual": y_train, "predicted": y_train_pred}),
            "val": pd.DataFrame({"date": val["date"], "ticker": val["ticker"],
                                "actual": y_val, "predicted": y_val_pred}),
            "test": pd.DataFrame({"date": test["date"], "ticker": test["ticker"],
                                 "actual": y_test, "predicted": y_test_pred}),
        }
    }


def run_fama_macbeth(train, val, test, features):
    """
    Fama-MacBeth two-pass regression.

    Following Fama & MacBeth (1973, JPE) methodology:
    1. For each month t, run cross-sectional OLS: ret_{i,t+1} ~ features_{i,t}
    2. Collect time-series of coefficients
    3. Compute mean and t-stat across months

    Note: Uses FM_FEATURES_BASE (excludes ret_12m) to avoid startup period missing data.
    """
    print("\n" + "="*60)
    print("FAMA-MACBETH CROSS-SECTIONAL REGRESSION")
    print("="*60)
    print("Using short-term features (excludes ret_12m) to maximize valid months")

    # Extract FM-specific features from current feature list
    # Remove ret_12m and any enriched long-term features
    fm_features = [f for f in features if 'ret_12m' not in f and '12m' not in f]

    print(f"FM features: {len(fm_features)} (vs. {len(features)} for OLS)")

    def run_cross_sectional_month(df_month, features_to_use):
        """Run OLS for one month's cross-section."""
        if len(df_month) < 10:  # Need enough observations
            return None

        try:
            # Extract features and target
            X_df = df_month[features_to_use].copy()
            y = df_month[TARGET_COL].values

            # Convert bool columns to int (from one-hot encoding)
            for col in X_df.columns:
                if X_df[col].dtype == bool:
                    X_df[col] = X_df[col].astype(int)

            # Ensure all numeric
            X_df = X_df.select_dtypes(include=[np.number])

            # Check if all features survived
            if X_df.shape[1] == 0:
                return None

            X = X_df.values

            # Check for NaN
            if np.any(pd.isna(X)) or np.any(pd.isna(y)):
                return None

            # Check sufficient observations
            if len(X) < 10:
                return None

            # OLS
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(X_scaled, y)

            # Pad coefficients if some features were dropped
            if len(model.coef_) < len(features_to_use):
                # This shouldn't happen but handle gracefully
                return None

            return {
                "intercept": model.intercept_,
                "coefficients": model.coef_,
                "n_obs": len(y)
            }
        except Exception as e:
            # Silently ignore errors in cross-sectional regression
            return None

    # Run for each period
    def fama_macbeth_period(df, period_name, features_to_use):
        print(f"\n{period_name}:")

        # Pre-filter: drop rows with missing target or features
        df_complete = df.dropna(subset=[TARGET_COL] + features_to_use).copy()

        if len(df_complete) == 0:
            print("  No observations after dropna")
            return None

        coef_list = []
        dates = []

        for date, group in df_complete.groupby("date"):
            result = run_cross_sectional_month(group, features_to_use)
            if result is not None:
                coef_list.append(result["coefficients"])
                dates.append(date)

        if len(coef_list) == 0:
            print("  No valid months (all cross-sections failed)")
            return None

        # Stack coefficients (T x K matrix)
        coef_matrix = np.vstack(coef_list)

        # Mean and std across time
        mean_coef = np.mean(coef_matrix, axis=0)
        std_coef = np.std(coef_matrix, axis=0, ddof=1)
        t_stat = mean_coef / (std_coef / np.sqrt(len(coef_list)))

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(coef_list)-1))

        results_df = pd.DataFrame({
            "feature": features_to_use,
            "mean_coefficient": mean_coef,
            "std_coefficient": std_coef,
            "t_statistic": t_stat,
            "p_value": p_value,
            "n_months": len(coef_list)
        })

        print(f"  {len(coef_list)} months with valid cross-sections")
        print(f"\n  Top 10 coefficients by |t-stat|:")
        print(results_df.sort_values("t_statistic", key=abs, ascending=False).head(10).to_string(index=False))

        return results_df

    fm_train = fama_macbeth_period(train, "Training (2015-2021)", fm_features)
    fm_val = fama_macbeth_period(val, "Validation (2022-2023)", fm_features)
    fm_test = fama_macbeth_period(test, "Test (2024-2025)", fm_features)

    return {
        "train": fm_train,
        "val": fm_val,
        "test": fm_test
    }


def main():
    parser = argparse.ArgumentParser(description="Extended baseline models (OLS + Fama-MacBeth)")
    parser.add_argument("--panel", choices=["original", "enriched"], default="original",
                        help="Which panel to use (original=46 cols, enriched=90 cols)")
    args = parser.parse_args()

    print("="*70)
    print(f"EXTENDED BASELINES: OLS & FAMA-MACBETH")
    print(f"Panel type: {args.panel}")
    print("="*70)

    # Load data
    df = load_panel(args.panel)
    features = get_feature_list(args.panel, df)
    print(f"\nInitial features: {len(features)}")

    # Split (will handle one-hot encoding)
    train, val, test, features = prepare_splits(df, features)

    print(f"Final features: {len(features)}")

    # Run OLS
    ols_results = run_ols_regression(train, val, test, features)

    # Run Fama-MacBeth
    fm_results = run_fama_macbeth(train, val, test, features)

    # Save results
    suffix = f"_{args.panel}"

    ols_results["coefficients"].to_csv(OUT_DIR / f"ols_coefficients{suffix}.csv", index=False)
    ols_results["metrics"].to_csv(OUT_DIR / f"ols_metrics{suffix}.csv", index=False)
    ols_results["predictions"]["test"].to_csv(OUT_DIR / f"ols_predictions_test{suffix}.csv", index=False)

    if fm_results["train"] is not None:
        fm_results["train"].to_csv(OUT_DIR / f"fama_macbeth_train{suffix}.csv", index=False)
    if fm_results["val"] is not None:
        fm_results["val"].to_csv(OUT_DIR / f"fama_macbeth_val{suffix}.csv", index=False)
    if fm_results["test"] is not None:
        fm_results["test"].to_csv(OUT_DIR / f"fama_macbeth_test{suffix}.csv", index=False)

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nResults saved to {OUT_DIR}/")
    print(f"  - ols_coefficients{suffix}.csv")
    print(f"  - ols_metrics{suffix}.csv")
    print(f"  - ols_predictions_test{suffix}.csv")
    print(f"  - fama_macbeth_*{suffix}.csv")


if __name__ == "__main__":
    main()
