"""One-off verification of the feature t-test table in
outputs/reports/v6_baseline_analysis.md (§3.2) and
outputs/reports/v6_final_classification_limitations_report.md (§1.2).

Recomputes per-feature Welch t-statistics on the V6 enriched test split,
splitting samples into reduce vs non-reduce groups, and prints actuals
against the documented values for cross-checking.

Run: python tests/verify_feature_ttest.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT = ROOT / "data" / "processed" / "splits" / "enriched_test_2024_2025.csv"

# Features to verify, with documented (mean_reduce, mean_nonreduce, t-stat).
# dividend_yield ambiguous (lag vs contemporaneous) -> test both.
DOCUMENTED = {
    "ret_1m": (0.0039, 0.0052, -0.24),
    "ret_3m": (0.0137, 0.0143, -0.09),
    "vol_annualized": (0.2129, 0.2253, -1.76),
    "dividend_yield": (None, None, -0.27),
    "dividend_yield_lag1": (None, None, None),
    "FEDFUNDS": (4.68, 4.83, -3.35),
}


def main() -> None:
    df = pd.read_csv(TEST_SPLIT)
    is_reduce = df["label"] == "reduce"
    reduce_grp = df[is_reduce]
    nonreduce_grp = df[~is_reduce]
    print(f"N total={len(df)}  reduce={len(reduce_grp)}  non-reduce={len(nonreduce_grp)}\n")

    header = f"{'feature':<22}{'mean_reduce':>14}{'mean_nonred':>14}{'t_stat':>10}{'doc_t':>9}  check"
    print(header)
    print("-" * len(header))

    for feat, (doc_mr, doc_nr, doc_t) in DOCUMENTED.items():
        if feat not in df.columns:
            print(f"{feat:<22}{'(column missing)':>40}")
            continue
        a = reduce_grp[feat].dropna()
        b = nonreduce_grp[feat].dropna()
        t_stat, _p = stats.ttest_ind(a, b, equal_var=False)  # Welch
        mr, nr = a.mean(), b.mean()
        if doc_t is None:
            check = "n/a"
        else:
            check = "OK" if abs(t_stat - doc_t) <= 0.1 else "MISMATCH"
        doc_t_str = f"{doc_t:>9.2f}" if doc_t is not None else f"{'-':>9}"
        print(f"{feat:<22}{mr:>14.4f}{nr:>14.4f}{t_stat:>10.2f}{doc_t_str}  {check}")


if __name__ == "__main__":
    main()
