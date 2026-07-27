"""V6 baseline reproduction regression test (CLAUDE.md §8 `make reproduce_v6`).

Re-runs the structured-ML baseline (src/11_quant_only_model.py, Zone 1, called
not modified) on the enriched panel, then diffs the regenerated key output
tables against the frozen golden snapshots in tests/fixtures/v6_reference/ to
6 decimal places.

The golden snapshots are the permanent reference; re-running regenerates
outputs/tables/ in place, so the golden copy is never at risk.

Run: python tests/test_v6_reproduction.py   (or: make reproduce_v6)
Exit 0 = all tables match; exit 1 = drift detected.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "11_quant_only_model.py"
REF_DIR = ROOT / "tests" / "fixtures" / "v6_reference"
OUT_DIR = ROOT / "outputs" / "tables"
TOL = 1e-6

TABLES = [
    "quant_only_confusion_matrix_test.csv",
    "quant_only_selected_hyperparameters_cv.csv",
    "quant_only_test_metrics.csv",
    "quant_only_reduce_probability_diagnostics_test.csv",
]


def compare_table(name: str) -> list[str]:
    """Return a list of diff messages; empty list means the table matches."""
    ref = pd.read_csv(REF_DIR / name)
    new = pd.read_csv(OUT_DIR / name)
    diffs: list[str] = []

    if list(ref.columns) != list(new.columns):
        return [f"  column mismatch: ref={list(ref.columns)} new={list(new.columns)}"]
    if ref.shape != new.shape:
        return [f"  shape mismatch: ref={ref.shape} new={new.shape}"]

    for col in ref.columns:
        r, n = ref[col], new[col]
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(n):
            delta = (r - n).abs()
            bad = delta[delta > TOL]
            for idx in bad.index:
                diffs.append(f"  [{col}][row {idx}] ref={r[idx]} new={n[idx]} (|Δ|={delta[idx]:.2e})")
        else:  # string / JSON columns: exact match
            for idx in r.index:
                if str(r[idx]) != str(n[idx]):
                    diffs.append(f"  [{col}][row {idx}] ref={r[idx]!r} new={n[idx]!r}")
    return diffs


def main() -> int:
    print("=== reproduce_v6: re-running src/11 on enriched panel ===")
    proc = subprocess.run(
        [sys.executable, str(SRC), "--panel", "enriched"],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        print(f"ERROR: src/11 exited with code {proc.returncode}")
        return 1

    print("\n=== Diffing regenerated tables against golden snapshots (tol=1e-6) ===")
    all_ok = True
    for name in TABLES:
        if not (OUT_DIR / name).exists():
            print(f"MISSING: {name} was not regenerated")
            all_ok = False
            continue
        diffs = compare_table(name)
        if diffs:
            all_ok = False
            print(f"MISMATCH: {name}")
            for d in diffs:
                print(d)
        else:
            print(f"OK: {name}")

    if all_ok:
        print("\nreproduce_v6 PASSED: all key tables match V6 reference to 6 dp.")
        return 0
    print("\nreproduce_v6 FAILED: drift detected (see above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
