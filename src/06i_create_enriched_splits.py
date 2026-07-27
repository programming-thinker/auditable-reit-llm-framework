"""
06i_create_enriched_splits.py
==============================
Generate train/val/test splits from enriched panel for classification baseline.

This creates split files compatible with 11_quant_only_model.py:
- data/processed/splits/enriched_train_2015_2021.csv
- data/processed/splits/enriched_validation_2022_2023.csv
- data/processed/splits/enriched_test_2024_2025.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENRICHED_PANEL = ROOT / "data" / "processed" / "backtest_ready_panel_enriched.csv"
OUT_DIR = ROOT / "data" / "processed" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Date ranges (must match 11_quant_only_model.py)
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-11-30")

print("="*70)
print("CREATING ENRICHED PANEL SPLITS")
print("="*70)

# Load enriched panel
print(f"\nLoading: {ENRICHED_PANEL}")
df = pd.read_csv(ENRICHED_PANEL, parse_dates=["date"])
print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Required columns
required = ["ticker", "date", "label", "future_ret_1m", "ret_1m"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Create splits
train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()

print(f"\nSplit sizes:")
print(f"  Train:      {len(train)} rows ({TRAIN_START.date()} to {TRAIN_END.date()})")
print(f"  Validation: {len(val)} rows ({VAL_START.date()} to {VAL_END.date()})")
print(f"  Test:       {len(test)} rows ({TEST_START.date()} to {TEST_END.date()})")

# Save
train_path = OUT_DIR / "enriched_train_2015_2021.csv"
val_path = OUT_DIR / "enriched_validation_2022_2023.csv"
test_path = OUT_DIR / "enriched_test_2024_2025.csv"

train.to_csv(train_path, index=False)
val.to_csv(val_path, index=False)
test.to_csv(test_path, index=False)

print(f"\nSaved:")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {test_path}")

# Class distribution check
print(f"\nClass distribution:")
for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
    counts = split_df["label"].value_counts()
    total = len(split_df)
    print(f"\n  {split_name}:")
    for label in ["increase", "hold", "reduce"]:
        count = counts.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"    {label:10s}: {count:4d} ({pct:5.1f}%)")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
