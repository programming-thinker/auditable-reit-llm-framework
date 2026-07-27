"""
99_quick_regime_shift_analysis.py
===================================
快速生成 regime shift 分析图表（用于毕业论文答辩）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Load data
train = pd.read_csv(ROOT / "data/processed/splits/enriched_train_2015_2021.csv", parse_dates=['date'])
val = pd.read_csv(ROOT / "data/processed/splits/enriched_validation_2022_2023.csv", parse_dates=['date'])
test = pd.read_csv(ROOT / "data/processed/splits/enriched_test_2024_2025.csv", parse_dates=['date'])

# Combine
train['split'] = 'Train (2015-2021)'
val['split'] = 'Validation (2022-2023)'
test['split'] = 'Test (2024-2025)'
all_data = pd.concat([train, val, test], ignore_index=True)

# ============================================================================
# Figure 1: Interest Rate Regime Shift
# ============================================================================
plt.figure(figsize=(12, 5))

monthly_avg = all_data.groupby('date').agg({
    'FEDFUNDS_lag1': 'first',
    'DGS10_lag1': 'first'
}).reset_index()

plt.subplot(1, 2, 1)
plt.plot(monthly_avg['date'], monthly_avg['FEDFUNDS_lag1'], linewidth=2, label='Fed Funds Rate', color='darkblue')
plt.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2021-12-31'), alpha=0.2, color='green', label='Train')
plt.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'), alpha=0.2, color='orange', label='Validation')
plt.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2025-12-31'), alpha=0.2, color='red', label='Test')
plt.axhline(train['FEDFUNDS_lag1'].mean(), color='green', linestyle='--', alpha=0.7, label=f'Train Mean: {train["FEDFUNDS_lag1"].mean():.2f}%')
plt.axhline(test['FEDFUNDS_lag1'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Test Mean: {test["FEDFUNDS_lag1"].mean():.2f}%')
plt.title('Federal Funds Rate: Train vs. Test Regime', fontsize=14, fontweight='bold')
plt.ylabel('Fed Funds Rate (%)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(loc='upper left', fontsize=9)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(monthly_avg['date'], monthly_avg['DGS10_lag1'], linewidth=2, label='10-Year Treasury', color='darkgreen')
plt.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2021-12-31'), alpha=0.2, color='green')
plt.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'), alpha=0.2, color='orange')
plt.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2025-12-31'), alpha=0.2, color='red')
plt.axhline(train['DGS10_lag1'].mean(), color='green', linestyle='--', alpha=0.7)
plt.axhline(test['DGS10_lag1'].mean(), color='red', linestyle='--', alpha=0.7)
plt.title('10-Year Treasury Yield: Train vs. Test Regime', fontsize=14, fontweight='bold')
plt.ylabel('10-Year Yield (%)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG / "regime_shift_interest_rates.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUT_FIG / 'regime_shift_interest_rates.png'}")

# ============================================================================
# Figure 2: Reduce Label Distribution Over Time
# ============================================================================
plt.figure(figsize=(14, 5))

# Count by month
label_counts = test.groupby('date')['label'].value_counts().unstack(fill_value=0)
label_pct = label_counts.div(label_counts.sum(axis=1), axis=0) * 100

plt.subplot(1, 2, 1)
label_pct.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#2ecc71', '#3498db', '#e74c3c'], width=0.8)
plt.title('Test Period: Label Distribution by Month', fontsize=14, fontweight='bold')
plt.ylabel('Percentage (%)', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.legend(title='Label', loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.axhline(33.33, color='gray', linestyle='--', alpha=0.5, label='Equal Distribution')
plt.grid(axis='y', alpha=0.3)

# Highlight high-reduce months
plt.subplot(1, 2, 2)
reduce_counts = label_counts['reduce']
plt.bar(range(len(reduce_counts)), reduce_counts, color=['red' if x > 10 else 'lightcoral' for x in reduce_counts])
plt.axhline(10, color='darkred', linestyle='--', linewidth=2, label='Threshold: 10 REITs')
plt.title('Test Period: Number of Reduce Events per Month', fontsize=14, fontweight='bold')
plt.ylabel('Number of Reduce Events', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(range(len(reduce_counts)), [d.strftime('%Y-%m') for d in reduce_counts.index], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Annotate extreme months
for i, val in enumerate(reduce_counts):
    if val > 15:
        plt.text(i, val + 0.5, f'{val}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_FIG / "test_period_reduce_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUT_FIG / 'test_period_reduce_distribution.png'}")

# ============================================================================
# Figure 3: Feature Distribution Comparison (Train vs Test)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features = ['ret_1m', 'vol_annualized', 'FEDFUNDS_lag1', 'DGS10_lag1', 'drawdown', 'dividend_yield_lag1']

for idx, feat in enumerate(features):
    ax = axes[idx // 3, idx % 3]

    if feat in train.columns and feat in test.columns:
        train_vals = train[feat].dropna()
        test_vals = test[feat].dropna()

        ax.hist(train_vals, bins=30, alpha=0.5, label='Train', color='green', density=True)
        ax.hist(test_vals, bins=30, alpha=0.5, label='Test', color='red', density=True)
        ax.axvline(train_vals.mean(), color='green', linestyle='--', linewidth=2, label=f'Train μ={train_vals.mean():.3f}')
        ax.axvline(test_vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Test μ={test_vals.mean():.3f}')
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.suptitle('Feature Distribution: Train vs. Test (Evidence of Domain Shift)', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUT_FIG / "feature_distribution_train_vs_test.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUT_FIG / 'feature_distribution_train_vs_test.png'}")

# ============================================================================
# Summary Statistics Table
# ============================================================================
summary = pd.DataFrame({
    'Period': ['Train (2015-2021)', 'Validation (2022-2023)', 'Test (2024-2025)'],
    'N Observations': [len(train), len(val), len(test)],
    'Mean FEDFUNDS (%)': [train['FEDFUNDS_lag1'].mean(), val['FEDFUNDS_lag1'].mean(), test['FEDFUNDS_lag1'].mean()],
    'Mean DGS10 (%)': [train['DGS10_lag1'].mean(), val['DGS10_lag1'].mean(), test['DGS10_lag1'].mean()],
    'Reduce %': [
        (train['label']=='reduce').sum() / len(train) * 100,
        (val['label']=='reduce').sum() / len(val) * 100,
        (test['label']=='reduce').sum() / len(test) * 100
    ]
})

summary.to_csv(OUT_FIG.parent / "tables" / "regime_shift_summary_statistics.csv", index=False)
print(f"✓ Saved: {OUT_FIG.parent / 'tables' / 'regime_shift_summary_statistics.csv'}")

print("\n" + "="*70)
print("REGIME SHIFT ANALYSIS COMPLETE")
print("="*70)
print("\n论文答辩用图表已生成：")
print(f"  1. {OUT_FIG / 'regime_shift_interest_rates.png'}")
print(f"  2. {OUT_FIG / 'test_period_reduce_distribution.png'}")
print(f"  3. {OUT_FIG / 'feature_distribution_train_vs_test.png'}")
print(f"  4. {OUT_FIG.parent / 'tables' / 'regime_shift_summary_statistics.csv'}")
print("\n这些图直接展示了 domain shift 问题，支持您的核心论点。")
print("="*70)
