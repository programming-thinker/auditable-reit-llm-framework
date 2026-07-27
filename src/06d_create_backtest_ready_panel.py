from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "reit_monthly_panel.csv"
OUT_PATH = ROOT / "data" / "processed" / "backtest_ready_panel.csv"

panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])

required = [
    "ticker",
    "date",
    "adj_close",
    "ret_1m",
    "ret_3m",
    "vol_annualized",
    "drawdown",
    "future_ret_1m",
    "label"
]

missing_cols = [c for c in required if c not in panel.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

bt = panel.dropna(subset=[
    "ticker",
    "date",
    "adj_close",
    "ret_1m",
    "future_ret_1m",
    "label"
]).copy()

bt = bt.sort_values(["date", "ticker"])
bt.to_csv(OUT_PATH, index=False)

print(bt.shape)
print(bt.head())
