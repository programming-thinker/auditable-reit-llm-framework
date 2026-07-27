"""Statistical power / minimal-detectable-effect (MDE) analysis for the headline
Table 7 comparison: LLM five-agent reduce recall (0.206) vs random-at-budget
(0.204) on the 575 test decisions (165 true reduce).

Replicates the EXACT resampling schemes of the two source scripts:
  - REIT-block bootstrap  (analysis/build_llm_comparison.py::block_bootstrap):
      resample the 25 tickers with replacement; within each resample compute
      LLM reduce recall and a single random-at-budget draw (k = round(p_fire*n)
      row indices without replacement); diff = LLM - random.
  - Month-block bootstrap (analysis/inference_robustness.py::month_block_bootstrap):
      identical, but resampling the 23 decision months with replacement.

Outputs (written to outputs/fundamentals_robustness/power_mde.csv):
  1. Bootstrap SE of the LLM-minus-random reduce-recall difference under both
     schemes (B = 2000, fixed seed), via the literal row-level resampler.
  2. MDE at 80% power, alpha = 0.05 two-sided:
     (a) analytic:  MDE = (z_{0.975} + z_{0.80}) * SE = 2.8016 * SE;
     (b) simulation: inject a true improvement delta by flipping
         round(delta * 165) randomly chosen missed true-reduce decisions to
         hits while flipping the same number of false-positive 'reduce' calls
         to non-fires (firing budget held exactly fixed); re-run the percentile
         bootstrap test at alpha = 0.05 (B = 2000) and find the smallest delta
         in {0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20} with >= 80% rejection
         rate (200 sims per delta; both schemes).

For the simulations the bootstrap is computed from per-block sufficient
statistics, with the random-at-budget arm drawn as
Hypergeometric(n_reduce, n - n_reduce, k) -- exactly the distribution of
`ir[rng.choice(n, k, replace=False)].sum()` in the source scripts, so the test
is distributionally identical while running ~100x faster.

No API calls. Read-only w.r.t. Zone 1 and audit_log; writes only the CSV above.

Usage:
    python analysis/power_mde.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEC = REPO / "audit_log/decisions.jsonl"
TEST_SPLIT = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
HEADLINE = REPO / "outputs/llm_deepseek_test/headline_comparison.csv"
OUT = REPO / "outputs/fundamentals_robustness/power_mde.csv"

SEED = 20260626          # same seed family as the source scripts
B = 2000                 # bootstrap replicates (per task spec)
N_SIM = 200              # simulations per delta
ALPHA = 0.05
DELTAS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
Z_975 = 1.959963984540054   # Phi^{-1}(0.975)
Z_80 = 0.8416212335729143   # Phi^{-1}(0.80)
SCHEME_ID = {"reit_block": 0, "month_block": 1}


# --------------------------------------------------------------------------- #
# Loading (mirrors build_llm_comparison.load_llm_predictions + truth merge)
# --------------------------------------------------------------------------- #
def load_merged() -> pd.DataFrame:
    """df[date, ticker, llm_pred, true_label] on the 575 test decisions."""
    rows = []
    with DEC.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec["final_probabilities"]
            rows.append(
                {
                    "date": pd.to_datetime(rec["decision_date_t"]).strftime("%Y-%m-%d"),
                    "ticker": rec["ticker"],
                    "llm_pred": max(p, key=p.get),
                }
            )
    llm = pd.DataFrame(rows).drop_duplicates(subset=["date", "ticker"], keep="last")
    truth = pd.read_csv(TEST_SPLIT)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    df = llm.merge(
        truth[["date", "ticker", "label"]].rename(columns={"label": "true_label"}),
        on=["date", "ticker"],
        how="inner",
    ).reset_index(drop=True)
    return df


def make_groups(df: pd.DataFrame, key: str) -> list[np.ndarray]:
    """Row-index arrays per unique value of `key` (ticker or date)."""
    return [np.asarray(idx) for _, idx in df.groupby(key).indices.items()]


# --------------------------------------------------------------------------- #
# 1. Literal block bootstrap (row-level, faithful to the source scripts)
# --------------------------------------------------------------------------- #
def boot_diffs_literal(
    is_red: np.ndarray,
    pred_red: np.ndarray,
    groups: list[np.ndarray],
    p_fire: float,
    rng: np.random.Generator,
    n_boot: int = B,
) -> np.ndarray:
    """Percentile-bootstrap draws of (LLM recall - random-at-budget recall),
    resampling the blocks in `groups` with replacement, exactly as in
    build_llm_comparison.block_bootstrap / inference_robustness."""
    g = len(groups)
    diffs = np.empty(n_boot)
    m = 0
    for _ in range(n_boot):
        samp = rng.integers(0, g, size=g)
        idx = np.concatenate([groups[i] for i in samp])
        ir = is_red[idx]
        n_red = int(ir.sum())
        if n_red == 0:
            continue
        llm = float((pred_red[idx] & ir).sum()) / n_red
        n = len(idx)
        k = int(round(p_fire * n))
        draw = rng.choice(n, size=k, replace=False)
        rnd = float(ir[draw].sum()) / n_red
        diffs[m] = llm - rnd
        m += 1
    return diffs[:m]


# --------------------------------------------------------------------------- #
# 2b. Fast bootstrap test from per-block sufficient statistics (for sims)
# --------------------------------------------------------------------------- #
def boot_diffs_fast(
    grp_n: np.ndarray,
    grp_red: np.ndarray,
    grp_hits: np.ndarray,
    p_fire: float,
    rng: np.random.Generator,
    n_boot: int = B,
) -> np.ndarray:
    """Same bootstrap distribution as boot_diffs_literal, vectorized.
    grp_n / grp_red / grp_hits: per-block row count, true-reduce count, and
    LLM true-positive count. Random arm drawn hypergeometrically."""
    g = len(grp_n)
    samp = rng.integers(0, g, size=(n_boot, g))
    n = grp_n[samp].sum(axis=1)
    n_red = grp_red[samp].sum(axis=1)
    hits = grp_hits[samp].sum(axis=1)
    valid = n_red > 0
    n, n_red, hits = n[valid], n_red[valid], hits[valid]
    llm = hits / n_red
    k = np.rint(p_fire * n).astype(np.int64)
    rnd_tp = rng.hypergeometric(n_red, n - n_red, k)
    return llm - rnd_tp / n_red


def simulate_power(
    is_red: np.ndarray,
    pred_red: np.ndarray,
    groups: list[np.ndarray],
    p_fire: float,
    delta: float,
    seed_seq: np.random.SeedSequence,
    n_sim: int = N_SIM,
    n_boot: int = B,
) -> float:
    """Fraction of simulations in which the two-sided alpha=0.05 percentile
    bootstrap test rejects H0: diff = 0, given a true injected recall gain
    `delta` (flip round(delta * n_reduce) misses to hits; flip the same number
    of false positives to non-fires so the firing budget stays fixed)."""
    miss_idx = np.flatnonzero(is_red & ~pred_red)   # true reduce, LLM missed
    fp_idx = np.flatnonzero(~is_red & pred_red)     # LLM fired, not reduce
    n_red = int(is_red.sum())
    n_flip = int(round(delta * n_red))
    if n_flip > len(miss_idx) or n_flip > len(fp_idx):
        raise ValueError(f"delta={delta} needs {n_flip} flips; only "
                         f"{len(miss_idx)} misses / {len(fp_idx)} FPs available")
    grp_n = np.array([len(idx) for idx in groups])
    grp_red = np.array([int(is_red[idx].sum()) for idx in groups])
    rejections = 0
    for child in seed_seq.spawn(n_sim):
        rng = np.random.default_rng(child)
        pred = pred_red.copy()
        pred[rng.choice(miss_idx, size=n_flip, replace=False)] = True
        pred[rng.choice(fp_idx, size=n_flip, replace=False)] = False
        grp_hits = np.array([int((pred[idx] & is_red[idx]).sum()) for idx in groups])
        diffs = boot_diffs_fast(grp_n, grp_red, grp_hits, p_fire, rng, n_boot=n_boot)
        lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        if lo > 0 or hi < 0:
            rejections += 1
    return rejections / n_sim


# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_merged()
    n = len(df)
    is_red = df["true_label"].values == "reduce"
    pred_red = df["llm_pred"].values == "reduce"
    p_fire = float(pred_red.mean())
    n_red = int(is_red.sum())
    obs_recall = float((pred_red & is_red).sum()) / n_red

    print(f"[info] n={n}  true_reduce={n_red}  tickers={df['ticker'].nunique()}  "
          f"months={df['date'].nunique()}")
    print(f"[info] LLM reduce recall={obs_recall:.4f}  fire rate p={p_fire:.4f}")

    schemes = {
        "reit_block": make_groups(df, "ticker"),
        "month_block": make_groups(df, "date"),
    }

    rows: list[dict] = []
    se: dict[str, float] = {}

    # ---- 1. bootstrap SE of the difference, both schemes (literal) ----
    for name, groups in schemes.items():
        rng = np.random.default_rng(np.random.SeedSequence([SEED, SCHEME_ID[name]]))
        diffs = boot_diffs_literal(is_red, pred_red, groups, p_fire, rng, n_boot=B)
        se[name] = float(np.std(diffs, ddof=1))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"quantity": "bootstrap_SE_diff", "scheme": name,
                     "value": round(se[name], 4),
                     "note": f"B={B}, seed=[{SEED},{SCHEME_ID[name]}]; "
                             f"mean diff={diffs.mean():.4f}, "
                             f"95% CI [{lo:.3f}, {hi:.3f}]"})
        print(f"[SE]  {name:12s} SE={se[name]:.4f}  mean={diffs.mean():+.4f}  "
              f"CI=[{lo:.3f}, {hi:.3f}]")

    # ---- 2a. analytic MDE ----
    mult = Z_975 + Z_80
    for name in schemes:
        mde_a = mult * se[name]
        rows.append({"quantity": "MDE_analytic_80pct", "scheme": name,
                     "value": round(mde_a, 4),
                     "note": f"(z_0.975 + z_0.80) * SE = {mult:.4f} * {se[name]:.4f}"})
        print(f"[MDE-analytic] {name:12s} {mde_a:.4f}  ({mde_a * 100:.1f} pp)")

    # ---- 2b. simulation MDE ----
    for name, groups in schemes.items():
        mde_sim: float | None = None
        for d_i, delta in enumerate(DELTAS):
            ss = np.random.SeedSequence([SEED, SCHEME_ID[name], d_i])
            power = simulate_power(is_red, pred_red, groups, p_fire, delta, ss)
            rows.append({"quantity": "sim_rejection_rate", "scheme": name,
                         "value": round(power, 3),
                         "note": f"delta={delta} (flip {int(round(delta * n_red))} "
                                 f"misses to hits, budget fixed), {N_SIM} sims, "
                                 f"B={B}, alpha={ALPHA} two-sided, "
                                 f"seed=[{SEED},{SCHEME_ID[name]},{d_i}]"})
            print(f"[power] {name:12s} delta={delta:.3f}  rejection rate={power:.3f}")
            if mde_sim is None and power >= 0.80:
                mde_sim = delta
        rows.append({"quantity": "MDE_simulation_80pct", "scheme": name,
                     "value": mde_sim,
                     "note": f"smallest delta in {DELTAS} with >=80% rejection"})
        print(f"[MDE-sim] {name:12s} {mde_sim}")

    # ---- observed effect, for context (traceable to headline CSV) ----
    hl = pd.read_csv(HEADLINE).set_index("method")["reduce_recall"]
    obs_diff = float(hl["LLM-DeepSeek"] - hl["Random-at-budget"])
    rows.append({"quantity": "observed_diff", "scheme": "point_estimate",
                 "value": round(obs_diff, 4),
                 "note": f"LLM {hl['LLM-DeepSeek']:.4f} - random-at-budget "
                         f"{hl['Random-at-budget']:.4f} "
                         f"(outputs/llm_deepseek_test/headline_comparison.csv)"})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\n[write] {OUT}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
