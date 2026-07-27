"""Economic-value and discrimination/calibration evaluation of the LLM framework.

Converts the classification comparison into the three evaluations a finance
referee expects beyond recall/precision:

1. PORTFOLIO VALUE — monthly-rebalanced, equal-weight "avoid" portfolios over the
   23 test months (2024-01..2025-11): hold every universe name EXCEPT those flagged
   reduce by (a) the LLM framework, (b) the threshold-logistic baseline at the SAME
   per-month budget, (c) a random screen at the same budget (Monte-Carlo), and
   (d) the oracle that avoids the true reduce names (upper bound). Reports annualised
   mean/vol, Sharpe (Lo 2002 iid SE), max drawdown, turnover, net-of-cost Sharpe,
   mean-variance certainty-equivalent difference vs the EW benchmark (gamma = 3,
   Campbell & Thompson 2008 convention), and iid month-bootstrap CIs for the
   mean/Sharpe differences.

2. DISCRIMINATION — threshold-free ROC-AUC and PR-AUC of the reduce probability
   (LLM vs logistic), with month-block bootstrap CIs and difference CIs.

3. CALIBRATION — 10-bin reliability curves and ECE for both models' reduce
   probabilities (figure input; complements the Brier/Murphy decomposition).

Inputs (read-only): audit_log/predictions.csv,
  outputs/tables/quant_only_test_predictions.csv (Zone 1, read-only),
  data/processed/backtest_ready_panel.csv (Zone 1, read-only; monthly_rf).
Outputs: outputs/fundamentals_robustness/{portfolio_value.csv, auc_metrics.csv,
  calibration_curve.csv}.

Deterministic: all Monte Carlo uses np.random.default_rng(42).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "fundamentals_robustness"
GAMMA = 3.0          # CRRA-style mean-variance risk aversion (Campbell-Thompson 2008)
COST_BPS = 25.0      # one-way transaction cost, basis points, for the net row
N_BOOT = 10_000
N_RANDOM = 2_000
RNG = np.random.default_rng(42)


def load_joined() -> pd.DataFrame:
    pred = pd.read_csv(REPO / "audit_log" / "predictions.csv")
    logit = pd.read_csv(REPO / "outputs" / "tables" / "quant_only_test_predictions.csv")
    rf = (pd.read_csv(REPO / "data" / "processed" / "backtest_ready_panel.csv",
                      usecols=["date", "monthly_rf"])
          .drop_duplicates("date"))
    df = pred.merge(
        logit[["date", "ticker", "pred_proba_reduce", "future_ret_1m"]],
        left_on=["decision_date", "ticker"], right_on=["date", "ticker"],
        how="inner", validate="1:1",
    ).merge(rf, on="date", how="left")
    assert len(df) == 575, f"join lost rows: {len(df)}"
    assert df["future_ret_1m"].notna().all() and df["monthly_rf"].notna().all()
    df["y"] = (df["true_label"] == "reduce").astype(int)
    df["llm_flag"] = (df["predicted_label"] == "reduce").astype(int)
    return df


# --------------------------- portfolio construction --------------------------- #

def monthly_series(df: pd.DataFrame, held: pd.Series) -> pd.DataFrame:
    """Equal-weight next-month return of held names, per decision month."""
    d = df[held.astype(bool)]
    grp = d.groupby("decision_date")
    out = grp["future_ret_1m"].mean().rename("ret").to_frame()
    out["n_held"] = grp.size()
    out["rf"] = df.drop_duplicates("decision_date").set_index("decision_date")["monthly_rf"]
    return out.sort_index()


def turnover(df: pd.DataFrame, held: pd.Series) -> float:
    """Mean month-over-month one-way turnover of equal-weight holdings."""
    months = sorted(df["decision_date"].unique())
    prev: set[str] | None = None
    tos: list[float] = []
    for m in months:
        cur = set(df.loc[(df["decision_date"] == m) & held.astype(bool), "ticker"])
        if prev is not None and cur:
            union_universe = df.loc[df["decision_date"] == m, "ticker"]
            # weight change lower bound: names entering + leaving, over 2
            w_prev = {t: 1 / len(prev) for t in prev}
            w_cur = {t: 1 / len(cur) for t in cur}
            names = set(w_prev) | set(w_cur)
            tos.append(sum(abs(w_cur.get(t, 0.0) - w_prev.get(t, 0.0)) for t in names) / 2)
        prev = cur
    return float(np.mean(tos)) if tos else 0.0


def perf(ts: pd.DataFrame, to: float) -> dict[str, float]:
    ex = ts["ret"] - ts["rf"]
    mu_a, sd_a = ex.mean() * 12, ex.std(ddof=1) * np.sqrt(12)
    sr = mu_a / sd_a if sd_a > 0 else np.nan
    t_ = len(ex)
    sr_se = np.sqrt((1 + sr**2 / 2) / t_)          # Lo (2002), iid, annualised units
    cum = (1 + ts["ret"]).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    cost_drag_a = 2 * to * (COST_BPS / 1e4) * 12   # round-trip per rebalance
    sr_net = (mu_a - cost_drag_a) / sd_a if sd_a > 0 else np.nan
    cer_a = mu_a - GAMMA / 2 * sd_a**2
    return {"ann_ret": ts["ret"].mean() * 12, "ann_excess": mu_a, "ann_vol": sd_a,
            "sharpe": sr, "sharpe_se_lo2002": sr_se, "max_drawdown": mdd,
            "turnover_1way": to, "sharpe_net_25bps": sr_net, "cer": cer_a,
            "n_months": t_}


def boot_diff(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """iid month-bootstrap CI for mean- and Sharpe-difference (a minus b)."""
    ea = (a["ret"] - a["rf"]).to_numpy()
    eb = (b["ret"] - b["rf"]).to_numpy()
    t_ = len(ea)
    idx = RNG.integers(0, t_, size=(N_BOOT, t_))
    da, db = ea[idx], eb[idx]
    dmu = (da.mean(1) - db.mean(1)) * 12
    with np.errstate(invalid="ignore"):
        dsr = (da.mean(1) / da.std(1, ddof=1) - db.mean(1) / db.std(1, ddof=1)) * np.sqrt(12)
    return {"dmean_lo": float(np.nanpercentile(dmu, 2.5)),
            "dmean_hi": float(np.nanpercentile(dmu, 97.5)),
            "dsharpe_lo": float(np.nanpercentile(dsr, 2.5)),
            "dsharpe_hi": float(np.nanpercentile(dsr, 97.5))}


def matched_budget_flags(df: pd.DataFrame, score: str) -> pd.Series:
    """Flag the k_t highest-`score` names per month, k_t = LLM flag count that month."""
    flags = pd.Series(0, index=df.index)
    for m, g in df.groupby("decision_date"):
        k = int(g["llm_flag"].sum())
        if k:
            flags.loc[g[score].nlargest(k).index] = 1
    return flags


def random_avoid_mc(df: pd.DataFrame) -> dict[str, float]:
    """MC distribution of ann. excess mean and Sharpe for random same-budget screens."""
    mus, srs = [], []
    months = [g for _, g in df.groupby("decision_date")]
    for _ in range(N_RANDOM):
        rets, rfs = [], []
        for g in months:
            k = int(g["llm_flag"].sum())
            keep = g.index if k == 0 else RNG.permutation(g.index)[: len(g) - k]
            rets.append(g.loc[keep, "future_ret_1m"].mean())
            rfs.append(g["monthly_rf"].iloc[0])
        ex = np.array(rets) - np.array(rfs)
        mus.append(ex.mean() * 12)
        srs.append(ex.mean() / ex.std(ddof=1) * np.sqrt(12))
    return {"ann_excess": float(np.mean(mus)), "sharpe": float(np.mean(srs)),
            "sharpe_p2.5": float(np.percentile(srs, 2.5)),
            "sharpe_p97.5": float(np.percentile(srs, 97.5))}


# ------------------------------ AUC + calibration ----------------------------- #

def auc_pair(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def auc_with_month_boot(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"].to_numpy()
    rows = []
    months = df["decision_date"].to_numpy()
    uniq = np.unique(months)
    month_idx = {m: np.flatnonzero(months == m) for m in uniq}
    draws = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    scores = {"llm": df["prob_reduce"].to_numpy(),
              "logit": df["pred_proba_reduce"].to_numpy()}
    boot: dict[str, dict[str, list[float]]] = {k: {"roc": [], "pr": []} for k in scores}
    for d in draws:
        idx = np.concatenate([month_idx[uniq[j]] for j in d])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, s in scores.items():
            r, p = auc_pair(yb, s[idx])
            boot[k]["roc"].append(r)
            boot[k]["pr"].append(p)
    for k, s in scores.items():
        roc, pr = auc_pair(y, s)
        rows.append({"model": k, "roc_auc": roc, "pr_auc": pr,
                     "roc_lo": np.percentile(boot[k]["roc"], 2.5),
                     "roc_hi": np.percentile(boot[k]["roc"], 97.5),
                     "pr_lo": np.percentile(boot[k]["pr"], 2.5),
                     "pr_hi": np.percentile(boot[k]["pr"], 97.5)})
    droc = np.array(boot["llm"]["roc"]) - np.array(boot["logit"]["roc"])
    rows.append({"model": "llm_minus_logit",
                 "roc_auc": rows[0]["roc_auc"] - rows[1]["roc_auc"],
                 "pr_auc": rows[0]["pr_auc"] - rows[1]["pr_auc"],
                 "roc_lo": np.percentile(droc, 2.5), "roc_hi": np.percentile(droc, 97.5),
                 "pr_lo": np.nan, "pr_hi": np.nan})
    rows.append({"model": "base_rate", "roc_auc": 0.5, "pr_auc": float(y.mean()),
                 "roc_lo": np.nan, "roc_hi": np.nan, "pr_lo": np.nan, "pr_hi": np.nan})
    return pd.DataFrame(rows)


def dm_hln(df: pd.DataFrame) -> pd.DataFrame:
    """Diebold-Mariano (1995) tests on monthly Brier-loss differentials with the
    Harvey-Leybourne-Newbold (1997) small-sample correction (h = 1)."""
    from scipy import stats
    d = df.copy()
    base = d["y"].mean()
    d["l_llm"] = (d["prob_reduce"] - d["y"]) ** 2
    d["l_logit"] = (d["pred_proba_reduce"] - d["y"]) ** 2
    d["l_base"] = (base - d["y"]) ** 2
    m = d.groupby("decision_date")[["l_llm", "l_logit", "l_base"]].mean()
    t_ = len(m)
    rows = []
    for name, a, b in [("llm_vs_base", "l_llm", "l_base"),
                       ("llm_vs_logit", "l_llm", "l_logit"),
                       ("logit_vs_base", "l_logit", "l_base")]:
        diff = (m[a] - m[b]).to_numpy()
        dm = diff.mean() / (diff.std(ddof=1) / np.sqrt(t_))
        hln = dm * np.sqrt((t_ - 1) / t_)
        p = 2 * (1 - stats.t.cdf(abs(hln), df=t_ - 1))
        rows.append({"pair": name, "mean_loss_diff": diff.mean(),
                     "dm_hln_t": hln, "p_value": p, "n_months": t_})
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0, 1, 11)
    for model, col in [("llm", "prob_reduce"), ("logit", "pred_proba_reduce")]:
        s, y = df[col].to_numpy(), df["y"].to_numpy()
        which = np.clip(np.digitize(s, bins) - 1, 0, 9)
        ece = 0.0
        for b in range(10):
            m = which == b
            if not m.any():
                continue
            conf, acc, n = s[m].mean(), y[m].mean(), int(m.sum())
            ece += n / len(s) * abs(acc - conf)
            rows.append({"model": model, "bin_lo": bins[b], "bin_hi": bins[b + 1],
                         "n": n, "mean_forecast": conf, "observed_freq": acc})
        rows.append({"model": model, "bin_lo": np.nan, "bin_hi": np.nan, "n": len(s),
                     "mean_forecast": np.nan, "observed_freq": np.nan, "ece": ece})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_joined()
    months = df["decision_date"].nunique()
    print(f"joined {len(df)} decisions over {months} months; "
          f"LLM flags {int(df.llm_flag.sum())} ({df.llm_flag.mean():.1%})")

    held_all = pd.Series(1, index=df.index)
    held_llm = 1 - df["llm_flag"]
    held_log = 1 - matched_budget_flags(df, "pred_proba_reduce")
    held_orc = 1 - df["y"]

    port_rows = []
    ts_all = monthly_series(df, held_all)
    for name, held in [("ew_all", held_all), ("llm_avoid", held_llm),
                       ("logit_avoid", held_log), ("oracle_avoid", held_orc)]:
        ts = monthly_series(df, held)
        row = {"portfolio": name, **perf(ts, turnover(df, held))}
        if name != "ew_all":
            row.update(boot_diff(ts, ts_all))
            row["dcer"] = row["cer"] - port_rows[0]["cer"]
        port_rows.append(row)
    rnd = random_avoid_mc(df)
    port_rows.append({"portfolio": "random_avoid_mc", **rnd})
    port = pd.DataFrame(port_rows)
    port.to_csv(OUT / "portfolio_value.csv", index=False)
    print(port.round(4).to_string(index=False))

    auc = auc_with_month_boot(df)
    auc.to_csv(OUT / "auc_metrics.csv", index=False)
    print(auc.round(4).to_string(index=False))

    dm = dm_hln(df)
    dm.to_csv(OUT / "dm_test.csv", index=False)
    print(dm.round(4).to_string(index=False))

    cal = calibration(df)
    cal.to_csv(OUT / "calibration_curve.csv", index=False)
    print(cal[cal["ece"].notna()][["model", "ece"]].round(4).to_string(index=False)
          if "ece" in cal else cal.head())


if __name__ == "__main__":
    main()
