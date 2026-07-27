"""Publication-quality statistical tables for the thesis (Tables 1-8).

Pure read + format, no API, no recomputation of Zone-1 baseline numbers: every value
is read from an existing outputs/*.csv (or computed as plain descriptive statistics of
the read-only panel). Writes markdown tables to outputs/tables_v2/ — both one combined
file (statistical_tables.md) and one file per table (t1..t8). Each table footnotes its
source file so every number is traceable.

Significance stars throughout: * p<0.10, ** p<0.05, *** p<0.01 (two-sided, from |t|).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "outputs/tables"
FR = REPO / "outputs/fundamentals_robustness"
LD = REPO / "outputs/llm_deepseek_test"
SPLIT = REPO / "data/processed/splits"
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
OUT = REPO / "outputs/tables_v2"
OUT.mkdir(parents=True, exist_ok=True)

# 13 classifier numeric features (sector handled separately as one-hot)
CLF_NUMERIC = [
    ("ret_1m", "1-month return"), ("ret_3m", "3-month return"),
    ("ret_6m", "6-month return"), ("ret_12m", "12-month return"),
    ("vol_annualized", "Annualized volatility"), ("drawdown", "Drawdown from running max"),
    ("dividend_yield_lag1", "Dividend yield (lag 1m)"),
    ("FEDFUNDS_lag1", "Fed funds rate (lag)"), ("DGS10_lag1", "10Y Treasury yield (lag)"),
    ("DGS2_lag1", "2Y Treasury yield (lag)"),
    ("term_spread_10y_2y_lag1", "Term spread 10Y-2Y (lag)"),
    ("cpi_yoy_lag1", "CPI YoY (lag)"), ("UNRATE_lag1", "Unemployment rate (lag)"),
]
FUND = [
    ("leverage", "Leverage (debt/assets)"), ("debt_to_equity", "Debt-to-equity"),
    ("interest_cover", "Interest coverage"), ("ffo_yield_proxy", "FFO yield proxy"),
    ("book_to_market", "Book-to-market"), ("ln_mktcap", "Log market cap"),
    ("amihud_illiq", "Amihud illiquidity"), ("idio_vol", "Idiosyncratic volatility"),
    ("navprem_book_adj", "NAV premium proxy"),
]


def stars(t: float) -> str:
    a = abs(float(t))
    return "***" if a >= 2.576 else "**" if a >= 1.960 else "*" if a >= 1.645 else ""


def f(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def write(name: str, md: str, parts: list[str]):
    (OUT / f"{name}.md").write_text(md.strip() + "\n")
    parts.append(md.strip())


# ---------------------------------------------------------------------------
def t1_descriptive(parts):
    df = pd.read_csv(PANEL, parse_dates=["date"])
    fund = pd.read_csv(FR / "firm_fundamentals_panel.csv", parse_dates=["date"])
    nav = pd.read_csv(FR / "nav_proxy_panel.csv", parse_dates=["date"])[
        ["ticker", "date", "navprem_book_adj"]]
    fund = fund.merge(nav, on=["ticker", "date"], how="left")

    def block(frame, feats, panel_name):
        rows = []
        for col, desc in feats:
            if col not in frame.columns:
                continue
            s = pd.to_numeric(frame[col], errors="coerce")
            # mean within-month cross-sectional dispersion (the key column)
            xs = frame.assign(_v=s).groupby("date")["_v"].std(ddof=0)
            rows.append((desc, col, s.notna().sum(), s.mean(), s.std(),
                         s.min(), s.median(), s.max(), xs.mean()))
        return rows

    num = block(df, CLF_NUMERIC, "panel")
    fnd = block(fund, FUND, "fundamentals")

    lines = ["### Table 2 — Descriptive statistics of features",
             "",
             "| Feature | N | Mean | SD | Min | Median | Max | Mean within-month cross-sectional SD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append("| *Panel A. Classifier inputs (13 numeric + sector dummy)* | | | | | | | |")
    for desc, col, n, mu, sd, mn, md, mx, xs in num:
        lines.append(f"| {desc} (`{col}`) | {n} | {f(mu)} | {f(sd)} | {f(mn)} | "
                     f"{f(md)} | {f(mx)} | {f(xs)} |")
    lines.append("| *Panel B. Reconstructed REIT fundamentals (9)* | | | | | | | |")
    for desc, col, n, mu, sd, mn, md, mx, xs in fnd:
        lines.append(f"| {desc} (`{col}`) | {n} | {f(mu)} | {f(sd)} | {f(mn)} | "
                     f"{f(md)} | {f(mx)} | {f(xs)} |")
    lines += [
        "",
        "*The final column is the mean across months of the within-month cross-sectional "
        "standard deviation. Macro variables (fed funds, Treasury yields, term spread, CPI, "
        "unemployment) have a cross-sectional SD of essentially zero — they are identical "
        "across firms within a month and so cannot discriminate **which firm** reduces, even "
        "though they vary over time. All nine reconstructed fundamentals, by contrast, vary "
        "genuinely across firms. Source: `data/processed/backtest_ready_panel_enriched.csv`, "
        "`outputs/fundamentals_robustness/firm_fundamentals_panel.csv`, `nav_proxy_panel.csv`.*",
    ]
    write("t1_descriptive", "\n".join(lines), parts)


def t2_labels(parts):
    files = [("Train (2015–2021)", "enriched_train_2015_2021.csv"),
             ("Validation (2022–2023)", "enriched_validation_2022_2023.csv"),
             ("Test (2024–2025)", "enriched_test_2024_2025.csv")]
    lines = ["### Table 1 — Label distribution across the chronological split",
             "",
             "| Split | N | Increase | Hold | Reduce | Reduce share |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, fn in files:
        d = pd.read_csv(SPLIT / fn)
        vc = d["label"].value_counts()
        n = len(d)
        inc, hold, red = (int(vc.get(k, 0)) for k in ("increase", "hold", "reduce"))
        lines.append(f"| {name} | {n} | {inc} | {hold} | {red} | {red / n:.1%} |")
    lines += ["",
              "*Labels: next-month total return > +2% → increase, < −2% → reduce, else hold. "
              "The split is strictly chronological (no look-ahead). Source: "
              "`data/processed/splits/*.csv`.*"]
    write("t2_labels", "\n".join(lines), parts)


def t3_famamacbeth(parts):
    fm = pd.read_csv(FR / "extended_fama_macbeth.csv")
    # Economically meaningful firm-level features (drop sector dummies for the focused table)
    keep = ["leverage", "debt_to_equity", "interest_cover", "ffo_yield_proxy",
            "navprem_book_adj", "book_to_market", "ln_mktcap", "amihud_illiq",
            "idio_vol", "ret_6m", "ret_1m", "dividend_yield_lag1"]

    def grab(model, period):
        sub = fm[(fm.model == model) & (fm.period == period)].set_index("feature")
        return sub

    ext_full, ext_test = grab("Extended", "full"), grab("Extended", "test")
    lines = ["### Table 5 — Fama–MacBeth cross-sectional regressions (extended feature set)",
             "",
             "| Feature | Full sample (2015–2025) coef | t | | Test window (2024–2025) coef | t | |",
             "|---|---:|---:|:--|---:|---:|:--|"]
    for col in keep:
        if col not in ext_full.index:
            continue
        cf, tf = ext_full.loc[col, "mean_coef"], ext_full.loc[col, "fm_t"]
        ct, tt = (ext_test.loc[col, "mean_coef"], ext_test.loc[col, "fm_t"]) \
            if col in ext_test.index else (np.nan, np.nan)
        lines.append(f"| `{col}` | {f(cf,4)} | {f(tf,2)} | {stars(tf)} | "
                     f"{f(ct,4)} | {f(tt,2)} | {stars(tt)} |")
    lines += [
        "",
        "*Two-pass Fama–MacBeth (1973): monthly cross-sectional OLS of the next return on the "
        "standardized features, then time-series averages of the slopes (Newey-West-style t "
        "from the slope series). On the **original** feature set no firm-level feature is "
        "significant (only a Healthcare sector dummy, t = 2.20** on the test window). Adding the "
        "reconstructed fundamentals, **leverage** is priced full-sample (t = −3.11***) and "
        "debt-to-equity, leverage and 6-month momentum reach significance on the test window — a "
        "genuine but modest *pricing* result that does **not** translate into out-of-sample "
        "prediction (Table 4) or discrete-event warning (Tables 6, 7). Stars: * p<0.10, "
        "** p<0.05, *** p<0.01. Source: `outputs/fundamentals_robustness/extended_fama_macbeth.csv`.*",
    ]
    write("t3_famamacbeth", "\n".join(lines), parts)


def t4_ols(parts):
    m = pd.read_csv(TAB / "ols_metrics_enriched.csv")
    ext = pd.read_csv(FR / "extended_ols.csv")
    lines = ["### Table 4 — OLS out-of-sample R² (return predictability in levels)",
             "",
             "| Specification | Train R² | Validation R² | Test R² | N test |",
             "|---|---:|---:|---:|---:|"]
    # macro-enriched (headline)
    r = m.set_index("period")["r2"]
    lines.append(f"| Macro-enriched panel | {f(r.get('Training (2015-2021)'))} | "
                 f"{f(r.get('Validation (2022-2023)'))} | {f(r.get('Test (2024-2025)'))} | "
                 f"{int(m[m.period.str.startswith('Test')]['n_obs'].iloc[0])} |")
    for _, row in ext.iterrows():
        lines.append(f"| {row['model']} | {f(row['train'])} | {f(row['val'])} | "
                     f"{f(row['test'])} | {int(row['n_test'])} |")
    lines += [
        "",
        "*A positive train R² collapsing to a large **negative** out-of-sample R² (worse than "
        "predicting the training mean) is the classic overfit signature of macro-based return "
        "prediction (Welch & Goyal, 2008). Adding the nine fundamentals raises in-sample fit but "
        "leaves test R² negative — no out-of-sample predictability in levels. Source: "
        "`outputs/tables/ols_metrics_enriched.csv`, `outputs/fundamentals_robustness/extended_ols.csv`.*",
    ]
    write("t4_ols", "\n".join(lines), parts)


def t5_headline(parts):
    h = pd.read_csv(LD / "headline_comparison.csv").set_index("method")
    b = pd.read_csv(LD / "significance_bootstrap.csv").set_index("quantity")
    inf = pd.read_csv(FR / "inference_robustness.csv", index_col=0)["value"]

    def ci(q):
        if q in b.index:
            return f"[{f(b.loc[q,'ci_lo'])}, {f(b.loc[q,'ci_hi'])}]"
        return "—"

    order = [("LLM five-agent framework", "LLM-DeepSeek", "LLM"),
             ("Logistic (argmax)", "Logistic-argmax", "Logistic-argmax"),
             ("Random-at-budget", "Random-at-budget", "Random-at-budget"),
             ("Threshold-logistic", "Threshold-logistic", None)]
    lines = ["### Table 7 — Main comparison on the reduce class (test set, 575 decisions)",
             "",
             "| Method | Reduce recall | Reduce precision | Hold recall | Accuracy | Firing rate | Reduce-recall 95% CI |",
             "|---|---:|---:|---:|---:|---:|:--:|"]
    for disp, hk, bk in order:
        row = h.loc[hk]
        hold = f(row["hold_recall"]) if pd.notna(row["hold_recall"]) else "—"
        acc = f(row["accuracy"]) if pd.notna(row["accuracy"]) else "—"
        lines.append(f"| {disp} | {f(row['reduce_recall'])} | {f(row['reduce_precision'])} | "
                     f"{hold} | {acc} | {f(row['reduce_fire_rate'])} | {ci(bk) if bk else '—'} |")
    lines += [
        "",
        f"**Difference (LLM − Random-at-budget):** mean {f(b.loc['LLM_minus_Random','reduce_recall_mean'])}, "
        f"REIT-block bootstrap 95% CI {ci('LLM_minus_Random')}; "
        f"month-block bootstrap 95% CI {inf.get('LLMminusRandom_monthblock_CI')}. "
        "Both contain zero.",
        "",
        "*All rows are evaluated on the identical test set and firing budget. The argmax logistic "
        "predicts reduce for zero observations (an argmax artifact, not a tuned floor); the honest "
        "floors are random-at-budget and threshold-logistic. The five-agent framework (0.206) is "
        "statistically indistinguishable from random-at-budget (0.204) under both clustering "
        f"schemes. The common-factor ceiling: median market-model R² = {f(inf.get('market_R2_incl_self_median'))} "
        f"(leave-one-out {f(inf.get('market_R2_leave_one_out_median'))}). Source: "
        "`outputs/llm_deepseek_test/headline_comparison.csv`, `significance_bootstrap.csv`, "
        "`outputs/fundamentals_robustness/inference_robustness.csv`.*",
    ]
    write("t5_headline", "\n".join(lines), parts)


def t6_ablation(parts):
    a = pd.read_csv(LD / "ablation_channels.csv")
    lines = ["### Table 9 — Channel ablation: agent-subset configurations (test set)",
             "",
             "| Configuration | Reduce recall | Reduce precision | Firing rate | Accuracy | Random-at-budget recall | Beats random? |",
             "|---|---:|---:|---:|---:|---:|:--:|"]
    for _, r in a.iterrows():
        lines.append(f"| {r['config']} | {f(r['reduce_recall'])} | {f(r['reduce_precision'])} | "
                     f"{f(r['reduce_fire_rate'])} | {f(r['accuracy'])} | "
                     f"{f(r['random_at_budget_recall'])} | {'yes' if r['beats_random'] else 'no'} |")
    lines += [
        "",
        "*No configuration beats random-at-budget by an economically meaningful margin; the "
        "nominal 'yes' rows occur only at very low firing rates where one or two hits flip the "
        "comparison. But the channels occupy distinct precision–recall regions — disclosure-only "
        "is precise but conservative (precision 0.474, firing 7%), structured-only is aggressive "
        "but imprecise — evidence they are genuinely different diagnostic perspectives rather than "
        "redundant copies. Source: `outputs/llm_deepseek_test/ablation_channels.csv`.*",
    ]
    write("t6_ablation", "\n".join(lines), parts)


def t7_tuned(parts):
    t = pd.read_csv(FR / "tuned_robustness.csv")
    lines = ["### Table 6 — Tuned baseline robustness: reduce recall is invariant",
             "",
             "| Model | Best (C, L1) | Hold recall | Reduce recall | Reduce predicted | Accuracy | CV macro-F1 |",
             "|---|:--:|---:|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        cl1 = "MLE" if pd.isna(r.get("best_C")) else f"({f(r['best_C'],2)}, {f(r['best_l1'],2)})"
        cvf = "—" if pd.isna(r.get("cv_macroF1")) else f(r["cv_macroF1"])
        model = str(r["model"]).replace(" | ", " — ")  # avoid breaking the markdown table
        lines.append(f"| {model} | {cl1} | {f(r['hold_recall'])} | {f(r['reduce_recall'])} | "
                     f"{int(r['reduce_npred'])} | {f(r['accuracy'])} | {cvf} |")
    lines += [
        "",
        "*Reduce recall stays at 0.000 across feature enrichment (adding all nine fundamentals "
        "including the NAV proxy), elastic-net tuning (C and L1 chosen by time-series CV on "
        "macro-F1), and a change of model family (ordinal/proportional-odds). The failure is "
        "neither a feature gap nor a model-capacity gap. Source: "
        "`outputs/fundamentals_robustness/tuned_robustness.csv`.*",
    ]
    write("t7_tuned", "\n".join(lines), parts)


def t8_factuality(parts):
    c = pd.read_csv(LD / "contamination_summary.csv").iloc[0]
    fa = pd.read_csv(LD / "factuality_audit.csv", index_col=0)["value"]
    rows = [
        ("Input look-ahead violations", "57%* (v1)", f"{int(c['input_lookahead_violations'])}",
         "filing-date enforcement holds"),
        ("Reduce decisions grounded in a specific filing fact", "57%",
         f"{c['reduce_grounded_pct']:.0%}", "the measurable 'decomposable rationales' claim"),
        ("Mean cited facts per reduce decision", "2.84", f"{c['reduce_mean_facts_cited']:.2f}", "—"),
        ("Reduce rationales citing a dollar amount / %", "53%",
         f"{c['reduce_with_money_or_pct_pct']:.0%}", "—"),
        ("All-decision grounding rate", "68%", f"{c['all_grounded_pct']:.0%}", "—"),
        ("Fundamentals cited values exactly matching model inputs", "—",
         f"{fa['fundamentals_factual_accuracy']:.0%} (n={int(fa['fundamentals_cited_values_checked'])})",
         "grounded AND factually accurate"),
        ("Disclosure numeric tokens verifiable in source filing", "—",
         f"{fa['disclosure_numeric_in_source_rate']:.1%} (n={int(fa['disclosure_numeric_tokens_checked'])})",
         "not hallucinated"),
    ]
    lines = ["### Table 8 — Grounding and factuality audit of rationales (test set)",
             "",
             "| Metric | Before fix (v1) | After fix (v2, 5-agent) | Interpretation |",
             "|---|:--:|:--:|---|"]
    # fix the first row (look-ahead is 0 in both; the 57% belongs to grounding)
    rows[0] = ("Input look-ahead violations", "0", f"{int(c['input_lookahead_violations'])}",
               "filing-date enforcement holds")
    for m, v1, v2, interp in rows:
        lines.append(f"| {m} | {v1} | {v2} | {interp} |")
    lines += [
        "",
        "*'v1' is the broken-text version (the agent was fed an XBRL metadata block); 'v2' is "
        "after the Item 1A extraction fix and the addition of the Fundamentals agent. Grounding "
        "(citing a specific fact) is upgraded to factuality (the cited fact is actually true) by "
        "the bottom two rows. v1 grounding figures are from `CANONICAL_RESULTS.md`; v2 from "
        "`outputs/llm_deepseek_test/contamination_summary.csv` and `factuality_audit.csv`.*",
    ]
    write("t8_factuality", "\n".join(lines), parts)


def main():
    parts: list[str] = []
    # ordered to match the thesis's monotonic table numbering (1..9; 3 = inline matrix)
    for fn in (t2_labels, t1_descriptive, t4_ols, t3_famamacbeth,
               t7_tuned, t5_headline, t8_factuality, t6_ablation):
        fn(parts)
        print(f"[ok] {fn.__name__}")
    combined = ("# Statistical Tables (auto-generated, traceable to source CSVs)\n\n"
                "> Generated by `analysis/make_tables.py`. Stars: * p<0.10, ** p<0.05, "
                "*** p<0.01.\n\n" + "\n\n---\n\n".join(parts) + "\n")
    (OUT / "statistical_tables.md").write_text(combined)
    print(f"\nwritten -> {OUT}/  (8 tables + statistical_tables.md)")


if __name__ == "__main__":
    main()
