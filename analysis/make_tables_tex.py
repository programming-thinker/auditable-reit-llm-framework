"""
NOTE (2026-07-02 sync): the shipped paper/_tables/*.tex additionally carry manual
note cleanups that MUST be preserved if regenerating: (1) no "Sources: ..." file
paths in table notes (provenance lives in Appendix C); (2) the significance-stars
note only on tables that print stars (table3); (3) British spelling ("artefact");
(4) table3 note reads "for terms classically significant at the 5% level or
better". (5) table4: the difference row shows the point-estimate difference 0.002 (not the
bootstrap-mean 0.001) and its notes explain both plus the Monte-Carlo nature of the
random-at-budget recall; (6) table5: notes carry the same Monte-Carlo sentence;
(7) table7: the provenance sentence with file paths is replaced by a pointer to
Appendix C. (8) table1: row labels carry NO \texttt mnemonics (they live in Table A1) — width fix;
(9) table2: Panel B labels shortened ("Tuned logit, extended (+9 fundamentals)" etc.)
and the Panel A header rule is a full-width \midrule; (10) table5: compact headers
(Recall / Precision / Random recall) with meanings in the notes; (11) tableA1: float
spec [H] and Source cells "XBRL, prices" (no breaking "+"). Diff against the shipped
.tex before overwriting.
NOTE (2026-07-08 polish sync, implemented IN THIS CODE — do not revert): (12) table3
Panel B row labels are readable names via fm_names (no \texttt mnemonics; the note
points to Table A1); (13) all numeric columns are right-aligned (colspecs lrr.../
p{...}rr), matching the shipped .tex; keep them right-aligned if regenerating.
NOTE (2026-07-08 final polish, IN THIS CODE): (14) no \texttt in any row label or
column header of tables 1-8 (mnemonics live only in Table A1; notes may keep \texttt
for file-free technical terms such as TimeSeriesSplit); (15) header rules that span
every column are \midrule, \cmidrule(r){1-k} only for partial spans (table1 Panel B
header rule is \midrule); (16) \addlinespace directly after every panel-title line;
en-dashes (--) for all numeric/text ranges; minus signs uniformly as math-mode $-$
(never \textminus or a text hyphen).
NOTE (2026-07-08, remaining shipped-vs-generated BODY deltas — manual, re-apply
after any regeneration): (17) table2 Panel A header is the one-line
"True label & Pred.\ increase & Pred.\ hold & Pred.\ reduce & Recall &" + \midrule
(not the two-row "Predicted" spanner) and the Panel B column is "$n$ reduce pred.";
(18) table4 difference row prints the point-estimate difference $0.002$, not the
bootstrap-mean $0.001$ this code reads from significance_bootstrap.csv; (19) table5
header has no tnote on "Random recall", \tnote{a} sits on "$\Delta$ hits", and the
notes end with "\item Random recall is a seeded Monte-Carlo estimate."; (20) tableA1
mnemonics are plain \texttt (no \allowbreak).

NOTE (2026-07-08 econ-value round, SHIPPED-ONLY manual tables — not yet generated
here): (21) table4 gained "Panel A. Classification at matched budget" header and a
new "Panel B. Threshold-free discrimination" (ROC-AUC / PR-AUC with month-block
bootstrap CIs) sourced from outputs/fundamentals_robustness/auc_metrics.csv, plus a
Panel B sentence in the notes; (22) table7 was REPLACED by the portfolio
economic-value table (source: outputs/fundamentals_robustness/portfolio_value.csv,
analysis/econ_value_eval.py, seed 42); (23) table8 now MERGES the former table7
(grounding/factuality) and table8 (judge + human calibration) as Panels A-C; the
former standalone grounding table is retired; (24) tableA1 moved to
THESIS_SUPPLEMENT.md with the appendices. Regenerating tables 4/7/8 from this
script will NOT reproduce the shipped files until items (21)-(23) are ported.
Journal-grade LaTeX tables for the thesis (booktabs + threeparttable).

Writes paper/_tables/table1.tex .. table8.tex and tableA1.tex. Pure read + format:
every number is read from an existing outputs/*.csv or computed as plain descriptive
statistics of the read-only panel (no re-estimation of any Zone-1 result). Where a
value exists only in the machine-verified draft prose / CANONICAL_RESULTS.md, it is
hardcoded here with a comment citing the source file.

Formatting conventions (journal style):
  * \begin{table}[t] + \centering + threeparttable, caption above the tabular
  * no vertical rules; \toprule/\midrule/\bottomrule only
  * decimal alignment via fixed-format strings in right-aligned columns
  * recall/precision 3 dp; R^2 2 dp (3 dp where 2 dp would destroy information,
    e.g. the firm-features R^2 = 0.007 in Table 6); t-statistics 2 dp in
    parentheses under coefficients; stars from the classical t (|t| >= 1.645 /
    1.960 / 2.576 for 10% / 5% / 1%)
  * every table carries flushleft tablenotes beginning "Notes: This table reports ..."
    and the standard significance-star note.

Run:  python3 analysis/make_tables_tex.py
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "outputs/tables"                       # Zone 1, read-only
FR = REPO / "outputs/fundamentals_robustness"
LD = REPO / "outputs/llm_deepseek_test"
SPLIT = REPO / "data/processed/splits"              # Zone 1, read-only
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"  # Zone 1, read-only
OUT = REPO / "paper/_tables"
OUT.mkdir(parents=True, exist_ok=True)

DASH = r"---"  # em-dash placeholder for empty cells

# ---------------------------------------------------------------- formatting --
def dec(x: float, n: int) -> str:
    """Fixed-format with ROUND_HALF_UP so 1.965 -> 1.97 (matches the draft)."""
    q = Decimal(1).scaleb(-n) if n > 0 else Decimal(1)
    return str(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def num(x: float, n: int = 3) -> str:
    """Math-mode fixed-format number (clean minus sign)."""
    return f"${dec(x, n)}$"


def pct(x: float, n: int = 1) -> str:
    return dec(100.0 * float(x), n) + r"\%"


def stars(t: float) -> str:
    a = abs(float(t))
    s = "***" if a >= 2.576 else "**" if a >= 1.960 else "*" if a >= 1.645 else ""
    return s


def coef(c: float, t: float, n: int = 4) -> str:
    # f-string (float round-half-even) formatting, matching the machine-verified
    # draft's Table 5 exactly (e.g. -0.00605 -> -0.0060, not -0.0061)
    s = stars(t)
    return f"${float(c):.{n}f}" + (f"^{{{s}}}$" if s else "$")


MODEL_NAMES = {
    "TUNED logit | Original 13+sector": "Tuned logit, original 13 $+$ sector",
    "TUNED logit | Extended +9 fundamentals (incl NAV)":
        "Tuned logit, extended $+$ 9 fundamentals",
    "ORDERED logit | Extended +9 fundamentals":
        "Ordered logit, extended $+$ 9 fundamentals",
    "Original 13+sector": "Original 13 $+$ sector",
    "Extended +9 fundamentals": "Extended $+$ 9 fundamentals",
}


def code(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def big_n(x: float) -> str:
    """Math-mode thousands separator: 2995 -> 2{,}995."""
    return f"{int(x):,}".replace(",", "{,}")


def code_br(name: str) -> str:
    """Breakable \\texttt mnemonic for narrow p-columns (break after underscores)."""
    return r"\texttt{" + name.replace("_", r"\_\allowbreak ") + "}"


def panel(text: str, ncols: int) -> str:
    return rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{text}}}}}\\"


SIG_NOTE = (r"\item *, **, and *** denote significance at the 10\%, 5\%, and 1\% "
            r"levels.")


def write_table(name: str, caption: str, colspec: str, body: list[str],
                notes: list[str], size: str = r"\footnotesize",
                caption_star: bool = False, label: str | None = None) -> None:
    cap = r"\caption*" if caption_star else r"\caption"
    lab = rf"\label{{{label}}}" if label else ""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{threeparttable}",
        rf"{cap}{{{caption}}}{lab}",
        size,
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        *notes,
        SIG_NOTE,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {name}.tex")


# ------------------------------------------------------------------- Table 1 --
# 13 classifier numeric features + 9 reconstructed fundamentals (labels as in the
# draft's Table 2, source outputs/tables/v6_selected_features.csv for descriptions)
CLF_NUMERIC = [
    ("ret_1m", "1-month return"), ("ret_3m", "3-month return"),
    ("ret_6m", "6-month return"), ("ret_12m", "12-month return"),
    ("vol_annualized", "Annualized volatility"), ("drawdown", "Drawdown"),
    ("dividend_yield_lag1", "Dividend yield, lag"),
    ("FEDFUNDS_lag1", "Fed funds rate, lag"), ("DGS10_lag1", "10Y Treasury, lag"),
    ("DGS2_lag1", "2Y Treasury, lag"),
    ("term_spread_10y_2y_lag1", "Term spread, lag"),
    ("cpi_yoy_lag1", "CPI YoY, lag"), ("UNRATE_lag1", "Unemployment, lag"),
]
FUND = [
    ("leverage", "Leverage"), ("debt_to_equity", "Debt-to-equity"),
    ("interest_cover", "Interest coverage"), ("ffo_yield_proxy", "FFO yield proxy"),
    ("book_to_market", "Book-to-market"), ("ln_mktcap", "Log market cap"),
    ("amihud_illiq", "Amihud illiquidity"), ("idio_vol", "Idiosyncratic vol"),
    ("navprem_book_adj", "NAV premium proxy"),
]


def desc_rows(frame: pd.DataFrame, feats: list[tuple[str, str]]) -> list[str]:
    rows = []
    for col, desc in feats:
        s = pd.to_numeric(frame[col], errors="coerce")
        xs = frame.assign(_v=s).groupby("date")["_v"].std(ddof=0)  # within-month CS SD
        rows.append(  # mnemonics are defined in Table A1 (keeps columns narrow)
            f"{desc} & {s.notna().sum()} & {num(s.mean())} & "
            f"{num(s.std())} & {num(s.min())} & {num(s.median())} & "
            f"{num(s.max())} & {num(xs.mean())} \\\\")
    return rows


def table1() -> None:
    df = pd.read_csv(PANEL, parse_dates=["date"],
                     usecols=["ticker", "date"] + [c for c, _ in CLF_NUMERIC])
    fund = pd.read_csv(FR / "firm_fundamentals_panel.csv", parse_dates=["date"])
    nav = pd.read_csv(FR / "nav_proxy_panel.csv", parse_dates=["date"])[
        ["ticker", "date", "navprem_book_adj"]]
    fund = fund.merge(nav, on=["ticker", "date"], how="left")

    splits = [("Train (2015--2021)", "enriched_train_2015_2021.csv"),
              ("Validation (2022--2023)", "enriched_validation_2022_2023.csv"),
              ("Test (2024--2025)", "enriched_test_2024_2025.csv")]
    lab_rows = []
    for disp, fn in splits:
        d = pd.read_csv(SPLIT / fn, usecols=["label"])
        vc, n = d["label"].value_counts(), len(d)
        cells = " & ".join(
            f"{int(vc.get(k, 0))} ({pct(vc.get(k, 0) / n)})"
            for k in ("increase", "hold", "reduce"))
        lab_rows.append(rf"{disp} & {n} & {cells} & & \\")

    body = [
        panel("Panel A. Label distribution by chronological split", 8),
        r"\addlinespace",
        r"Split & N & Increase & Hold & Reduce & & & \\",
        r"\cmidrule(r){1-5}",
        *lab_rows,
        r"\midrule",
        panel("Panel B. Classifier inputs (13 numeric features + sector dummy)", 8),
        r"\addlinespace",
        r" & N & Mean & SD & Min & Median & Max & CS SD\tnote{a} \\",
        r"\midrule",  # full-span header rule: \midrule, not \cmidrule(r){1-8}
        *desc_rows(df, CLF_NUMERIC),
        r"\midrule",
        panel("Panel C. Reconstructed REIT fundamentals (9)", 8),
        r"\addlinespace",
        *desc_rows(fund, FUND),
    ]
    notes = [
        r"\item Notes: This table reports summary statistics for the 25-REIT monthly "
        r"panel, 2015--2025. Panel A reports the label distribution (next-month total return "
        r"$> +2\%$ $\rightarrow$ increase, $< -2\%$ $\rightarrow$ reduce, else hold; "
        r"cell shares in parentheses) across the strictly chronological split; Panel B covers the 13 numeric classifier inputs (a sector dummy enters one-hot); Panel C the nine reconstructed point-in-time fundamentals. All "
        r"macro variables have a within-month cross-sectional SD of essentially zero, "
        r"so they cannot discriminate which firm reduces; all nine fundamentals vary "
        r"genuinely across firms. Variable definitions and mnemonics are in "
        r"Table~A1. Sources: "
        r"\texttt{data/processed/backtest\_ready\_panel\_enriched.csv}, "
        r"\texttt{outputs/fundamentals\_robustness/firm\_fundamentals\_panel.csv}, "
        r"\texttt{nav\_proxy\_panel.csv}, \texttt{data/processed/splits/*.csv}.",
        r"\item[a] Mean across months of the within-month cross-sectional standard "
        r"deviation.",
    ]
    write_table("table1", "Summary statistics", r"@{}lrrrrrrr@{}", body, notes,
                label="tab:summary")


# ------------------------------------------------------------------- Table 2 --
def table2() -> None:
    cm = pd.read_csv(TAB / "quant_only_confusion_matrix_test.csv", index_col=0)
    tr = pd.read_csv(FR / "tuned_robustness.csv")

    labels = [("true_increase", "Increase"), ("true_hold", "Hold"),
              ("true_reduce", "Reduce")]
    cm_rows = []
    for key, disp in labels:
        r = cm.loc[key]
        n = int(r.sum())
        rec = int(r[f"pred_{disp.lower()}"]) / n
        cm_rows.append(
            rf"{disp} ($n={n}$) & {int(r['pred_increase'])} & {int(r['pred_hold'])} "
            rf"& {int(r['pred_reduce'])} & {num(rec)} & \\")

    tuned_rows = []
    for _, r in tr.iterrows():
        if pd.isna(r.get("best_C")):
            cl1 = "MLE"
        else:
            cl1 = f"({dec(r['best_C'], 2)}, {dec(r['best_l1'], 2)})"
        model = MODEL_NAMES[str(r["model"])]
        tuned_rows.append(
            rf"{model} & {cl1} & {num(r['hold_recall'])} & {num(r['reduce_recall'])} "
            rf"& {int(r['reduce_npred'])} & {num(r['accuracy'])} \\")

    body = [
        panel("Panel A. Test confusion matrix, primary multinomial logit (575 decisions)", 6),
        r"\addlinespace",
        r" & \multicolumn{3}{c}{Predicted} & & \\",
        r"\cmidrule(lr){2-4}",
        r"True label & Increase & Hold & Reduce & Recall & \\",
        r"\cmidrule(r){1-5}",
        *cm_rows,
        r"\midrule",
        panel("Panel B. Robustness: tuning, feature enrichment, and model family", 6),
        r"\addlinespace",
        r"Model & Best $(C, L1)$ & Hold recall & Reduce recall & "
        r"$n$ reduce & Accuracy \\",
        r"\midrule",
        *tuned_rows,
    ]
    notes = [
        r"\item Notes: This table reports the classification performance of the "
        r"structured baseline on the 2024--2025 test set. Panel A shows the "
        r"confusion matrix of the primary tuned multinomial logistic regression "
        r"(elastic net, balanced class weights, purged walk-forward CV): reduce is "
        r"never the argmax class, so reduce recall is 0.000 (0/165)---an argmax "
        r"artifact of an uninformative classifier, not a tuned floor. Panel B shows "
        r"that the zero is invariant to an independent hyper-parameter re-selection "
        r"(\texttt{TimeSeriesSplit}), to enriching the model with all nine "
        r"reconstructed fundamentals (including the NAV proxy), and to a change of "
        r"model family (ordered/proportional-odds logit). ``$n$ reduce'' is the "
        r"number of test observations predicted reduce. CV macro-F1: 0.277, 0.281, "
        r"and --- respectively. Sources: "
        r"\texttt{outputs/tables/quant\_only\_confusion\_matrix\_test.csv}, "
        r"\texttt{outputs/fundamentals\_robustness/tuned\_robustness.csv}.",
    ]
    write_table("table2", "Baseline classification: confusion matrix and robustness",
                r"@{}lrrrrr@{}", body, notes, label="tab:baseline")


# ------------------------------------------------------------------- Table 3 --
def table3() -> None:
    m = pd.read_csv(TAB / "ols_metrics_enriched.csv").set_index("period")
    ext = pd.read_csv(FR / "extended_ols.csv")
    fm = pd.read_csv(FR / "extended_fama_macbeth.csv")
    nw = pd.read_csv(LD / "fama_macbeth_nw.csv")

    r2 = m["r2"]
    ols_rows = [
        rf"Macro-enriched panel & {num(r2['Training (2015-2021)'], 2)} & "
        rf"{num(r2['Validation (2022-2023)'], 2)} & {num(r2['Test (2024-2025)'], 2)} "
        rf"& {int(m.loc['Test (2024-2025)', 'n_obs'])} \\"]
    for _, r in ext.iterrows():
        name = MODEL_NAMES[str(r["model"])]
        ols_rows.append(
            rf"{name} & {num(r['train'], 2)} & {num(r['val'], 2)} & "
            rf"{num(r['test'], 2)} & {int(r['n_test'])} \\")

    keep = ["leverage", "debt_to_equity", "ret_6m", "navprem_book_adj",
            "amihud_illiq", "idio_vol", "book_to_market", "ffo_yield_proxy",
            "ln_mktcap", "interest_cover"]
    # journal style: readable row labels, no code mnemonics (those live in Table A1)
    fm_names = {
        "leverage": "Leverage", "debt_to_equity": "Debt-to-equity",
        "ret_6m": "6-month return", "navprem_book_adj": "NAV premium proxy",
        "amihud_illiq": "Amihud illiquidity", "idio_vol": "Idiosyncratic volatility",
        "book_to_market": "Book-to-market", "ffo_yield_proxy": "FFO yield proxy",
        "ln_mktcap": "Log market cap", "interest_cover": "Interest coverage",
    }
    full = fm[(fm.model == "Extended") & (fm.period == "full")].set_index("feature")
    test = fm[(fm.model == "Extended") & (fm.period == "test")].set_index("feature")
    nw_full = nw[(nw.model == "Extended") & (nw.period == "full")].set_index("feature")
    nw_test = nw[(nw.model == "Extended") & (nw.period == "test")].set_index("feature")

    fm_rows = []
    for c in keep:
        cf, tf = full.loc[c, "mean_coef"], full.loc[c, "fm_t"]
        ct, tt = test.loc[c, "mean_coef"], test.loc[c, "fm_t"]
        # Newey-West t reported only for the classically significant (5%) terms
        nwf = num(nw_full.loc[c, "t_newey_west"], 2) if full.loc[c, "sig_5pct"] else ""
        nwt = num(nw_test.loc[c, "t_newey_west"], 2) if test.loc[c, "sig_5pct"] else ""
        fm_rows.append(rf"{fm_names[c]} & {coef(cf, tf)} & {nwf} & {coef(ct, tt)} & {nwt} \\")
        fm_rows.append(rf" & $({dec(tf, 2)})$ & & $({dec(tt, 2)})$ & \\")

    body = [
        panel("Panel A. OLS on the next-month return: out-of-sample $R^2$", 5),
        r"\addlinespace",
        r"Specification & Train $R^2$ & Validation $R^2$ & Test $R^2$ & $N$ test \\",
        r"\midrule",
        *ols_rows,
        r"\midrule",
        panel("Panel B. Fama--MacBeth cross-sectional regressions (extended set)", 5),
        r"\addlinespace",
        r" & \multicolumn{2}{c}{Full sample (2015--2025)} & "
        r"\multicolumn{2}{c}{Test window (2024--2025)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Feature & Coefficient & NW $t$ & Coefficient & NW $t$ \\",
        r"\midrule",
        *fm_rows,
    ]
    notes = [
        r"\item Notes: This table reports return predictability in levels. Panel A: "
        r"a positive train $R^2$ collapsing to a large negative out-of-sample $R^2$ "
        r"(worse than the training-mean forecast) is the classic overfit signature "
        r"of macro-based return prediction; adding the nine fundamentals raises "
        r"in-sample fit but leaves test $R^2$ negative. Panel B: two-pass "
        r"Fama--MacBeth regressions (monthly cross-sectional slopes, then "
        r"time-series means); classical $t$-statistics in parentheses; the NW $t$ "
        r"column reports Newey--West HAC $t$-statistics (Bartlett kernel; 4 lags "
        r"full sample, $T=118$; 2 lags test window, $T=23$) for the classically "
        r"significant terms, all of which survive. Sector dummies included but not "
        r"reported. Variable mnemonics are given in Table~A1. Sources: "
        r"\texttt{outputs/tables/ols\_metrics\_enriched.csv}, "
        r"\texttt{outputs/fundamentals\_robustness/extended\_ols.csv}, "
        r"\texttt{extended\_fama\_macbeth.csv}, "
        r"\texttt{outputs/llm\_deepseek\_test/fama\_macbeth\_nw.csv}.",
    ]
    write_table("table3", "Return predictability: out-of-sample $R^2$ and "
                "Fama--MacBeth pricing", r"@{}lrrrr@{}", body, notes,
                label="tab:returns")


# ------------------------------------------------------------------- Table 4 --
def table4() -> None:
    h = pd.read_csv(LD / "headline_comparison.csv").set_index("method")
    b = pd.read_csv(LD / "significance_bootstrap.csv").set_index("quantity")
    inf = pd.read_csv(FR / "inference_robustness.csv", index_col=0)["value"]
    mde = pd.read_csv(FR / "power_mde.csv")
    mde_sim = mde[mde.quantity == "MDE_simulation_80pct"].set_index("scheme")["value"]

    def ci(q: str) -> str:
        return (rf"$[{dec(b.loc[q, 'ci_lo'], 3)}, {dec(b.loc[q, 'ci_hi'], 3)}]$")

    def cells(key: str) -> str:
        r = h.loc[key]
        hold = num(r["hold_recall"]) if pd.notna(r["hold_recall"]) else DASH
        acc = num(r["accuracy"]) if pd.notna(r["accuracy"]) else DASH
        return (f"{num(r['reduce_recall'])} & {num(r['reduce_precision'])} & "
                f"{hold} & {acc} & {num(r['reduce_fire_rate'])}")

    body = [
        r"Method & Reduce recall & Reduce precision & Hold recall & Accuracy & "
        r"Firing rate \\",
        r"\midrule",
        rf"LLM five-agent framework & {cells('LLM-DeepSeek')} \\",
        rf" & {ci('LLM')} & & & & \\",
        r"\addlinespace",
        rf"Random-at-budget & {cells('Random-at-budget')} \\",
        rf"Threshold logistic & {cells('Threshold-logistic')} \\",
        rf"Multinomial logit (argmax) & {cells('Logistic-argmax')} \\",
        r"\midrule",
        rf"LLM $-$ random-at-budget & "
        rf"{num(b.loc['LLM_minus_Random', 'reduce_recall_mean'])} & "
        rf"{DASH} & {DASH} & {DASH} & {DASH} \\",
        rf" & {ci('LLM_minus_Random')} & & & & \\",
    ]
    notes = [
        r"\item Notes: This table reports the main comparison on the reduce class: "
        r"all methods are evaluated on the identical 2024--2025 test set (575 "
        r"decisions, 165 true reduce events) and, except the argmax logit, at the "
        r"identical firing budget (117 reduce calls, rate 0.203). Brackets are "
        r"REIT-block bootstrap 95\% confidence intervals ($B=2{,}000$). The "
        r"month-block bootstrap CI for the difference, which respects the temporal "
        rf"clustering of reduce events, is $[-0.066, 0.085]$"
        r"---both intervals contain zero, so we find no statistically reliable "
        r"evidence that the framework outperforms random-at-budget. A TOST "
        r"equivalence test against a pre-specified $\pm 5$pp margin is also "
        r"inconclusive (the CI exceeds the band on at least one side), so strict "
        r"equivalence is not asserted either. A simulation-based power analysis "
        r"puts the minimum detectable effect at 80\% power at "
        rf"{dec(float(mde_sim['reit_block']) * 100, 1)}pp of reduce recall under "
        rf"REIT-block resampling ({dec(float(mde_sim['month_block']) * 100, 1)}pp "
        r"under month-block): an economically large edge is unlikely, but a "
        r"sub-10pp edge would be undetectable at this sample size. The argmax "
        r"logit predicts reduce for zero observations (an argmax artifact, not a "
        r"tuned floor). Sources: "
        r"\texttt{outputs/llm\_deepseek\_test/headline\_comparison.csv}, "
        r"\texttt{significance\_bootstrap.csv}, "
        r"\texttt{outputs/fundamentals\_robustness/inference\_robustness.csv}, "
        r"\texttt{power\_mde.csv}.",
    ]
    # sanity: month-block CI hardwired in the note matches inference_robustness.csv
    assert str(inf.get("LLMminusRandom_monthblock_CI")).replace(" ", "") == \
        "[-0.066,0.085]", "month-block CI drifted from inference_robustness.csv"
    write_table("table4", "Main comparison on the reduce class (test set, 575 "
                "decisions)", r"@{}lrrrrr@{}", body, notes, label="tab:main")


# ------------------------------------------------------------------- Table 5 --
def table5() -> None:
    a = pd.read_csv(LD / "ablation_channels.csv")
    n_reduce = 165  # true reduce events in the test set (Table 1, Panel C)
    rows = []
    for _, r in a.iterrows():
        d_pp = (r["reduce_recall"] - r["random_at_budget_recall"]) * 100
        hits = (r["reduce_recall"] - r["random_at_budget_recall"]) * n_reduce
        cfg = str(r["config"]).replace("+", r"$+$")
        sd, sh = (("+" if v >= 0 else "") + dec(v, 1) for v in (d_pp, hits))
        rows.append(
            rf"{cfg} & {num(r['reduce_recall'])} & {num(r['reduce_precision'])} & "
            rf"{num(r['reduce_fire_rate'])} & {num(r['random_at_budget_recall'])} & "
            rf"${sd}$ & ${sh}$ \\")
    body = [
        r"Configuration & Recall & Precision & Firing rate & "
        r"Random recall\tnote{a} & $\Delta$ recall (pp) & $\Delta$ hits\tnote{b} \\",
        r"\midrule",
        *rows,
    ]
    notes = [
        r"\item Notes: This table reports the channel ablation: the aggregation is "
        r"re-run over agent subsets (D $=$ Disclosure, M $=$ Macro, P $=$ Price, "
        r"F $=$ Fundamentals) on the identical test set, and each configuration is "
        r"compared with a random baseline at its own firing budget. No "
        r"configuration beats random-at-budget by a margin that survives sampling "
        r"uncertainty: the gaps range from $-5.4$ to $+7.1$ hits out of 165, well "
        r"inside the $\pm 12$pp REIT-block bootstrap interval of Table 4. The "
        r"channels nonetheless occupy distinct precision--recall regions "
        r"(disclosure-only is conservative but precise; structured-only is "
        r"aggressive but imprecise), evidence that they are genuinely different "
        r"diagnostic perspectives rather than redundant copies. Recall and "
        r"precision refer to the reduce class. Source: "
        r"\texttt{outputs/llm\_deepseek\_test/ablation\_channels.csv}.",
        r"\item[a] Expected reduce recall of a random classifier at the "
        r"configuration's own firing budget.",
        r"\item[b] Implied hit difference: ($\Delta$ recall) $\times$ 165 true "
        r"reduce events.",
    ]
    write_table("table5", "Channel ablation: agent-subset configurations (test set)",
                r"@{}lrrrrrr@{}", body, notes, label="tab:ablation")


# ------------------------------------------------------------------- Table 6 --
def table6() -> None:
    s = pd.read_csv(FR / "systematic_vs_idiosyncratic.csv", index_col=0)["value"]
    ma = pd.read_csv(FR / "market_adjusted_labels.csv")
    vd = ma[ma.section == "variance_decomposition_market_adjusted"].set_index(
        "metric")["value"]
    lab = ma[(ma.section == "label_distribution") & (ma.item == "test")].set_index(
        "metric")["value"]
    tl = ma[ma.section == "tuned_logit_market_adjusted"]
    # reduce recall is 0.000 for BOTH feature sets under market-adjusted labels
    assert (tl[tl.metric == "reduce_recall"]["value"] == 0.0).all()

    rows = [
        ("Month-fixed-effects $R^2$ (who reduces)",
         num(s["D_month_FE_R2_on_reduce"]), num(vd["month_FE_R2_on_reduce"])),
        ("Firm-features $R^2$ (who reduces)",
         num(s["D_firm_features_R2_on_reduce"]), num(vd["firm_features_R2_on_reduce"])),
        ("Monthly reduce-rate dispersion vs.\\ binomial null",
         f"${dec(s['C_dispersion_ratio_obs_over_null'], 2)}\\times$",
         f"${dec(vd['dispersion_ratio_obs_over_null'], 2)}\\times$"),
        ("Share of reduce events in high-reduce months",
         pct(s["C_pct_reduce_in_high_reduce_months"]),
         pct(vd["pct_reduce_in_high_reduce_months"])),
        ("$n$ high-reduce months (of 130)",
         str(int(s["C_n_high_reduce_months"])), str(int(vd["n_high_reduce_months"]))),
        ("Tuned-logit reduce recall (test)\\tnote{a}", num(0.0), num(0.0)),
        ("Test reduce events", "165", str(int(lab["reduce_n"]))),
    ]
    body = [
        r"Quantity & Raw labels & Market-adjusted labels \\",
        r"\midrule",
        *[f"{q} & {a} & {b} \\\\" for q, a, b in rows],
    ]
    notes = [
        r"\item Notes: This table reports the systematic decomposition of the "
        r"reduce outcome under the original labels (raw next-month total return, "
        r"$\pm 2\%$ band) and under market-adjusted labels, where each firm-month "
        r"is relabelled on its return in excess of a leave-one-out equal-weight "
        r"REIT index (same band), stripping the common factor out of the label by "
        r"construction. Under raw labels, knowing \emph{when} (month fixed "
        r"effects, $R^2 = 0.433$) explains sixty times more than knowing "
        r"everything firm-specific ($R^2 = 0.007$), and reduce events cluster in "
        r"time ($3.26\times$ the firm-independent dispersion; 53\% of events in 27 "
        r"of 130 months). Market adjustment removes the clustering exactly as the "
        r"systematic account predicts ($0.90\times$, month-FE $R^2 = 0.03$), yet "
        r"firm-level predictability does not improve: the tuned classifier still "
        r"attains reduce recall 0.000 and firm features still explain almost "
        r"nothing ($R^2 = 0.011$). The null is not an artifact of labelling on raw "
        r"returns. Sources: "
        r"\texttt{outputs/fundamentals\_robustness/systematic\_vs\_idiosyncratic.csv}, "
        r"\texttt{market\_adjusted\_labels.csv}.",
        r"\item[a] 0.000 for both feature sets (original 13 $+$ sector and "
        r"extended $+$ 9 fundamentals); the classifier fires zero times in both "
        r"label regimes.",
    ]
    write_table("table6", "Systematic versus idiosyncratic downside: raw and "
                "market-adjusted labels", r"@{}lrr@{}", body, notes,
                label="tab:systematic")


# ------------------------------------------------------------------- Table 7 --
def table7() -> None:
    c = pd.read_csv(LD / "contamination_summary.csv").iloc[0]
    fa = pd.read_csv(LD / "factuality_audit.csv", index_col=0)["value"]
    rq = pd.read_csv(LD / "rationale_quality_audit_summary.csv", index_col=0)["value"]
    sc = pd.read_csv(LD / "spotcheck_agreement.csv").set_index("dimension")

    n_jud = int(float(rq["reduce_decisions_judged"]))
    ent = sc.loc["entailment"]
    # v1 column values exist only in the verified draft (THESIS_DRAFT.md Table 8)
    # and CANONICAL_RESULTS.md section 4 (source: v1 run's contamination audit).
    V1 = {"lookahead": "0", "grounded": r"57\%", "facts": "2.84",
          "money": r"53\%", "all": r"68\%"}
    rows = [
        ("Input look-ahead violations", V1["lookahead"],
         str(int(c["input_lookahead_violations"]))),
        ("Reduce decisions grounded in a specific filing fact", V1["grounded"],
         pct(c["reduce_grounded_pct"], 0)),
        ("Mean cited facts per reduce decision", V1["facts"],
         dec(c["reduce_mean_facts_cited"], 2)),
        (r"Reduce rationales citing a dollar amount or \%", V1["money"],
         pct(c["reduce_with_money_or_pct_pct"], 0)),
        ("All-decision grounding rate", V1["all"], pct(c["all_grounded_pct"], 0)),
        ("MIDRULE", "", ""),
        ("Fundamentals cited values matching model inputs", DASH,
         pct(fa["fundamentals_factual_accuracy"], 0)
         + rf" ($n={big_n(fa['fundamentals_cited_values_checked'])}$)"),
        ("Disclosure numeric tokens verifiable in source filing", DASH,
         pct(fa["disclosure_numeric_in_source_rate"], 1)
         + rf" ($n={int(fa['disclosure_numeric_tokens_checked'])}$)"),
        ("MIDRULE", "", ""),
        ("Judged materially relevant (relevance $\\geq 1$)", DASH,
         pct(rq["pct_relevance_material_ge1"], 0)
         + rf", mean {dec(rq['mean_relevance'], 2)}/2 ($n={n_jud}$)"),
        ("Judged at least partially entailing (entailment $\\geq 1$)", DASH,
         pct(rq["pct_entailment_supported_ge1"], 0)
         + rf"; {pct(rq['pct_entailment_strong_eq2'], 0)} fully "
         rf"(mean {dec(rq['mean_entailment'], 2)}/2)"),
        ("MIDRULE", "", ""),
        ("Human--judge agreement on entailment (45 decisions)", DASH,
         pct(ent["exact_agreement"], 0)
         + rf" exact; $\kappa = {dec(ent['quadratic_kappa'], 2)}$"),
    ]
    body = [r"Metric & v1 & v2 (five-agent) \\", r"\midrule"]
    for m, v1, v2 in rows:
        body.append(r"\midrule" if m == "MIDRULE" else f"{m} & {v1} & {v2} \\\\")
    notes = [
        r"\item Notes: This table reports the grounding and factuality audit of the "
        r"framework's rationales on the test set. v1 is the broken-text version "
        r"(the Disclosure agent was fed an XBRL metadata block); v2 is after the "
        r"Item 1A extraction fix and the addition of the Fundamentals agent. Zero "
        r"look-ahead violations in both versions confirm the filing-date "
        r"enforcement holds; the rise from 57\% to 88\% grounded reduce decisions "
        r"is the measurable ``decomposable rationales'' claim, and grounding is a "
        r"deterministic, regex-verifiable property of the audit log. The factual "
        r"accuracy rows upgrade ``grounded'' (cites a fact) to ``factually "
        r"accurate'' (the cited value is true in the source); the judge rows "
        r"(independent DeepSeek V4-Pro auto-judge, blind to outcomes) upgrade it "
        r"further to diagnostic relevance---uniformly relevant but only partially "
        r"entailing, the signature of a systematic rather than firm-specific risk; "
        r"and the human spot-check (45 stratified decisions, blind) independently "
        r"reproduces the partial-entailment reading. v1 figures are from the v1 "
        r"run's audit (\texttt{CANONICAL\_RESULTS.md}); v2 from "
        r"\texttt{outputs/llm\_deepseek\_test/contamination\_summary.csv}, "
        r"\texttt{factuality\_audit.csv}, "
        r"\texttt{rationale\_quality\_audit\_summary.csv}, "
        r"\texttt{spotcheck\_agreement.csv}.",
    ]
    write_table("table7", "Grounding and factuality audit of rationales (test set)",
                r"@{}p{0.58\linewidth}rr@{}", body, notes, label="tab:grounding")


# ------------------------------------------------------------------- Table 8 --
def table8() -> None:
    rq = pd.read_csv(LD / "rationale_quality_audit_summary.csv", index_col=0)["value"]
    sc = pd.read_csv(LD / "spotcheck_agreement.csv")

    n_jud = int(float(rq["reduce_decisions_judged"]))
    judge_rows = [
        rf"Entailment & {dec(rq['mean_entailment'], 2)} & "
        rf"{pct(rq['pct_entailment_supported_ge1'], 0)} & "
        rf"{pct(rq['pct_entailment_strong_eq2'], 0)} & & & \\",
        rf"Relevance & {dec(rq['mean_relevance'], 2)} & "
        rf"{pct(rq['pct_relevance_material_ge1'], 0)} & {DASH} & & & \\",
        rf"Actionability & {dec(rq['mean_actionability'], 2)} & {DASH} & {DASH} "
        r"& & & \\",
    ]
    agree_rows = []
    for _, r in sc.iterrows():
        name = str(r["dimension"])
        disp = "Pooled (3 dimensions)" if name.startswith("OVERALL") \
            else name.capitalize()
        agree_rows.append(
            rf"{disp} & {int(r['n'])} & {dec(r['human_mean'], 2)} & "
            rf"{dec(r['machine_mean'], 2)} & {pct(r['exact_agreement'], 0)} & "
            rf"{pct(r['within_1_agreement'], 0)} & "
            rf"{dec(r['quadratic_kappa'], 2)} \\")
    # the pooled row sits after a midrule
    pooled = agree_rows.pop()

    body = [
        panel(rf"Panel A. Automated judge scores (DeepSeek V4-Pro, $n={n_jud}$ "
              r"reduce rationales, 0--2 scale)", 7),
        r"\addlinespace",
        r"Dimension & Mean & Share $\geq 1$ & Share $=2$ & & & \\",
        r"\cmidrule(r){1-4}",
        *judge_rows,
        r"\midrule",
        panel("Panel B. Human--judge agreement (45-decision blind spot-check)", 7),
        r"\addlinespace",
        r"Dimension & $n$ & Human mean & Judge mean & Exact & Within one & "
        r"Quadratic $\kappa$ \\",
        r"\midrule",
        *agree_rows,
        r"\midrule",
        pooled,
    ]
    notes = [
        r"\item Notes: This table reports the rationale-quality audit. Panel A: an "
        r"independent automated judge (DeepSeek V4-Pro, distinct from the V4-Flash "
        r"generator to limit self-evaluation bias) scores each reduce rationale, "
        r"blind to the realized outcome, on entailment (do the cited facts "
        r"logically support a downside view?), relevance (are they materially "
        r"relevant to near-term downside?), and actionability, each on a 0--2 "
        r"scale; 114 of 117 reduce decisions are parseable. Rationales are "
        r"uniformly relevant and at least partially entailing, but only 22\% fully "
        r"entailing---consistent with a systematic rather than firm-specific risk. "
        r"Panel B: a human rater (the author) scored a 45-decision stratified "
        r"subset blind to the machine scores. Agreement is substantial on the "
        r"core entailment dimension ($\kappa = 0.76$); the low relevance $\kappa$ "
        r"reflects a saturated margin (both raters near the top of the scale), not "
        r"disagreement; the raters diverge most on the subjective actionability "
        r"dimension, though never by more than one point. Sources: "
        r"\texttt{outputs/llm\_deepseek\_test/rationale\_quality\_audit\_summary.csv}, "
        r"\texttt{spotcheck\_agreement.csv}.",
    ]
    write_table("table8", "Rationale quality: automated judge scores and human "
                "calibration", r"@{}lrrrrrr@{}", body, notes, label="tab:quality")


# ------------------------------------------------------------------ Table A1 --
def tableA1() -> None:
    # Definitions from THESIS_DRAFT.md sections 3.3-3.4, outputs/tables/
    # v6_selected_features.csv (Zone 1, read-only), and the construction code in
    # analysis/build_fundamentals.py and analysis/build_nav_proxy.py.
    lagged = "Lagged one month ($t-1$)"
    through_t = "Computed through month-end $t$"
    filed = r"Most recent annual value with SEC \texttt{filed} date $\leq t$"
    clf = [
        ("ret_1m", "Trailing 1-month total return (adjusted close)", through_t,
         "Prices"),
        ("ret_3m", "Trailing 3-month total return", through_t, "Prices"),
        ("ret_6m", "Trailing 6-month total return", through_t, "Prices"),
        ("ret_12m", "Trailing 12-month total return", through_t, "Prices"),
        ("vol_annualized",
         "Annualized standard deviation of daily returns, 60-day rolling window",
         through_t, "Prices"),
        ("drawdown", "Decline of adjusted close from its running maximum",
         through_t, "Prices"),
        ("dividend_yield_lag1", "Trailing dividend yield (dividends over price)",
         lagged, "Prices"),
        ("FEDFUNDS_lag1", "Effective federal funds rate", lagged, "FRED"),
        ("DGS10_lag1", "10-year Treasury constant-maturity yield", lagged, "FRED"),
        ("DGS2_lag1", "2-year Treasury constant-maturity yield", lagged, "FRED"),
        ("term_spread_10y_2y_lag1", "10-year minus 2-year Treasury yield", lagged,
         "FRED"),
        ("cpi_yoy_lag1", "CPI year-over-year change", lagged, "FRED"),
        ("UNRATE_lag1", "Civilian unemployment rate", lagged, "FRED"),
        ("sector", "REIT property-sector classification (one-hot dummies)",
         "Static", "Universe file"),
    ]
    fnd = [
        ("leverage", r"Long-term debt over total assets "
         r"(\texttt{LongTermDebt}/\texttt{Assets})", filed, "XBRL"),
        ("debt_to_equity", r"Total liabilities over stockholders' equity", filed,
         "XBRL"),
        ("interest_cover",
         r"(Net income $+$ interest expense $+$ D\&A) over interest expense",
         filed, "XBRL"),
        ("ffo_yield_proxy",
         r"(Net income $+$ D\&A) over market capitalization; NAREIT FFO logic "
         r"(Vincent, 1999)", filed + "; price at $t$", "XBRL $+$ prices"),
        ("book_to_market",
         r"Stockholders' equity over market capitalization (Fama--French, 1992)",
         filed + "; price at $t$", "XBRL $+$ prices"),
        ("ln_mktcap",
         r"Log market capitalization (shares outstanding $\times$ price)",
         "Shares filed $\\leq t$; price at $t$", "XBRL $+$ prices"),
        ("amihud_illiq",
         r"Mean of $|$daily return$|$ over dollar volume, trailing 21 trading "
         r"days (Amihud, 2002)", through_t, "Prices"),
        ("idio_vol",
         r"Annualized standard deviation of market-model residuals, trailing 63 "
         r"trading days (Ang et al., 2006)", through_t, "Prices"),
        ("navprem_book_adj",
         r"Market capitalization over depreciation-adjusted book equity "
         r"(equity $+$ accumulated real-estate depreciation); NAV-premium proxy "
         r"(Clayton \& MacKinnon, 2003)", filed + "; price at $t$",
         "XBRL $+$ prices"),
    ]

    def rows(items):
        return [f"{code_br(v)} & {d} & {t} & {s} \\\\" for v, d, t, s in items]

    body = [
        r"Variable & Definition / construction & Timing & Source \\",
        r"\midrule",
        panel("Panel A. Classifier inputs (13 numeric features + sector)", 4),
        r"\addlinespace",
        *rows(clf),
        r"\midrule",
        panel("Panel B. Reconstructed REIT fundamentals (9)", 4),
        r"\addlinespace",
        *rows(fnd),
    ]
    notes = [
        r"\item Notes: This table defines every classifier feature and "
        r"reconstructed fundamental. ``Lagged one month'' means the value known at "
        r"the end of month $t-1$ enters the decision at $t$; accounting items "
        r"follow the strict point-in-time rule that a value is attached to "
        r"decision month $t$ only if its SEC \texttt{filed} date is on or before "
        r"$t$ (missing values are left missing, never imputed). Sources: prices "
        r"$=$ local daily price/volume histories; FRED $=$ Federal Reserve "
        r"Economic Data; XBRL $=$ SEC company-facts filings. Construction details: "
        r"THESIS\_DRAFT.md \S 3.3--3.4, "
        r"\texttt{analysis/build\_fundamentals.py}, "
        r"\texttt{analysis/build\_nav\_proxy.py}.",
    ]
    write_table("tableA1", "Table A1 --- Variable definitions",
                r"@{}p{0.20\linewidth}p{0.40\linewidth}p{0.22\linewidth}"
                r"p{0.10\linewidth}@{}",
                body, notes, caption_star=True)


def main() -> None:
    for fn in (table1, table2, table3, table4, table5, table6, table7, table8,
               tableA1):
        fn()
    print(f"\nwritten -> {OUT}/ (9 tables)")


if __name__ == "__main__":
    main()
