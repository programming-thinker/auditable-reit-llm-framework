"""Three-way agreement analysis once a second human rater has scored the 45-decision
spot-check subset (audit_log/rater2/rationale_spotcheck_worksheet_rater2.md).

Raters:
  R1 = the author  (audit_log/rationale_spotcheck_human.csv, columns h_*)
  R2 = the second independent rater (filled worksheet .md, or a CSV with columns
       ticker, decision_date, r2_entailment, r2_relevance, r2_actionability)
  J  = the automated judge (outputs/llm_deepseek_test/rationale_quality_audit.csv)

Metrics per dimension and pooled, for each pair (R1-R2, R1-J, R2-J):
  n, mean_a, mean_b, exact agreement, within-1 agreement, quadratic-weighted kappa;
plus three-rater Krippendorff's alpha (interval metric on the ordinal 0/1/2 scale).

Validation: `--validate` recomputes R1-vs-J and asserts it reproduces the published
outputs/llm_deepseek_test/spotcheck_agreement.csv (kappa to 3 dp) — proving this
implementation matches the one behind the thesis numbers before R2 data are used.

Usage:
  python3 analysis/second_rater_agreement.py --validate
  python3 analysis/second_rater_agreement.py --rater2 audit_log/rater2/rationale_spotcheck_worksheet_rater2.md

Output: outputs/llm_deepseek_test/second_rater_agreement.csv (deterministic, no API).
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R1_CSV = REPO / "audit_log/rationale_spotcheck_human.csv"
JUDGE_CSV = REPO / "outputs/llm_deepseek_test/rationale_quality_audit.csv"
PUBLISHED = REPO / "outputs/llm_deepseek_test/spotcheck_agreement.csv"
OUT = REPO / "outputs/llm_deepseek_test/second_rater_agreement.csv"
DIMS = ["entailment", "relevance", "actionability"]
CATS = np.array([0, 1, 2])

WORKSHEET_ITEM_RE = re.compile(r"^## \d+\.\s+([A-Z]+)\s+—\s+(\d{4}-\d{2}-\d{2})", re.M)
SCORE_RES = {
    "entailment": re.compile(r"entailment[^=\n]*=\s*([012])"),
    "relevance": re.compile(r"relevance[^=\n]*=\s*([012])"),
    "actionability": re.compile(r"actionability[^=\n]*=\s*([012])"),
}


def parse_worksheet(path: Path) -> pd.DataFrame:
    """Parse a filled rater-2 worksheet: '## N. TICKER — DATE' sections with
    'entailment ... = X' style lines."""
    text = path.read_text(encoding="utf-8")
    items = list(WORKSHEET_ITEM_RE.finditer(text))
    rows = []
    for k, m in enumerate(items):
        chunk = text[m.end(): items[k + 1].start() if k + 1 < len(items) else len(text)]
        row = {"ticker": m.group(1), "decision_date": m.group(2)}
        for dim, rx in SCORE_RES.items():
            hit = rx.search(chunk)
            if hit is None:
                raise SystemExit(f"[!] unscored or unparseable {dim} for "
                                 f"{row['ticker']} {row['decision_date']} — fill every '= ___'")
            row[f"r2_{dim}"] = int(hit.group(1))
        rows.append(row)
    if len(rows) != 45:
        raise SystemExit(f"[!] parsed {len(rows)} items, expected 45")
    return pd.DataFrame(rows)


def load_rater2(path: Path) -> pd.DataFrame:
    if path.suffix == ".md":
        return parse_worksheet(path)
    df = pd.read_csv(path)
    need = ["ticker", "decision_date"] + [f"r2_{d}" for d in DIMS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[!] rater-2 CSV missing columns: {missing}")
    return df[need]


def quadratic_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa with quadratic weights on categories {0,1,2}."""
    k = len(CATS)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[int(x), int(y)] += 1
    obs /= obs.sum()
    pa, pb = obs.sum(axis=1), obs.sum(axis=0)
    exp = np.outer(pa, pb)
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    denom = (w * exp).sum()
    return float("nan") if denom == 0 else 1.0 - (w * obs).sum() / denom


def krippendorff_alpha_interval(cols: list[np.ndarray]) -> float:
    """Krippendorff's alpha, interval metric, complete data, m raters."""
    data = np.vstack(cols).astype(float)          # raters x units
    m, n = data.shape
    do = sum(((data[i, u] - data[j, u]) ** 2)
             for u in range(n) for i, j in combinations(range(m), 2))
    do /= n * m * (m - 1) / 2
    flat = data.flatten()
    de = np.mean([(x - y) ** 2 for i, x in enumerate(flat) for y in flat[i + 1:]])
    return float("nan") if de == 0 else 1.0 - do / de


def pair_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "n": len(a),
        "mean_a": round(float(np.mean(a)), 3),
        "mean_b": round(float(np.mean(b)), 3),
        "exact_agreement": round(float(np.mean(a == b)), 3),
        "within_1_agreement": round(float(np.mean(np.abs(a - b) <= 1)), 3),
        "quadratic_kappa": round(quadratic_kappa(a, b), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater2", type=Path, default=None,
                    help="filled worksheet .md or CSV with r2_* columns")
    ap.add_argument("--validate", action="store_true",
                    help="reproduce the published R1-vs-judge numbers and exit")
    args = ap.parse_args()

    r1 = pd.read_csv(R1_CSV)
    judge = pd.read_csv(JUDGE_CSV)
    base = r1.merge(judge, on=["ticker", "decision_date"], how="inner",
                    suffixes=("", "_j"))
    assert len(base) == 45, f"R1-judge merge produced {len(base)} rows, expected 45"

    if args.validate:
        pub = pd.read_csv(PUBLISHED).set_index("dimension")
        ok = True
        for dim in DIMS:
            got = pair_metrics(base[f"h_{dim}"].to_numpy(), base[dim].to_numpy())
            want_k = float(pub.loc[dim, "quadratic_kappa"])
            match = abs(got["quadratic_kappa"] - want_k) < 5e-3
            ok &= match
            print(f"validate {dim:14s} kappa {got['quadratic_kappa']:.3f} "
                  f"vs published {want_k:.3f} -> {'OK' if match else 'MISMATCH'}")
        sys.exit(0 if ok else 1)

    if args.rater2 is None:
        raise SystemExit("provide --rater2 <filled worksheet .md / csv> or --validate")

    r2 = load_rater2(args.rater2)
    df = base.merge(r2, on=["ticker", "decision_date"], how="inner")
    assert len(df) == 45, f"rater-2 merge produced {len(df)} rows, expected 45"

    pairs = {
        "R1_vs_R2": ("h_{d}", "r2_{d}"),
        "R1_vs_judge": ("h_{d}", "{d}"),
        "R2_vs_judge": ("r2_{d}", "{d}"),
    }
    rows = []
    for pname, (ca, cb) in pairs.items():
        pooled_a, pooled_b = [], []
        for dim in DIMS:
            a = df[ca.format(d=dim)].to_numpy()
            b = df[cb.format(d=dim)].to_numpy()
            rows.append({"pair": pname, "dimension": dim, **pair_metrics(a, b)})
            pooled_a.append(a)
            pooled_b.append(b)
        rows.append({"pair": pname, "dimension": "POOLED_3_DIMS",
                     **pair_metrics(np.concatenate(pooled_a), np.concatenate(pooled_b))})

    for dim in DIMS + ["POOLED_3_DIMS"]:
        if dim == "POOLED_3_DIMS":
            cols = [np.concatenate([df[f"h_{d}"] for d in DIMS]),
                    np.concatenate([df[f"r2_{d}"] for d in DIMS]),
                    np.concatenate([df[d] for d in DIMS])]
        else:
            cols = [df[f"h_{dim}"].to_numpy(), df[f"r2_{dim}"].to_numpy(),
                    df[dim].to_numpy()]
        rows.append({"pair": "ALL_3_RATERS", "dimension": dim,
                     "n": len(cols[0]),
                     "krippendorff_alpha_interval":
                         round(krippendorff_alpha_interval(cols), 3)})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nwritten -> {OUT}")
    ent = out[(out.pair == "R1_vs_R2") & (out.dimension == "entailment")].iloc[0]
    print("\nThesis-ready line (adapt numbers): 'A second, independent human rater "
          f"reproduces the entailment reading (inter-rater quadratic kappa = "
          f"{ent['quadratic_kappa']}, exact agreement {ent['exact_agreement']:.0%}).'")


if __name__ == "__main__":
    main()
