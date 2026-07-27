"""Exposure audit for the Item 1A anchor bug in the REPORTED test run (no API).

Bug: the as-run llm/orchestrator.py _extract_item1a anchored on the FIRST
'item 1a' regex hit in the filing. That hit is frequently a cross-reference
("as set forth in Item 1A ... 'Risk Factors'") inside the Item 1 business
description, so the 15k-char 10-K excerpt fed to the Disclosure agent was
mostly business-description text, not risk factors. (Extractor improved on
2026-07-04 for future runs; this script replays the ORIGINAL logic.)

For each of the 575 test decisions in audit_log/decisions.jsonl we replay the
as-run extraction on the latest 10-K as of the decision date (local metadata +
filings/ only, filing-date enforced exactly like EdgarClient) and flag whether
the as-run excerpt contains a genuine 'Item 1A ... Risk Factors' SECTION
header. Operationalisation: the corrected extractor locates the genuine
section header in the full filing (header-form match followed by 'item 1b' at
a section-scale distance); the flag is True iff that position falls inside the
as-run excerpt span. A raw regex-in-excerpt flag is also recorded (it
overcounts: cross-references match the header form too).

Reports the share with a genuine header in the excerpt, and reduce recall +
grounding rate (analysis/contamination_audit.py rule) for header-in-excerpt
vs not.

Reads (read-only): audit_log/decisions.jsonl, data/interim/filing_metadata.csv,
filings/clean_text/, data/processed/splits/enriched_test_2024_2025.csv.
Writes: outputs/llm_deepseek_test/audit_item1a_anchor.csv (per decision)
        outputs/llm_deepseek_test/audit_item1a_anchor_summary.csv
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.contamination_audit import specificity  # noqa: E402  (grounding rule)
from llm.orchestrator import (  # noqa: E402  (corrected header locator)
    _ITEM1A_HEADER_RE,
    _ITEM1B_RE,
    _MIN_ITEM1A_SECTION_CHARS,
)

DECISIONS = REPO / "audit_log/decisions.jsonl"
METADATA = REPO / "data/interim/filing_metadata.csv"
CLEAN_TEXT = REPO / "filings/clean_text"
TEST_SPLIT = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
OUT_DIR = REPO / "outputs/llm_deepseek_test"
OUT_CSV = OUT_DIR / "audit_item1a_anchor.csv"
OUT_SUMMARY = OUT_DIR / "audit_item1a_anchor_summary.csv"
MAX_CHARS_10K = 15000  # as-run _truncate budget for the 10-K excerpt


def original_excerpt_span(raw: str, max_chars: int = MAX_CHARS_10K) -> Tuple[int, int]:
    """Replay the ORIGINAL (as-run) _extract_item1a + _truncate as a char span.

    Original logic: anchor at the FIRST 'item 1a' hit, end at the first
    'item 1b' hit after it; if no 'item 1a', anchor at 'risk factors'; else
    the whole text. Then truncate to max_chars.
    """
    starts = [m.start() for m in re.finditer(r"item\s*1a", raw, re.I)]
    ends = [m.start() for m in re.finditer(r"item\s*1b", raw, re.I)]
    if starts:
        s = starts[0]
        e = next((x for x in ends if x > s), len(raw))
    else:
        m = re.search(r"risk factors", raw, re.I)
        s = m.start() if m else 0
        e = len(raw)
    return s, s + min(e - s, max_chars)


def genuine_header_pos(raw: str) -> Optional[int]:
    """Position of the genuine Item 1A section header (corrected locator).

    Mirrors the fixed llm/orchestrator._extract_item1a selection: header-form
    matches followed later by 'item 1b', preferring section-scale distances,
    taking the last. None if the filing has no locatable genuine section
    (e.g. AMT 10-Ks carry only cross-references; VICI has no Item 1B).
    """
    ends = [m.start() for m in _ITEM1B_RE.finditer(raw)]
    headers = [m.start() for m in _ITEM1A_HEADER_RE.finditer(raw)]
    candidates = [s for s in headers if any(e > s for e in ends)]
    if not candidates:
        return None
    def section_len(s: int) -> int:
        return next(e for e in ends if e > s) - s
    long_enough = [s for s in candidates if section_len(s) >= _MIN_ITEM1A_SECTION_CHARS]
    return (long_enough or candidates)[-1]


def latest_10k_file(meta: pd.DataFrame, ticker: str, as_of: str) -> Optional[str]:
    """Latest 10-K clean_text filename as of a date (EdgarClient selection)."""
    m = meta[(meta["ticker"] == ticker)
             & (meta["form"] == "10-K")
             & (meta["filing_date"] <= pd.Timestamp(as_of))]
    if m.empty:
        return None
    row = m.iloc[-1]  # meta pre-sorted ascending by filing_date
    return (f"{row['ticker']}_{row['form']}_{row['filing_date'].date()}"
            f"_{row['accession_nodash']}.txt")


def load_decisions() -> pd.DataFrame:
    rows = []
    with DECISIONS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec["decision_date_t"].startswith(("2024", "2025")):
                continue
            probs = rec["final_probabilities"]
            disc = rec["agent_outputs"]["disclosure"]
            spec = specificity(disc.get("facts_cited") or [], disc.get("rationale", ""))
            rows.append({
                "date": rec["decision_date_t"],
                "ticker": rec["ticker"],
                "pred": max(probs, key=probs.get),
                "grounded": bool(spec["grounded"]),
            })
    return pd.DataFrame(rows)


def group_metrics(g: pd.DataFrame) -> dict:
    red = g[g["true_label"] == "reduce"]
    pred_red = g[g["pred"] == "reduce"]
    return {
        "n": int(len(g)),
        "n_true_reduce": int(len(red)),
        "reduce_recall": round(float((red["pred"] == "reduce").mean()), 4) if len(red) else float("nan"),
        "grounding_rate": round(float(g["grounded"].mean()), 4) if len(g) else float("nan"),
        "n_pred_reduce": int(len(pred_red)),
        "pred_reduce_grounded_pct": round(float(pred_red["grounded"].mean()), 4) if len(pred_red) else float("nan"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_decisions()
    assert len(df) == 575, f"expected 575 test decisions, got {len(df)}"

    truth = pd.read_csv(TEST_SPLIT)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    df = df.merge(truth[["date", "ticker", "label"]].rename(columns={"label": "true_label"}),
                  on=["date", "ticker"], how="left", validate="one_to_one")
    assert df["true_label"].notna().all(), "unmatched decisions vs test split"

    meta = pd.read_csv(METADATA, dtype=str)
    meta["filing_date"] = pd.to_datetime(meta["filing_date"])
    meta = meta.sort_values("filing_date")

    # per-file cache: (as-run excerpt span, genuine header pos, regex-in-excerpt)
    file_cache: dict = {}
    recs = []
    for _, r in df.iterrows():
        fname = latest_10k_file(meta, r["ticker"], r["date"])
        if fname is None or not (CLEAN_TEXT / fname).exists():
            recs.append({"tenk_file": fname, "has_10k": False,
                         "as_run_anchor": -1, "excerpt_len": 0,
                         "genuine_header_pos": -1, "genuine_header_in_doc": False,
                         "genuine_header_in_excerpt": False,
                         "header_regex_in_excerpt": False})
            continue
        if fname not in file_cache:
            raw = (CLEAN_TEXT / fname).read_text(encoding="utf-8")
            s, e = original_excerpt_span(raw)
            gpos = genuine_header_pos(raw)
            file_cache[fname] = {
                "as_run_anchor": s,
                "excerpt_len": e - s,
                "genuine_header_pos": -1 if gpos is None else gpos,
                "genuine_header_in_doc": gpos is not None,
                "genuine_header_in_excerpt": gpos is not None and s <= gpos < e,
                "header_regex_in_excerpt": bool(_ITEM1A_HEADER_RE.search(raw[s:e])),
            }
        recs.append({"tenk_file": fname, "has_10k": True, **file_cache[fname]})

    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    df["correct"] = df["pred"] == df["true_label"]
    df.to_csv(OUT_CSV, index=False)

    with_hdr = df[df["genuine_header_in_excerpt"]]
    without_hdr = df[~df["genuine_header_in_excerpt"]]
    summary_rows = [
        {"group": "genuine header in as-run excerpt", **group_metrics(with_hdr)},
        {"group": "no genuine header in excerpt", **group_metrics(without_hdr)},
        {"group": "all", **group_metrics(df)},
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    n10k = int(df["has_10k"].sum())
    print("=== ITEM 1A ANCHOR BUG EXPOSURE AUDIT (as-run replay, n=575) ===")
    print(f"  decisions with a 10-K available: {n10k}")
    print(f"  share with genuine 'Item 1A ... Risk Factors' header in as-run excerpt: "
          f"{df['genuine_header_in_excerpt'].mean():.1%} "
          f"({int(df['genuine_header_in_excerpt'].sum())}/{len(df)})")
    print(f"  (raw header-form regex hit in excerpt, overcounts via cross-refs: "
          f"{df['header_regex_in_excerpt'].mean():.1%})")
    print(f"  filings with NO locatable genuine section at all: "
          f"{(~df['genuine_header_in_doc'] & df['has_10k']).sum()} decisions")
    print("\n  group comparison (reduce recall / grounding):")
    for row in summary_rows:
        print(f"    {row['group']:34} n={row['n']:3} true_reduce={row['n_true_reduce']:3} "
              f"reduce_recall={row['reduce_recall']} grounding_rate={row['grounding_rate']} "
              f"pred_reduce_grounded={row['pred_reduce_grounded_pct']}")
    print(f"\n  wrote {OUT_CSV.relative_to(REPO)} and {OUT_SUMMARY.relative_to(REPO)}")


if __name__ == "__main__":
    main()
