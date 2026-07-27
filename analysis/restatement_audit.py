"""Point-in-time restatement audit (reviewer critique: 'prove the XBRL data is truly
point-in-time, not merely filing-date enforced').

The fundamentals pipeline (analysis/build_fundamentals.py::asof) attaches, to each
decision month t, the value of a concept (Assets, LongTermDebt, ...) from the most
recent annual datapoint whose SEC `filed` date <= t. This is a *valid* point-in-time
definition — every value used was publicly available at t — but it is NOT the stricter
'as-originally-reported' standard: if a fiscal period was later restated (10-K/A, or a
subsequent-year 10-K re-filing prior-year comparatives) and that restatement was itself
filed on or before t, asof() returns the restated value.

This script QUANTIFIES that exposure on the actual test-window decisions, so the thesis
can bound it rather than hand-wave. For every (ticker, concept) used at each test
decision date, it replicates asof() and checks whether the value used is the
as-originally-reported value for its fiscal period or a later restatement, and by how
much they differ. No look-ahead is involved (asof enforces filed <= t); the question is
purely original-vs-restated.

Read-only: reads cached SEC companyfacts JSON (same cache build_fundamentals used) and
audit_log/decisions.jsonl. Writes outputs/llm_deepseek_test/restatement_audit.csv.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEC = REPO / "audit_log/decisions.jsonl"
CIKS = REPO / "config/reit_universe_with_cik.csv"
OUT = REPO / "outputs/llm_deepseek_test/restatement_audit.csv"
# cache written by analysis/build_fundamentals.py (prior session scratchpad); fall back to SEC
CACHE_DIRS = [
    Path("/private/tmp/claude-501/-Users-zhilanc-Desktop-us-reit-data/"
         "9f964396-140e-4e6a-8de0-6e36649e48d8/scratchpad/sec_cache"),
    Path("/private/tmp/claude-501/-Users-zhilanc-Desktop-us-reit-data/"
         "1a583b76-114e-47c0-987f-bb5a776ecc60/scratchpad/sec_cache"),
]
UA = "thesis-research zhilanc lankunchen2001@gmail.com"
# concepts build_fundamentals.py actually consumes
CONCEPTS = [("Assets", "us-gaap"), ("Liabilities", "us-gaap"), ("LongTermDebt", "us-gaap"),
            ("InterestExpense", "us-gaap"), ("StockholdersEquity", "us-gaap"),
            ("NetIncomeLoss", "us-gaap"),
            ("DepreciationDepletionAndAmortization", "us-gaap"),
            ("EntityCommonStockSharesOutstanding", "dei")]


def load_companyfacts(cik: str) -> dict | None:
    for d in CACHE_DIRS:
        cf = d / f"CIK{cik}.json"
        if cf.exists():
            return json.loads(cf.read_text())
    # offline cache miss -> fetch once (read-only external, SEC fair-access)
    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        data = json.load(urllib.request.urlopen(req, timeout=30))
        dest = CACHE_DIRS[-1]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"CIK{cik}.json").write_text(json.dumps(data))
        time.sleep(0.2)
        return data
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] companyfacts fetch failed CIK{cik}: {type(e).__name__}")
        return None


def annual_series(facts: dict, concept: str, ns: str) -> list[dict]:
    """Mirror build_fundamentals.annual_series: annual (10-K/FY) points sorted by filed."""
    try:
        units = facts["facts"][ns][concept]["units"]
    except KeyError:
        return []
    pts = []
    for unit_vals in units.values():
        for v in unit_vals:
            form = v.get("form", "")
            if form.startswith("10-K") or v.get("fp") == "FY":
                pts.append({"filed": v["filed"], "val": v["val"],
                            "end": v.get("end"), "form": form})
    pts.sort(key=lambda r: r["filed"])
    return pts


def used_point(pts: list[dict], t: pd.Timestamp) -> dict | None:
    """Mirror asof(): the most recent point with filed <= t (keep the whole point)."""
    best = None
    for p in pts:
        if pd.Timestamp(p["filed"]) <= t:
            best = p
        else:
            break
    return best


def main() -> None:
    cik_df = pd.read_csv(CIKS, dtype={"cik_str": str})
    cik_map = dict(zip(cik_df["ticker"], cik_df["cik_str"].str.zfill(10)))

    # test-window decision (ticker, date) pairs from the audit log
    pairs = []
    with DEC.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                pairs.append((d["ticker"], pd.Timestamp(d["decision_date_t"])))
    pairs = sorted(set(pairs))
    print(f"[restatement-audit] {len(pairs)} unique (ticker, decision-date) pairs "
          f"over the test window\n")

    facts_cache: dict[str, dict | None] = {}
    rows = []
    n_uses = n_restated = 0
    reldiffs = []
    for tk, t in pairs:
        cik = cik_map.get(tk)
        if cik is None:
            continue
        if cik not in facts_cache:
            facts_cache[cik] = load_companyfacts(cik)
        facts = facts_cache[cik]
        if facts is None:
            continue
        for concept, ns in CONCEPTS:
            pts = annual_series(facts, concept, ns)
            if not pts:
                continue
            up = used_point(pts, t)
            if up is None:
                continue
            n_uses += 1
            # earliest-filed datapoint for the SAME fiscal period -> as-originally-reported
            same_end = [p for p in pts if p["end"] == up["end"]]
            first = min(same_end, key=lambda p: p["filed"])
            restated = (up["filed"] != first["filed"]) and (up["val"] != first["val"])
            if restated:
                n_restated += 1
                denom = abs(first["val"]) if first["val"] else np.nan
                rd = abs(up["val"] - first["val"]) / denom if denom else np.nan
                reldiffs.append(rd)
                rows.append({"ticker": tk, "decision_date": t.strftime("%Y-%m-%d"),
                             "concept": concept, "period_end": up["end"],
                             "original_filed": first["filed"], "original_val": first["val"],
                             "used_filed": up["filed"], "used_val": up["val"],
                             "rel_diff": round(rd, 5) if rd == rd else np.nan,
                             "used_form": up["form"]})

    # ---- context: prevalence of restatements in the RAW data (proves detector works) ----
    raw_periods = raw_restated = 0
    for cik, facts in facts_cache.items():
        if facts is None:
            continue
        for concept, ns in CONCEPTS:
            byend: dict = {}
            for p in annual_series(facts, concept, ns):
                byend.setdefault(p["end"], set()).add(p["val"])
            for vals in byend.values():
                raw_periods += 1
                if len(vals) > 1:
                    raw_restated += 1

    detail = pd.DataFrame(rows)
    rd = np.array([x for x in reldiffs if x == x])  # drop nan
    summary = {
        "raw_firm_period_concepts": raw_periods,
        "raw_restated_periods": raw_restated,
        "raw_restated_share": round(raw_restated / raw_periods, 4) if raw_periods else float("nan"),
        "value_uses_audited": n_uses,
        "restated_value_uses": n_restated,
        "restated_share": round(n_restated / n_uses, 4) if n_uses else float("nan"),
        "reldiff_median": round(float(np.median(rd)), 4) if rd.size else float("nan"),
        "reldiff_p90": round(float(np.percentile(rd, 90)), 4) if rd.size else float("nan"),
        "reldiff_max": round(float(rd.max()), 4) if rd.size else float("nan"),
        "reldiff_gt_5pct_share": round(float((rd > 0.05).mean()), 4) if rd.size else float("nan"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUT, index=False)
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        OUT.with_name("restatement_audit_summary.csv"))

    print("=== RESTATEMENT AUDIT (test-window value-uses) ===\n")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n[written] {OUT}")
    print(f"[written] {OUT.with_name('restatement_audit_summary.csv')}")
    print("\n[reading] Every value used was publicly filed <= decision date (no look-ahead).")
    print("  This bounds the stricter 'as-originally-reported' exposure: the restated share")
    print("  and its magnitude quantify how far the PIT panel departs from first-filed values.")


if __name__ == "__main__":
    main()
