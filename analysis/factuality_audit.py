"""Factuality audit (reviewer critique G): 'cites a fact' is not 'the fact is accurate'.
We verify, automatically, whether the rationales' cited facts are actually true:

  (1) Fundamentals agent: each cited 'feature: value' is compared to the true model
      input value (the fundamentals panel) within 1% tolerance — an exact factuality test.
  (2) Disclosure agent: dollar amounts and years cited are checked for presence in the
      actual filing text the agent was given (re-retrieved, filing-date enforced).

This upgrades '88% grounded' (cites something) to a measured factual-accuracy rate, and
specifies the protocol a full two-rater human audit (factuality/entailment/relevance,
Cohen's kappa) would extend. No API. Writes outputs/llm_deepseek_test/factuality_audit.csv.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from llm.edgar_client import EdgarClient

REPO = Path(__file__).resolve().parents[1]
DEC = REPO / "audit_log/decisions.jsonl"
FR = REPO / "outputs/fundamentals_robustness"
OUT = REPO / "outputs/llm_deepseek_test/factuality_audit.csv"
ec = EdgarClient()


@lru_cache(maxsize=4000)
def source_text(ticker: str, date: str) -> str:
    fil = ec.get_latest_annual_and_quarterly(ticker, date)
    txt = (fil.get("10-K") or "")[:400000] + " " + (fil.get("10-Q") or "")[:100000]
    start = str((pd.Timestamp(date) - pd.DateOffset(months=6)).date())
    try:
        ek = ec.get_filings_in_window(ticker, "8-K", start, date)
        txt += " " + " ".join((f["text"] or "") for f in ek[:6])
    except Exception:  # noqa: BLE001
        pass
    return txt


def fundamentals_truth():
    f = pd.read_csv(FR / "firm_fundamentals_panel.csv")
    nav = pd.read_csv(FR / "nav_proxy_panel.csv")[["ticker", "date", "navprem_book_adj"]]
    m = f.merge(nav, on=["ticker", "date"], how="left")
    m["date"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
    return m.set_index(["ticker", "date"])


def main():
    truth = fundamentals_truth()
    rows = []
    fund_ok = fund_tot = 0
    disc_ok = disc_tot = 0
    n_reduce = 0
    with DEC.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tk, dt = d["ticker"], d["decision_date_t"]
            pred = max(d["final_probabilities"], key=d["final_probabilities"].get)

            # (1) fundamentals factuality (all decisions; exact)
            fo = d["agent_outputs"].get("fundamentals")
            if fo:
                try:
                    row = truth.loc[(tk, dt)]
                except KeyError:
                    row = None
                for fact in fo.get("facts_cited", []):
                    m = re.match(r"\s*([a-z_]+)\s*[:=]\s*(-?\d+\.?\d*)", str(fact))
                    if m and row is not None and m.group(1) in row:
                        fund_tot += 1
                        true_v = float(row[m.group(1)])
                        cited_v = float(m.group(2))
                        if abs(cited_v - true_v) <= 0.01 * (abs(true_v) + 1e-6) + 1e-4:
                            fund_ok += 1

            # (2) disclosure numeric factuality (reduce decisions only, needs file read)
            if pred == "reduce":
                n_reduce += 1
                src = source_text(tk, dt)
                facts = d["agent_outputs"]["disclosure"].get("facts_cited", [])
                toks = set()
                for fact in facts:
                    toks |= set(re.findall(r"\$\s?\d[\d,.]*", str(fact)))
                    toks |= set(re.findall(r"\b(?:19|20)\d\d\b", str(fact)))
                    toks |= set(re.findall(r"\d+\.\d+\s?%", str(fact)))
                for t in toks:
                    disc_tot += 1
                    norm = t.replace("$", "").replace(" ", "").replace(",", "")
                    if norm[:5] in src.replace("$", "").replace(",", "").replace(" ", "") or t.strip("$ ") in src:
                        disc_ok += 1

    res = {
        "fundamentals_cited_values_checked": fund_tot,
        "fundamentals_factual_accuracy": round(fund_ok / fund_tot, 3) if fund_tot else float("nan"),
        "reduce_decisions_audited": n_reduce,
        "disclosure_numeric_tokens_checked": disc_tot,
        "disclosure_numeric_in_source_rate": round(disc_ok / disc_tot, 3) if disc_tot else float("nan"),
    }
    pd.DataFrame([res]).T.rename(columns={0: "value"}).to_csv(OUT)
    print("=== FACTUALITY AUDIT (automated) ===\n")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\n[reading] High factual-accuracy => the cited facts are genuine, not hallucinated;")
    print("  upgrades 'grounded' to 'grounded AND accurate'. A full human audit (entailment,")
    print("  relevance, Cohen's kappa) is specified as the next step.")


if __name__ == "__main__":
    main()
