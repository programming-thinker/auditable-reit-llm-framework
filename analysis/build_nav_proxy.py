"""Reconstruct REIT NAV-premium proxies from FREE SEC XBRL data.

There is NO free feed of analyst/consensus NAV (Green Street, S&P are paid; managers'
supplemental NAV is non-standardised). But the academic literature approximates NAV
two ways, both buildable from SEC facts (point-in-time by `filed` date):

  Proxy 1 — Depreciation-adjusted book ("cost NAV"):
     adj_equity = StockholdersEquity + RealEstateAccumulatedDepreciation
     (REIT real estate is carried at depreciated cost; GAAP depreciation is largely
      artificial, so adding it back corrects the worst NAV distortion. Deterministic,
      no assumptions.)  premium = market_cap / adj_equity.

  Proxy 2 — Capitalised-NOI ("market NAV"; the textbook appraisal approach):
     NAV = NOI / cap_rate - net_debt ; NOI ≈ OperatingIncomeLoss + D&A.
     Needs a cap-rate assumption -> report a sensitivity grid {5%, 6.5%, 8%}.

Reads cached SEC JSON from build_fundamentals + panel prices. Writes to
outputs/fundamentals_robustness/. Zone-1 read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
CIKS = REPO / "config/reit_universe_with_cik.csv"
OUT_DIR = REPO / "outputs/fundamentals_robustness"
CACHE = Path("/private/tmp/claude-501/-Users-zhilanc-Desktop-us-reit-data/"
             "9f964396-140e-4e6a-8de0-6e36649e48d8/scratchpad/sec_cache")
CAP_RATES = {"navprem_cap5": 0.05, "navprem_cap65": 0.065, "navprem_cap8": 0.08}


def annual_series(facts, concept, ns="us-gaap"):
    try:
        units = facts["facts"][ns][concept]["units"]
    except KeyError:
        return []
    pts = []
    for uv in units.values():
        for v in uv:
            if v.get("form", "").startswith("10-K") or v.get("fp") == "FY":
                pts.append({"filed": v["filed"], "val": v["val"]})
    pts.sort(key=lambda r: r["filed"])
    return pts


def asof(pts, t):
    best = None
    for p in pts:
        if pd.Timestamp(p["filed"]) <= t:
            best = p["val"]
        else:
            break
    return best


def main() -> None:
    panel = pd.read_csv(PANEL)[["ticker", "date", "Close"]]
    cik_df = pd.read_csv(CIKS, dtype={"cik_str": str})
    cik_map = dict(zip(cik_df["ticker"], cik_df["cik_str"].str.zfill(10)))
    month_ends = sorted(panel["date"].unique())

    rows = []
    for tk, cik in cik_map.items():
        cf = CACHE / f"CIK{cik}.json"
        if not cf.exists():
            continue
        facts = json.loads(cf.read_text())
        S = {c: annual_series(facts, c) for c in
             ["StockholdersEquity", "RealEstateAccumulatedDepreciation",
              "OperatingIncomeLoss", "DepreciationDepletionAndAmortization",
              "LongTermDebt"]}
        shares = annual_series(facts, "EntityCommonStockSharesOutstanding", ns="dei")
        price_by_date = panel[panel["ticker"] == tk].set_index("date")["Close"].to_dict()

        for me in month_ends:
            t = pd.Timestamp(me)
            eq = asof(S["StockholdersEquity"], t)
            acc_dep = asof(S["RealEstateAccumulatedDepreciation"], t)
            oi = asof(S["OperatingIncomeLoss"], t)
            da = asof(S["DepreciationDepletionAndAmortization"], t)
            debt = asof(S["LongTermDebt"], t)
            sh = asof(shares, t)
            price = price_by_date.get(me)
            mktcap = sh * price if (sh and price) else None
            rec = {"ticker": tk, "date": me}

            # Proxy 1: depreciation-adjusted book
            adj_eq = (eq + acc_dep) if (eq is not None and acc_dep is not None) else None
            rec["navprem_book_adj"] = (mktcap / adj_eq) if (mktcap and adj_eq and adj_eq > 0) else np.nan

            # Proxy 2: capitalised NOI, cap-rate sensitivity
            noi = (oi + da) if (oi is not None and da is not None) else None
            for col, cap in CAP_RATES.items():
                nav = (noi / cap - (debt or 0)) if noi is not None else None
                rec[col] = (mktcap / nav) if (mktcap and nav and nav > 0) else np.nan
            rows.append(rec)

    nav = pd.DataFrame(rows)
    nav.to_csv(OUT_DIR / "nav_proxy_panel.csv", index=False)

    feats = ["navprem_book_adj"] + list(CAP_RATES.keys())
    print(f"rows: {len(nav)}  -> {OUT_DIR/'nav_proxy_panel.csv'}\n")
    print(f"{'nav proxy':<20}{'coverage':>10}{'xs_std(avg)':>14}{'median':>10}")
    for f in feats:
        cov = nav[f].notna().mean()
        xs = nav.groupby("date")[f].std(ddof=0).mean()
        med = nav[f].median()
        print(f"{f:<20}{cov:>10.1%}{xs:>14.4g}{med:>10.3f}")

    # how different is the depreciation-adjusted premium from plain book-to-market?
    fund = pd.read_csv(OUT_DIR / "firm_fundamentals_panel.csv")[["ticker", "date", "book_to_market"]]
    m = nav.merge(fund, on=["ticker", "date"], how="left")
    m["price_to_book_plain"] = 1.0 / m["book_to_market"]
    corr = m[["navprem_book_adj", "price_to_book_plain"]].corr().iloc[0, 1]
    print(f"\ncorr(depreciation-adjusted NAV premium, plain price-to-book) = {corr:.3f}")
    print("(low/moderate corr => the depreciation add-back carries info beyond plain B/M)")


if __name__ == "__main__":
    main()
