"""Build a firm-level REIT fundamentals panel to fill the feature gap identified
in FEATURE_INVENTORY_AND_GAPS.md, using ONLY free sources:

  (A) Local daily prices (data/raw/prices/daily_prices.csv):
        - amihud_illiq   : Amihud (2002) illiquidity
        - idio_vol       : idiosyncratic volatility, market-model residual (Ang et al. 2006)
  (B) SEC EDGAR XBRL companyfacts (free API), point-in-time by `filed` date:
        - leverage        : LongTermDebt / Assets
        - debt_to_equity  : Liabilities / StockholdersEquity
        - interest_cover  : (NetIncome + InterestExpense + D&A) / InterestExpense
        - book_to_market  : StockholdersEquity / market cap        (Fama-French 1992)
        - ln_mktcap       : ln(market cap)                          (size)
        - ffo_yield_proxy : (NetIncome + D&A) / market cap          (FFO proxy; Vincent 1999)

LOOK-AHEAD DISCIPLINE: every fundamental at month-end t uses the most recent
annual (10-K) figure whose SEC `filed` date <= t. This mirrors the filing-date
enforcement the thesis already applies to text (CLAUDE.md / edgar_client.py).

Writes outputs/fundamentals_robustness/firm_fundamentals_panel.csv. Does NOT touch
Zone 1 (reads data/* read-only; writes only to the new output dir). SEC JSON is
cached under the scratchpad so re-runs are offline.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
DAILY = REPO / "data/raw/prices/daily_prices.csv"
CIKS = REPO / "config/reit_universe_with_cik.csv"
OUT_DIR = REPO / "outputs/fundamentals_robustness"
CACHE = Path("/private/tmp/claude-501/-Users-zhilanc-Desktop-us-reit-data/"
             "9f964396-140e-4e6a-8de0-6e36649e48d8/scratchpad/sec_cache")
UA = "thesis-research zhilanc lankunchen2001@gmail.com"


# --------------------------------------------------------------------------- #
# (A) Local price-based features
# --------------------------------------------------------------------------- #
def local_features(panel_dates: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(DAILY, parse_dates=["Date"]).sort_values(["ticker", "Date"])
    d = d.rename(columns={"Date": "date"})
    d["ret"] = d.groupby("ticker")["Adj Close"].pct_change()
    d["dollar_vol"] = d["Volume"] * d["Close"]
    d["amihud_daily"] = (d["ret"].abs() / d["dollar_vol"].replace(0, np.nan)) * 1e9
    # equal-weight market daily return (cross-sectional mean over all tickers)
    mkt = d.groupby("date")["ret"].mean().rename("mkt_ret")
    d = d.merge(mkt, on="date", how="left")

    out = []
    month_ends = sorted(panel_dates["date"].unique())
    for tk, g in d.groupby("ticker"):
        g = g.set_index("date").sort_index()
        for me in month_ends:
            me = pd.Timestamp(me)
            win = g.loc[:me]
            if len(win) < 30:
                continue
            # Amihud: mean over the trailing month (~21 trading days)
            amihud = win["amihud_daily"].tail(21).mean()
            # idiosyncratic vol: residual std from market model over trailing 63 days
            w = win.tail(63).dropna(subset=["ret", "mkt_ret"])
            idio = np.nan
            if len(w) >= 40 and w["mkt_ret"].std() > 0:
                x = w["mkt_ret"].values
                y = w["ret"].values
                beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
                alpha = y.mean() - beta * x.mean()
                resid = y - (alpha + beta * x)
                idio = resid.std(ddof=1) * np.sqrt(252)
            out.append({"ticker": tk, "date": me.strftime("%Y-%m-%d"),
                        "amihud_illiq": amihud, "idio_vol": idio})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# (B) SEC XBRL fundamentals
# --------------------------------------------------------------------------- #
def fetch_companyfacts(cik: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"CIK{cik}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    cf.write_text(json.dumps(data))
    time.sleep(0.2)  # SEC fair-access
    return data


def annual_series(facts: dict, concept: str, ns: str = "us-gaap") -> list[dict]:
    """Return annual (10-K / FY) datapoints: [{filed, val}] sorted by filed."""
    try:
        units = facts["facts"][ns][concept]["units"]
    except KeyError:
        return []
    pts = []
    for unit_vals in units.values():
        for v in unit_vals:
            form = v.get("form", "")
            if form.startswith("10-K") or v.get("fp") == "FY":
                pts.append({"filed": v["filed"], "val": v["val"], "end": v.get("end")})
    # keep latest val per filed date
    pts.sort(key=lambda r: r["filed"])
    return pts


def asof(pts: list[dict], t: pd.Timestamp):
    """Most recent val with filed <= t."""
    best = None
    for p in pts:
        if pd.Timestamp(p["filed"]) <= t:
            best = p["val"]
        else:
            break
    return best


def sec_features(panel: pd.DataFrame, cik_map: dict) -> pd.DataFrame:
    concepts = ["Assets", "Liabilities", "LongTermDebt", "InterestExpense",
                "StockholdersEquity", "NetIncomeLoss",
                "DepreciationDepletionAndAmortization"]
    month_ends = sorted(panel["date"].unique())
    rows = []
    for tk, cik in cik_map.items():
        try:
            facts = fetch_companyfacts(cik)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] SEC fetch failed {tk}: {type(e).__name__}")
            continue
        series = {c: annual_series(facts, c) for c in concepts}
        shares = annual_series(facts, "EntityCommonStockSharesOutstanding", ns="dei")
        # price at each month-end from the panel (Close)
        price_by_date = panel[panel["ticker"] == tk].set_index("date")["Close"].to_dict()
        for me in month_ends:
            t = pd.Timestamp(me)
            vals = {c: asof(series[c], t) for c in concepts}
            sh = asof(shares, t)
            price = price_by_date.get(me)
            mktcap = sh * price if (sh and price) else None
            rec = {"ticker": tk, "date": me}
            assets = vals["Assets"]
            ltd = vals["LongTermDebt"]
            liab = vals["Liabilities"]
            eq = vals["StockholdersEquity"]
            ni = vals["NetIncomeLoss"]
            ie = vals["InterestExpense"]
            da = vals["DepreciationDepletionAndAmortization"]
            rec["leverage"] = (ltd / assets) if (ltd and assets) else np.nan
            rec["debt_to_equity"] = (liab / eq) if (liab and eq and eq != 0) else np.nan
            rec["interest_cover"] = ((ni + ie + (da or 0)) / ie) if (ni is not None and ie) else np.nan
            rec["book_to_market"] = (eq / mktcap) if (eq and mktcap) else np.nan
            rec["ln_mktcap"] = float(np.log(mktcap)) if mktcap and mktcap > 0 else np.nan
            rec["ffo_yield_proxy"] = ((ni + (da or 0)) / mktcap) if (ni is not None and mktcap) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL)[["ticker", "date", "Close", "label"]]
    cik_df = pd.read_csv(CIKS, dtype={"cik_str": str})
    cik_map = dict(zip(cik_df["ticker"], cik_df["cik_str"].str.zfill(10)))

    print("[1/3] local price features (Amihud, idio vol)...")
    loc = local_features(panel[["date"]].drop_duplicates())

    print("[2/3] SEC XBRL fundamentals (PIT by filed date)...")
    sec = sec_features(panel, cik_map)

    print("[3/3] merge + cross-sectional variation...")
    fund = panel.merge(loc, on=["ticker", "date"], how="left").merge(
        sec, on=["ticker", "date"], how="left")
    out_path = OUT_DIR / "firm_fundamentals_panel.csv"
    fund.to_csv(out_path, index=False)

    new_feats = ["amihud_illiq", "idio_vol", "leverage", "debt_to_equity",
                 "interest_cover", "book_to_market", "ln_mktcap", "ffo_yield_proxy"]
    print(f"\nrows: {len(fund)}  written -> {out_path}\n")
    print(f"{'feature':<18}{'coverage':>10}{'xs_std(avg)':>14}{'has_signal':>12}")
    rep = []
    for f in new_feats:
        cov = fund[f].notna().mean()
        xs = fund.groupby("date")[f].std(ddof=0).mean()
        sig = bool(xs and xs > 1e-9)
        print(f"{f:<18}{cov:>10.2%}{xs:>14.4g}{str(sig):>12}")
        rep.append({"feature": f, "coverage": cov, "xs_std_avg": xs, "has_signal": sig})
    pd.DataFrame(rep).to_csv(OUT_DIR / "fundamentals_coverage_variation.csv", index=False)
    print("\nAll new firm-level features should show has_signal=True "
          "(unlike the 64 macro vars with xs_std=0).")


if __name__ == "__main__":
    main()
