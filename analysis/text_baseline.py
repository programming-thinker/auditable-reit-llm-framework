"""Transparent text baseline (reviewer critique F): does the SEC-disclosure TEXT
itself carry firm-level reduce signal, independent of the LLM design?

We build, for each decision, the same disclosure blob the LLM Disclosure agent sees
(Item 1A risk factors from the latest 10-K + 8-K texts in the prior 6 months,
filing-date enforced), then fit two transparent classifiers on train (2015-2021) and
evaluate on test (2024-2025):
  (1) TF-IDF (top 3000) + multinomial logistic (class-weight balanced);
  (2) Loughran-McDonald negative/uncertainty word fractions + logistic.
We compare reduce recall to the random-at-budget baseline and the LLM framework.

If these transparent text models also ~ random, the TEXT lacks firm-level reduce signal
(not merely the LLM). No API. Writes outputs/llm_deepseek_test/text_baseline.csv.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from llm.edgar_client import EdgarClient

REPO = Path(__file__).resolve().parents[1]
SPLIT = REPO / "data/processed/splits"
OUT = REPO / "outputs/llm_deepseek_test/text_baseline.csv"
SEED = 20260626

# compact Loughran-McDonald negative / uncertainty subset (representative)
LM_NEG = """loss losses adverse adversely decline declined decrease decreased default
deteriorate deteriorating impairment litigation lawsuit bankruptcy insolvency covenant
breach downgrade weakness weaknesses risk risks volatile volatility vacancy delinquent
delinquency restructuring writeoff write-down shortfall deficiency liabilities terminate
termination unable failure failed concern doubt""".split()
LM_UNC = """may might could uncertain uncertainty possible possibly approximately
contingent depends dependent risk anticipate believe estimate assume potential""".split()

ec = EdgarClient()


def item1a(raw):
    if not raw:
        return ""
    s = [m.start() for m in re.finditer(r"item\s*1a", raw, re.I)]
    e = [m.start() for m in re.finditer(r"item\s*1b", raw, re.I)]
    if not s:
        m = re.search(r"risk factors", raw, re.I)
        return raw[m.start():m.start() + 15000] if m else raw[:15000]
    st = s[0]
    en = next((x for x in e if x > st), st + 15000)
    return raw[st:min(en, st + 15000)]


@lru_cache(maxsize=4000)
def blob(ticker: str, date: str) -> str:
    fil = ec.get_latest_annual_and_quarterly(ticker, date)
    txt = item1a(fil.get("10-K"))
    start = str((pd.Timestamp(date) - pd.DateOffset(months=6)).date())
    try:
        ek = ec.get_filings_in_window(ticker, "8-K", start, date)
        # ek is ASCENDING by filing_date -> [-5:] keeps the 5 NEWEST (fixed
        # 2026-07-04; [:5] kept the 5 oldest). Reported text_baseline.csv was
        # produced with the pre-fix slice.
        txt += " " + " ".join((f["text"] or "")[:3000] for f in ek[-5:])
    except Exception:  # noqa: BLE001
        pass
    return re.sub(r"[^A-Za-z ]", " ", txt).lower()


def lm_feats(text):
    w = text.split()
    n = max(len(w), 1)
    neg = sum(1 for x in w if x in set(LM_NEG))
    unc = sum(1 for x in w if x in set(LM_UNC))
    return [neg / n, unc / n, len(w)]


def metrics(y, pred, label):
    tp = int(((pred == "reduce") & (y == "reduce")).sum())
    fp = int(((pred == "reduce") & (y != "reduce")).sum())
    fn = int(((pred != "reduce") & (y == "reduce")).sum())
    return {"model": label,
            "reduce_recall": round(tp / (tp + fn), 3) if (tp + fn) else 0.0,
            "reduce_precision": round(tp / (tp + fp), 3) if (tp + fp) else 0.0,
            "reduce_fire_rate": round(float((pred == "reduce").mean()), 3),
            "accuracy": round(float((pred == y).mean()), 3)}


def main():
    tr = pd.read_csv(SPLIT / "enriched_train_2015_2021.csv")
    te = pd.read_csv(SPLIT / "enriched_test_2024_2025.csv")
    for d in (tr, te):
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    print(f"building text blobs: train={len(tr)} test={len(te)} ...")
    tr["text"] = [blob(t, d) for t, d in zip(tr["ticker"], tr["date"])]
    te["text"] = [blob(t, d) for t, d in zip(te["ticker"], te["date"])]

    rows = []
    # (1) TF-IDF + logistic
    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=5)
    Xtr = vec.fit_transform(tr["text"]); Xte = vec.transform(te["text"])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="multinomial").fit(Xtr, tr["label"])
    rows.append(metrics(te["label"].values, clf.predict(Xte), "TF-IDF + logistic"))
    # (2) LM sentiment + logistic
    Ftr = np.array([lm_feats(t) for t in tr["text"]]); Fte = np.array([lm_feats(t) for t in te["text"]])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    clf2 = LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="multinomial").fit(sc.transform(Ftr), tr["label"])
    rows.append(metrics(te["label"].values, clf2.predict(sc.transform(Fte)), "Loughran-McDonald sentiment + logistic"))

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    base = (te["label"] == "reduce").mean()
    print(f"\n=== TRANSPARENT TEXT BASELINE (test 575, reduce base rate {base:.3f}) ===\n")
    print(res.to_string(index=False))
    print(f"\n  reference: LLM 5-agent reduce recall 0.206; random-at-budget ~0.204")
    print("  VERDICT: if these text models ~ random too, the TEXT itself lacks firm-level")
    print("           reduce signal (answering the reviewer's incremental-information question).")


if __name__ == "__main__":
    main()
