"""Masked-identity contamination probe (thesis limitation #1).

Question: does the 5-agent DeepSeek V4-Flash framework's behaviour depend on
KNOWING WHICH REIT it is looking at (memorised training-data knowledge of the
2024-2025 test window = contamination), or only on the point-in-time inputs we
feed it? We re-run the same stratified 70-decision test subset used by
analysis/reasoner_subset.py (SEED=20260626, 40 true-reduce + 30 other), but
with firm identity withheld from every agent:

  - ticker  -> "REIT-X"
  - company -> "a large-cap U.S. equity REIT"
  - filing_10k_text / filing_10q_text / filing_8k_texts scrubbed of the
    company's full name, its distinctive name tokens, its ticker symbol,
    and SEC CIK / accession numbers (replaced with "the Company" / "[ID]").

DESIGN CHOICE — sector is KEPT. Property sector (e.g. "Retail", "Data Center")
is an economic feature the structured baseline also sees, not an identity
string; withholding it would confound "no memorisation" with "less economic
information". With 25 large-cap REITs across ~14 sectors, sector alone does
not pin down the firm in most cases (documented residual: single-firm sectors
like Gaming Net Lease could in principle be inferred).

KNOWN RESIDUAL LEAKS (documented, not scrubbed): property addresses, executive
names, subsidiary/brand names other than the curated aliases, and peer-firm
mentions inside filing text could in principle identify the firm to a
sufficiently motivated reader. The probe removes all EXPLICIT identity, which
is the channel a contaminated model would use first.

Mechanics: subclass llm.orchestrator.Orchestrator and override the four input
builders (call super(), then mask); the aggregator's inline inputs (built
inside run_single) are masked by monkeypatching llm.orchestrator's module-level
reference to run_aggregator_agent — no file under llm/ is modified. A leak
guard wraps LLMClient.query_messages and scans EVERY rendered message for the
decision's identity patterns BEFORE the API call, raising (and thus spending
nothing) on any hit; scan results are appended to
audit_log/masked_probe/masking_scan.jsonl as verification evidence.

Masked prompts differ from the unmasked run's prompts, so the diskcache
(key = sha256(messages+model+params)) guarantees fresh API calls.

Usage (from repo root, live API — DeepSeek key in .env):
    python analysis/masked_identity_subset.py scan   # all 70, NO API calls:
                                                     #   render + leak-scan only
    python analysis/masked_identity_subset.py dry    # 3 live decisions + prompt dump
    python analysis/masked_identity_subset.py full   # all 70 live decisions

Audit dir: audit_log/masked_probe/ (new, append-only; never the main audit_log).
Output:    outputs/llm_deepseek_test/masked_probe_comparison.csv
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Pattern

import pandas as pd

import llm.orchestrator as orch_mod
from llm.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parents[1]
TEST = REPO / "data/processed/splits/enriched_test_2024_2025.csv"
FLASH = REPO / "audit_log/decisions.jsonl"            # unmasked v2 flash (575)
UNIVERSE = REPO / "config/reit_universe.csv"
PANEL = REPO / "data/processed/backtest_ready_panel_enriched.csv"
AUDIT_DIR = REPO / "audit_log/masked_probe"           # new dir, append-only
SCAN_LOG = AUDIT_DIR / "masking_scan.jsonl"
DRYRUN_PROMPTS = AUDIT_DIR / "dryrun_rendered_prompts.txt"
SCAN_PROMPTS = AUDIT_DIR / "scan_rendered_prompts.txt"
OUT = REPO / "outputs/llm_deepseek_test/masked_probe_comparison.csv"

SEED = 20260626          # same as reasoner_subset.py -> identical 70 decisions
N_RED, N_OTH = 40, 30
MODEL = "deepseek_v4_flash"   # same model as the headline unmasked run
BUDGET_USD = 5.0              # hard cap for this probe

MASK_TICKER = "REIT-X"
MASK_COMPANY = "a large-cap U.S. equity REIT"
MASK_TEXT = "the Company"

# Generic name tokens that appear in ordinary filing prose / other firms'
# names; scrubbing them standalone would destroy text without hiding identity.
# The full-name PHRASE is always scrubbed regardless of this list.
GENERIC_TOKENS = {
    "american", "america", "mid", "tower", "property", "properties", "group",
    "communities", "realty", "income", "residential", "equity", "equities",
    "digital", "public", "storage", "trust", "real", "estate", "extra",
    "space", "apartment", "apartments", "centers", "homes", "inc", "corp",
    "corporation", "the", "and", "of",
}

# Curated extra aliases (former names, operating-partnership names, brands)
# not derivable from the two identity sources (panel 'company' column,
# config/reit_universe.csv). "BPLP" was found in BXP's actual 10-K text
# during dry-run verification.
EXTRA_ALIASES: Dict[str, List[str]] = {
    "BXP": ["Boston Properties", "BPLP"],
    "UDR": ["United Dominion"],
    "AVB": ["Avalon"],
    "MAA": ["Mid-America", "MidAmerica", "MAALP"],
    "EXR": ["Extra Space"],
    "PSA": ["Shurgard", "SHUR"],   # SHUR = Shurgard's Euronext ticker (found
    "EQR": ["ERP Operating"],      # in PSA 10-K text during verification)
    "AMT": ["ATC"],
}

# Tickers that collide with high-frequency English words / names when matched
# case-insensitively ("are", "well", "o", "kim"): matched CASE-SENSITIVELY as
# standalone tokens; their URL / XBRL-prefix forms are still caught by the
# targeted case-insensitive context patterns below.
AMBIGUOUS_TICKERS = {"O", "ARE", "WELL", "KIM"}

# Same grounding rubric as analysis/reasoner_subset.py (comparability).
RISK = ["covenant", "default", "litigation", "impairment", "tenant", "vacanc",
        "lease termination", "going concern", "refinanc", "maturity", "dividend cut",
        "downgrade", "leverage", "occupancy", "guidance"]


def grounded(rec: dict) -> bool:
    ao = rec["agent_outputs"]
    facts = (ao["disclosure"].get("facts_cited") or []) + \
            ((ao.get("fundamentals") or {}).get("facts_cited") or [])
    blob = " ".join(str(x) for x in facts) + " " + ao["disclosure"].get("rationale", "")
    has_money = bool(re.search(r"\$\s?\d|\d+(?:\.\d+)?\s?%", blob))
    has_risk = any(t in blob.lower() for t in RISK)
    return len(facts) >= 1 and (has_money or has_risk)


def amax(p: dict) -> str:
    return max(p, key=p.get)


# ── identity patterns ─────────────────────────────────────────────────────


def _phrase_pattern(name: str) -> Pattern[str]:
    """Case-insensitive pattern for a name phrase, flexible separators.

    Prefix boundary prevents in-word matches ("ATC" in "matching"); the
    suffix is deliberately unguarded so CamelCase XBRL identifiers such as
    "bxp:BostonPropertiesLimitedPartnershipMember" are still caught.
    """
    toks = [re.escape(t) for t in re.split(r"[^A-Za-z0-9]+", name) if t]
    body = r"[\s\-–,\.]{0,3}".join(toks)
    return re.compile(rf"(?<![A-Za-z0-9]){body}", re.IGNORECASE)


def build_identity_patterns(ticker: str, names: List[str]) -> List[Pattern[str]]:
    """Patterns identifying one firm: full-name phrases, distinctive tokens,
    ticker symbol (case-sensitive), CIK / accession numbers."""
    pats: List[Pattern[str]] = []
    tokens: set[str] = set()
    for name in names:
        if not name or name.upper() == ticker:
            continue  # name==ticker (BXP, UDR) handled by the ticker pattern
        pats.append(_phrase_pattern(name))
        for t in re.split(r"[^A-Za-z0-9]+", name):
            if len(t) >= 4 and t.lower() not in GENERIC_TOKENS:
                tokens.add(t.lower())
    for t in sorted(tokens):
        pats.append(re.compile(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])",
                               re.IGNORECASE))
    # ticker as standalone token. Case-insensitive by default (catches
    # possessives "BXP's", lowercase URL/logo/XBRL usage "bxp"); tickers that
    # collide with common English words are matched case-sensitively instead.
    tk = re.escape(ticker)
    flags = 0 if ticker in AMBIGUOUS_TICKERS else re.IGNORECASE
    pats.append(re.compile(rf"(?<![A-Za-z0-9]){tk}(?![A-Za-z0-9])", flags))
    # targeted contexts for ALL tickers (incl. ambiguous ones): web domains
    # ("www.are.com"), XBRL tag prefixes ("are:XyzMember"), and NYSE
    # preferred-share symbols ("KIMprL" — found in KIM 10-K during
    # verification; suffix is alnum so the standalone pattern misses it)
    pats.append(re.compile(rf"(?<![A-Za-z0-9]){tk}(?=\.(?:com|net|org)\b)",
                           re.IGNORECASE))
    pats.append(re.compile(rf"(?<![A-Za-z0-9]){tk}(?=:[A-Za-z])", re.IGNORECASE))
    pats.append(re.compile(rf"(?<![A-Za-z0-9]){tk}(?=pr[A-Za-z])", re.IGNORECASE))
    return pats


# SEC identifiers: 10-digit CIK and accession numbers (also scanned).
CIK_PATTERNS = [
    re.compile(r"\b\d{10}-\d{2}-\d{6}\b"),
    re.compile(r"\b\d{10}\b"),
]


def scrub(text: str | None, patterns: List[Pattern[str]]) -> str | None:
    if text is None:
        return None
    for pat in patterns:
        text = pat.sub(MASK_TEXT, text)
    for pat in CIK_PATTERNS:
        text = pat.sub("[ID]", text)
    return text


def scan_hits(text: str, patterns: List[Pattern[str]]) -> List[str]:
    hits: List[str] = []
    for pat in patterns + CIK_PATTERNS:
        for m in pat.finditer(text):
            hits.append(f"{pat.pattern[:40]!r}->{m.group(0)[:30]!r}@{m.start()}")
    return hits


# ── aggregator masking (monkeypatch; llm/ files untouched) ────────────────

_REAL_RUN_AGGREGATOR = orch_mod.run_aggregator_agent


def _masked_run_aggregator(inputs: dict, *args: Any, **kwargs: Any):
    masked = dict(inputs)
    masked["ticker"] = MASK_TICKER
    masked["company"] = MASK_COMPANY
    return _REAL_RUN_AGGREGATOR(masked, *args, **kwargs)


orch_mod.run_aggregator_agent = _masked_run_aggregator


# ── masked orchestrator ───────────────────────────────────────────────────


class LeakError(RuntimeError):
    """Identity string found in a rendered prompt; API call aborted."""


# Fake response for scan-only mode: the union of all agent output fields
# (Pydantic ignores extras), so every agent parses it and the full 5-agent
# pipeline renders every prompt without a single API call.
_FAKE_CONTENT = json.dumps({
    "probabilities": {"increase": 0.34, "hold": 0.33, "reduce": 0.33},
    "rationale": "scan-only fake response (no API call)",
    "facts_cited": [],
    "sentiment": "neutral",
    "regime_label": "neutral",
    "momentum_state": "neutral",
    "financial_health": "adequate",
    "agreement_score": 0.5,
})


class MaskedOrchestrator(Orchestrator):
    """Orchestrator with firm identity withheld from every agent prompt."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # identity variants from BOTH sources: panel 'company' + universe csv
        panel_names = (
            pd.read_csv(PANEL, usecols=["ticker", "company"])
            .drop_duplicates()
            .groupby("ticker")["company"].apply(list).to_dict()
        )
        uni = pd.read_csv(UNIVERSE)
        uni_names = uni.set_index("ticker")["company"].to_dict()
        self._identity: Dict[str, List[Pattern[str]]] = {}
        for tk in set(panel_names) | set(uni_names):
            names = list(dict.fromkeys(
                panel_names.get(tk, [])
                + ([uni_names[tk]] if tk in uni_names else [])
                + EXTRA_ALIASES.get(tk, [])
            ))
            self._identity[tk] = build_identity_patterns(tk, names)
        # leak guard state + wrapper around the LLM client
        self._guard_key: tuple[str, str] | None = None
        self._dump_path: Path | None = None
        self._scan_only = False
        self._n_scanned = 0
        real_query = self._llm_client.query_messages

        def guarded(messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
            tk, dt = self._guard_key if self._guard_key else ("?", "?")
            pats = self._identity.get(tk, [])
            all_hits: List[str] = []
            for i, msg in enumerate(messages):
                all_hits += [f"msg{i}:{h}" for h in scan_hits(msg["content"], pats)]
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": "scan_only" if self._scan_only else "live",
                "ticker": tk, "date": dt,
                "n_messages": len(messages),
                "chars": sum(len(m["content"]) for m in messages),
                "leak_hits": len(all_hits),
                "hits": all_hits[:10],
            }
            with SCAN_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._n_scanned += len(messages)
            if self._dump_path is not None:
                with self._dump_path.open("a", encoding="utf-8") as f:
                    f.write(f"\n{'=' * 78}\n[{tk} {dt}] leak_hits={len(all_hits)}\n")
                    for msg in messages:
                        f.write(f"--- {msg['role']} ---\n{msg['content']}\n")
            if all_hits:
                raise LeakError(f"{tk} {dt}: {len(all_hits)} identity hits "
                                f"in rendered prompt: {all_hits[:3]}")
            if self._scan_only:
                return {"content": _FAKE_CONTENT, "model": "scan-only",
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0},
                        "cached": False, "latency_sec": 0.0,
                        "has_reasoning_content": False}
            return real_query(messages, **kw)

        self._llm_client.query_messages = guarded  # type: ignore[method-assign]

    # -- builder overrides: super() builds with real identity (needed for
    #    EDGAR / fundamentals lookups), then identity is masked. -------------

    def _mask_common(self, d: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(d)
        d["ticker"] = MASK_TICKER
        d["company"] = MASK_COMPANY
        return d

    def _build_disclosure_inputs(self, base: Dict[str, Any],
                                 decision_date: str) -> Dict[str, Any]:
        d = super()._build_disclosure_inputs(base, decision_date)
        pats = self._identity[d["ticker"]]
        d = self._mask_common(d)
        d["filing_10k_text"] = scrub(d["filing_10k_text"], pats)
        d["filing_10q_text"] = scrub(d["filing_10q_text"], pats)
        d["filing_8k_texts"] = [scrub(t, pats) for t in d["filing_8k_texts"]]
        return d

    def _build_macro_inputs(self, base: Dict[str, Any]) -> Dict[str, Any]:
        return self._mask_common(super()._build_macro_inputs(base))

    def _build_price_inputs(self, base: Dict[str, Any]) -> Dict[str, Any]:
        return self._mask_common(super()._build_price_inputs(base))

    def _build_fundamentals_inputs(self, base: Dict[str, Any]) -> Dict[str, Any]:
        return self._mask_common(super()._build_fundamentals_inputs(base))


# ── main ──────────────────────────────────────────────────────────────────


def subset_keys() -> list[tuple[str, str, str]]:
    truth = pd.read_csv(TEST)
    truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
    red = truth[truth["label"] == "reduce"].sample(N_RED, random_state=SEED)
    oth = truth[truth["label"] != "reduce"].sample(N_OTH, random_state=SEED)
    sample = pd.concat([red, oth])
    return list(zip(sample["ticker"], sample["date"], sample["label"]))


def ledger_cost() -> float:
    """Actual probe spend: dedupe ledger by (ticker,date) — reruns of cached
    decisions re-log the same estimated cost but hit diskcache (zero spend)."""
    ledger = AUDIT_DIR / "cost_ledger.jsonl"
    if not ledger.exists():
        return 0.0
    seen: Dict[tuple[str, str], float] = {}
    with ledger.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                seen[(d["ticker"], d["decision_date_t"])] = d["cost_usd"]
    return round(sum(seen.values()), 4)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    assert mode in {"scan", "dry", "full"}, f"unknown mode {mode!r}"
    keys = subset_keys()
    if mode == "dry":
        keys = [keys[0], keys[1], keys[N_RED]]  # 2 reduce + 1 non-reduce

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if mode == "scan":
        # fake decision records must never touch audit_log/masked_probe/
        run_audit_dir = Path(tempfile.mkdtemp(prefix="masked_scan_audit_"))
    else:
        run_audit_dir = AUDIT_DIR
    orch = MaskedOrchestrator(audit_log_dir=run_audit_dir,
                              model_config_name=MODEL, panel_path=PANEL)
    orch._dump_path = {"dry": DRYRUN_PROMPTS, "scan": SCAN_PROMPTS}.get(mode)
    orch._scan_only = (mode == "scan")

    masked: Dict[tuple[str, str], dict] = {}
    fails: list[str] = []
    for tk, dt, lab in keys:
        if mode != "scan" and ledger_cost() > BUDGET_USD:
            print(f"BUDGET CAP {BUDGET_USD} USD reached — stopping.")
            break
        orch._guard_key = (tk, dt)
        try:
            rec = orch.run_single(tk, dt)
            masked[(tk, dt)] = json.loads(rec.model_dump_json())
        except Exception as e:  # noqa: BLE001
            msg = f"{tk} {dt}: {type(e).__name__}: {str(e)[:100]}"
            fails.append(msg)
            print(f"  fail {msg}")

    print(f"\nmasking guard: {orch._n_scanned} rendered messages scanned "
          f"across {len(masked)} ok / {len(fails)} failed decisions; any hit "
          f"raises LeakError before the API call; log: {SCAN_LOG}")

    if mode == "scan":
        print(f"scan-only done: {len(masked)}/{len(keys)} decisions rendered "
              f"with 0 API calls (fake responses; temp audit dir "
              f"{run_audit_dir}).")
        if fails:
            print("LEAKS / FAILURES to fix before live run:")
            for m in fails:
                print(f"  {m}")
        return

    if mode == "dry":
        print(f"dry run done: {len(masked)}/{len(keys)} decisions, "
              f"cost so far = {ledger_cost()} USD")
        print(f"rendered prompts dumped to {DRYRUN_PROMPTS}")
        return

    # ── compare vs unmasked flash v2 on the same keys ─────────────────────
    flash: Dict[tuple[str, str], dict] = {}
    with FLASH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            flash[(d["ticker"], d["decision_date_t"])] = d

    rows = []
    for tk, dt, lab in keys:
        r: Dict[str, Any] = {"ticker": tk, "date": dt, "true": lab}
        if (tk, dt) in masked:
            r["masked_pred"] = amax(masked[(tk, dt)]["final_probabilities"])
            r["masked_grounded"] = grounded(masked[(tk, dt)])
        if (tk, dt) in flash:
            r["flash_pred"] = amax(flash[(tk, dt)]["final_probabilities"])
            r["flash_grounded"] = grounded(flash[(tk, dt)])
        rows.append(r)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    def rr(col: str) -> str:
        sub = df[df["true"] == "reduce"].dropna(subset=[col])
        n = int((sub[col] == "reduce").sum())
        return f"{n}/{len(sub)} = {n / len(sub):.3f}" if len(sub) else "nan"

    redmask = df["true"] == "reduce"
    both = df.dropna(subset=["masked_pred", "flash_pred"])
    agree = (both["masked_pred"] == both["flash_pred"]).mean()
    print(f"\n=== MASKED-IDENTITY PROBE ({MODEL}, n={len(df)}, "
          f"reduce={int(redmask.sum())}, fails={len(fails)}) ===")
    print(f"  reduce recall   masked={rr('masked_pred')}   unmasked={rr('flash_pred')}")
    print(f"  label agreement masked vs unmasked (n={len(both)}): {agree:.3f}")
    print(f"  grounding (all)    masked={df['masked_grounded'].mean():.2f}   "
          f"unmasked={df['flash_grounded'].mean():.2f}")
    print(f"  grounding (reduce) masked={df[redmask]['masked_grounded'].mean():.2f}   "
          f"unmasked={df[redmask]['flash_grounded'].mean():.2f}")
    for lbl in ["increase", "hold", "reduce"]:
        m = (both["masked_pred"] == lbl).sum()
        u = (both["flash_pred"] == lbl).sum()
        print(f"  pred count {lbl:<8} masked={m:>3}  unmasked={u:>3}")
    print(f"  total probe cost: {ledger_cost()} USD "
          f"(audit_log/masked_probe/cost_ledger.jsonl, deduped by decision)")
    print(f"  per-decision comparison -> {OUT}")
    print("\n  VERDICT: identical behaviour under masking = no evidence the model")
    print("  relies on firm identity (contamination channel closed); divergence")
    print("  would indicate memorised firm-specific knowledge.")


if __name__ == "__main__":
    main()
