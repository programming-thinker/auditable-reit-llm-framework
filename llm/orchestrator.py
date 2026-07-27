"""Decision loop: runs all four agents for each (ticker, decision_date) pair
and writes results to audit_log/.

Workflow per CLAUDE.md Section 6 "Writing a new agent":
  agents are pure functions (inputs: dict) -> AgentOutput
  orchestrator wires them together after tests pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import structlog
import yaml

from llm.agents.aggregator import run_aggregator_agent
from llm.agents.base import load_prompt_template
from llm.agents.disclosure import run_disclosure_agent
from llm.agents.fundamentals import FUND_FEATURES, run_fundamentals_agent
from llm.agents.macro import run_macro_agent
from llm.agents.price import run_price_agent
from llm.edgar_client import EdgarClient
from llm.llm_client import LLMClient
from llm.schemas import (
    AgentOutputs,
    DecisionRecord,
    Probabilities,
    PromptsSHA,
    TokenUsage,
)

logger = structlog.get_logger(__name__)


class Orchestrator:
    """Run the multi-agent decision pipeline.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml (project-level config).
    audit_log_dir : Path
        Path to audit_log/ directory.
    model_config_name : str
        Key in config/llm_model_configs.yaml.
    panel_path : Path
        Path to the backtest-ready panel CSV.
    """

    def __init__(
        self,
        config_path: Path = Path("config/config.yaml"),
        audit_log_dir: Path = Path("audit_log"),
        model_config_name: str = "qwen3_8b_non_thinking",
        panel_path: Path = Path("data/processed/backtest_ready_panel.csv"),
    ) -> None:
        # ── project config ────────────────────────────────────────────
        with config_path.open("r", encoding="utf-8") as f:
            self._project_config: Dict[str, Any] = yaml.safe_load(f)

        # ── audit log paths ───────────────────────────────────────────
        self._audit_log_dir = audit_log_dir
        audit_log_dir.mkdir(parents=True, exist_ok=True)
        self._decisions_path = audit_log_dir / "decisions.jsonl"
        self._cost_ledger_path = audit_log_dir / "cost_ledger.jsonl"

        # ── panel data ────────────────────────────────────────────────
        self._panel = pd.read_csv(panel_path, parse_dates=["date"])
        logger.info(
            "panel_loaded",
            rows=len(self._panel),
            tickers=sorted(self._panel["ticker"].unique().tolist()),
        )

        # ── clients ───────────────────────────────────────────────────
        self._model_config_name = model_config_name
        self._llm_client = LLMClient(model_config_name=model_config_name)
        self._edgar_client = EdgarClient()

        # ── firm fundamentals panel (for Fundamentals agent) ──────────
        self._fundamentals = self._load_fundamentals()

        # ── prompt SHAs (computed once) ───────────────────────────────
        self._prompts_sha = self._compute_prompts_sha()

        logger.info("orchestrator_init", model_config=model_config_name)

    # ── public API ────────────────────────────────────────────────────

    def run_single(
        self,
        ticker: str,
        decision_date: str,
    ) -> DecisionRecord:
        """Run all four agents for one (ticker, date) and return the record.

        The record is also appended to audit_log/decisions.jsonl.
        """
        t0 = time.monotonic()

        # ── build inputs ──────────────────────────────────────────────
        base_inputs = self._build_base_inputs(ticker, decision_date)
        disclosure_inputs = self._build_disclosure_inputs(base_inputs, decision_date)
        macro_inputs = self._build_macro_inputs(base_inputs)
        price_inputs = self._build_price_inputs(base_inputs)
        fundamentals_inputs = self._build_fundamentals_inputs(base_inputs)

        inputs_hash = _compute_inputs_hash(
            {
                "disclosure": disclosure_inputs,
                "macro": macro_inputs,
                "price": price_inputs,
                "fundamentals": fundamentals_inputs,
            }
        )

        # ── run agents sequentially ──────────────────────────────────
        logger.info("run_single_start", ticker=ticker, date=decision_date)

        disclosure_out, disclosure_meta = run_disclosure_agent(
            disclosure_inputs, self._llm_client
        )
        macro_out, macro_meta = run_macro_agent(macro_inputs, self._llm_client)
        price_out, price_meta = run_price_agent(price_inputs, self._llm_client)
        fundamentals_out, fundamentals_meta = run_fundamentals_agent(
            fundamentals_inputs, self._llm_client
        )

        aggregator_inputs = {
            "ticker": ticker,
            "company": base_inputs["company"],
            "sector": base_inputs["sector"],
            "formation_date": decision_date,
        }
        aggregator_out, aggregator_meta = run_aggregator_agent(
            aggregator_inputs, disclosure_out, macro_out, price_out, self._llm_client,
            fundamentals_output=fundamentals_out, prompt_version=2,
        )

        latency = time.monotonic() - t0

        # ── aggregate token usage ─────────────────────────────────────
        all_metas = [disclosure_meta, macro_meta, price_meta, fundamentals_meta, aggregator_meta]
        total_prompt_tokens = sum(
            m.get("usage", {}).get("prompt_tokens", 0) for m in all_metas
        )
        total_completion_tokens = sum(
            m.get("usage", {}).get("completion_tokens", 0) for m in all_metas
        )
        total_cost = _estimate_cost(total_prompt_tokens, total_completion_tokens)

        # ── assemble record ───────────────────────────────────────────
        record = DecisionRecord(
            decision_date_t=decision_date,
            ticker=ticker,
            model_main=f"{self._model_config_name}",
            prompts_sha=self._prompts_sha,
            inputs_hash=inputs_hash,
            agent_outputs=AgentOutputs(
                disclosure=disclosure_out,
                macro=macro_out,
                price=price_out,
                fundamentals=fundamentals_out,
                aggregator=aggregator_out,
            ),
            final_probabilities=aggregator_out.probabilities,
            tokens=TokenUsage(
                prompt=total_prompt_tokens,
                completion=total_completion_tokens,
                cost_usd=total_cost,
            ),
            latency_sec=round(latency, 3),
        )

        # ── append to audit log + cost ledger ─────────────────────────
        self._append_decision(record)
        self._append_cost_ledger(record)

        logger.info(
            "run_single_done",
            ticker=ticker,
            date=decision_date,
            latency_sec=record.latency_sec,
            final_signal=_argmax_label(aggregator_out.probabilities),
        )
        return record

    def run_batch(
        self,
        tickers: List[str],
        date_range: Tuple[str, str],
    ) -> List[DecisionRecord]:
        """Run the pipeline for all tickers across a date range.

        Parameters
        ----------
        tickers : list[str]
            REIT ticker symbols.
        date_range : tuple[str, str]
            (start_date, end_date) in YYYY-MM-DD format.
        """
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])

        # get unique month-end dates from panel within range
        panel_dates = self._panel["date"].drop_duplicates().sort_values()
        eligible_dates = panel_dates[(panel_dates >= start) & (panel_dates <= end)]

        total = len(tickers) * len(eligible_dates)
        logger.info(
            "run_batch_start",
            n_tickers=len(tickers),
            n_dates=len(eligible_dates),
            total=total,
        )

        records: List[DecisionRecord] = []
        completed = 0

        for date in eligible_dates:
            date_str = str(date.date())
            for ticker in tickers:
                try:
                    record = self.run_single(ticker, date_str)
                    records.append(record)
                except Exception as exc:
                    logger.error(
                        "run_single_failed",
                        ticker=ticker,
                        date=date_str,
                        error=str(exc),
                    )
                completed += 1
                if completed % 10 == 0 or completed == total:
                    logger.info(
                        "batch_progress",
                        completed=completed,
                        total=total,
                    )

        logger.info("run_batch_done", n_records=len(records), total=total)
        return records

    # ── input builders ────────────────────────────────────────────────

    def _build_base_inputs(
        self, ticker: str, decision_date: str
    ) -> Dict[str, Any]:
        """Extract the panel row for (ticker, date) into a dict."""
        dt = pd.Timestamp(decision_date)
        mask = (self._panel["ticker"] == ticker) & (self._panel["date"] == dt)
        rows = self._panel.loc[mask]

        if rows.empty:
            raise ValueError(
                f"No panel row for ticker={ticker}, date={decision_date}"
            )

        row = rows.iloc[0]
        return row.to_dict()

    def _build_disclosure_inputs(
        self, base: Dict[str, Any], decision_date: str
    ) -> Dict[str, Any]:
        """Build Disclosure Agent inputs from panel row + EDGAR filings."""
        ticker = base["ticker"]

        # get filing texts via EdgarClient (filing-date enforced)
        filings = self._edgar_client.get_latest_annual_and_quarterly(
            ticker, decision_date
        )

        # get recent 8-K filings from prior 6 months
        six_months_ago = str(
            (pd.Timestamp(decision_date) - pd.DateOffset(months=6)).date()
        )
        eight_k_filings = self._edgar_client.get_filings_in_window(
            ticker, "8-K", six_months_ago, decision_date
        )

        return {
            "ticker": ticker,
            "company": base["company"],
            "sector": base["sector"],
            "formation_date": decision_date,
            "filing_10k_text": _truncate(_extract_item1a(filings.get("10-K")), max_chars=15000),
            "filing_10q_text": _truncate(_extract_item1a(filings.get("10-Q")), max_chars=10000),
            "filing_8k_texts": [
                _truncate(f["text"], max_chars=3000)
                # EdgarClient returns filings ASCENDING by filing_date, so the
                # 5 most recent are the LAST five ([-5:]), kept in ascending
                # order. ([:5] would silently keep the 5 OLDEST -- bug fixed
                # 2026-07-04; see analysis/audit_8k_recency.py for exposure.)
                for f in eight_k_filings[-5:]
            ],
        }

    def _build_macro_inputs(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Build Macro Agent inputs from panel row."""
        return {
            "ticker": base["ticker"],
            "company": base["company"],
            "sector": base["sector"],
            "formation_date": str(base["date"].date()) if isinstance(base["date"], pd.Timestamp) else str(base["date"]),
            "FEDFUNDS_lag1": _safe_float(base.get("FEDFUNDS_lag1")),
            "DGS10_lag1": _safe_float(base.get("DGS10_lag1")),
            "DGS2_lag1": _safe_float(base.get("DGS2_lag1")),
            "term_spread_10y_2y_lag1": _safe_float(base.get("term_spread_10y_2y_lag1")),
            "cpi_yoy_lag1": _safe_float(base.get("cpi_yoy_lag1")),
            "UNRATE_lag1": _safe_float(base.get("UNRATE_lag1")),
        }

    def _build_price_inputs(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Build Price Agent inputs from panel row."""
        return {
            "ticker": base["ticker"],
            "company": base["company"],
            "sector": base["sector"],
            "formation_date": str(base["date"].date()) if isinstance(base["date"], pd.Timestamp) else str(base["date"]),
            "ret_1m": _safe_float(base.get("ret_1m")),
            "ret_3m": _safe_float(base.get("ret_3m")),
            "ret_6m": _safe_float(base.get("ret_6m")),
            "ret_12m": _safe_float(base.get("ret_12m")),
            "vol_annualized": _safe_float(base.get("vol_annualized")),
            "drawdown": _safe_float(base.get("drawdown")),
        }

    @staticmethod
    def _load_fundamentals() -> pd.DataFrame:
        """Load the pre-built firm fundamentals panel (point-in-time) keyed by
        (ticker, date). Merges the SEC/price fundamentals with the NAV proxy."""
        base_dir = Path("outputs/fundamentals_robustness")
        fund = pd.read_csv(base_dir / "firm_fundamentals_panel.csv")
        nav = pd.read_csv(base_dir / "nav_proxy_panel.csv")[["ticker", "date", "navprem_book_adj"]]
        merged = fund.merge(nav, on=["ticker", "date"], how="left")
        merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
        return merged.set_index(["ticker", "date"])

    def _build_fundamentals_inputs(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Build Fundamentals Agent inputs from the pre-built fundamentals panel."""
        ticker = base["ticker"]
        date_str = str(base["date"].date()) if isinstance(base["date"], pd.Timestamp) else str(base["date"])
        out = {
            "ticker": ticker,
            "company": base["company"],
            "sector": base["sector"],
            "formation_date": date_str,
        }
        try:
            row = self._fundamentals.loc[(ticker, date_str)]
        except KeyError:
            row = None
        for f in FUND_FEATURES:
            out[f] = _safe_float(row[f]) if row is not None and f in row else "not_available"
        return out

    # ── audit log ─────────────────────────────────────────────────────

    def _append_decision(self, record: DecisionRecord) -> None:
        """Append a DecisionRecord to decisions.jsonl (append-only)."""
        with self._decisions_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
            f.flush()

    def _append_cost_ledger(self, record: DecisionRecord) -> None:
        """Append a cost entry to cost_ledger.jsonl (append-only)."""
        entry = {
            "decision_date_t": record.decision_date_t,
            "ticker": record.ticker,
            "prompt_tokens": record.tokens.prompt,
            "completion_tokens": record.tokens.completion,
            "cost_usd": record.tokens.cost_usd,
            "model": record.model_main,
        }
        with self._cost_ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

    # ── prompt SHA ────────────────────────────────────────────────────

    @staticmethod
    def _compute_prompts_sha(prompt_version: int = 1) -> PromptsSHA:
        """Compute SHA256 hashes for each agent's prompt template.

        v2 framework: disclosure/macro/price/fundamentals are v1; aggregator is v2.
        """
        versions = {"disclosure": 1, "macro": 1, "price": 1,
                    "fundamentals": 1, "aggregator": 2}
        shas = {}
        for agent_name, ver in versions.items():
            template = load_prompt_template(agent_name, ver)
            sha = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
            shas[agent_name] = sha
        return PromptsSHA(**shas)


# ── Module-level helpers ──────────────────────────────────────────────────


# DeepSeek V3 pricing (per 1M tokens) — update if model changes
_COST_PER_1M_PROMPT = 0.27  # USD
_COST_PER_1M_COMPLETION = 1.10  # USD


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate API cost in USD based on token counts."""
    cost = (
        prompt_tokens * _COST_PER_1M_PROMPT / 1_000_000
        + completion_tokens * _COST_PER_1M_COMPLETION / 1_000_000
    )
    return round(cost, 6)


def _compute_inputs_hash(inputs: Dict[str, Any]) -> str:
    """SHA256 hash of the serialised inputs dict."""
    payload = json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


# Genuine section header: 'item 1a' followed within ~40 chars by 'risk factors'.
# Whitespace-tolerant inside the words to survive text-cleaning artifacts seen in
# filings/clean_text (e.g. 'Item 1A. Ris k Factors' (MAA), 'Ri sk Factors' (PLD),
# 'R isk Factors' (REG), 'Item 1 A. Risk Factors' (KIM)).
_ITEM1A_HEADER_RE = re.compile(
    r"item\s*1\s*a\b.{0,40}?r\s*i\s*s\s*k\s*f\s*a\s*c\s*t\s*o\s*r\s*s",
    re.I | re.S,
)
_ITEM1B_RE = re.compile(r"item\s*1\s*b", re.I)
# A genuine Item 1A section runs at least this many chars before Item 1B; a
# header-like match closer than this to Item 1B is a trailing cross-reference.
_MIN_ITEM1A_SECTION_CHARS = 5000


def _extract_item1a(raw: Optional[str]) -> Optional[str]:
    """Extract the Item 1A 'Risk Factors' prose from a raw filing.

    Anchor selection (fixed 2026-07-04): the FIRST bare 'item 1a' hit is very
    often a cross-reference ("as set forth in Item 1A ... 'Risk Factors'")
    inside the Item 1 business description, so anchoring there feeds the
    Disclosure agent mostly business text (see analysis/audit_item1a_anchor.py
    for the as-run exposure). Instead:

      1. Collect case-insensitive matches of 'item 1a' followed within ~40
         chars by 'risk factors' (the section-header form).
      2. Keep those followed later by an 'item 1b' match (table-of-contents
         entries and late cross-references in MD&A have no Item 1B after them).
      3. Among those, prefer matches at least _MIN_ITEM1A_SECTION_CHARS before
         the next 'item 1b' (a nearer match is a cross-reference sitting just
         above Item 1B), and take the LAST one -- earlier qualifying matches
         are cross-references in Item 1 that precede the real section.
      4. If no header-form match qualifies, fall back to the original logic
         (first 'item 1a' hit; then 'risk factors'; then the raw text).

    The character budget is unchanged: truncation to 15k/10k chars still
    happens at the call site via _truncate().
    """
    if not raw:
        return raw
    ends = [m.start() for m in _ITEM1B_RE.finditer(raw)]

    headers = [m.start() for m in _ITEM1A_HEADER_RE.finditer(raw)]
    candidates = [s for s in headers if any(e > s for e in ends)]
    if candidates:
        def _section_len(start: int) -> int:
            return next(e for e in ends if e > start) - start

        long_enough = [
            s for s in candidates if _section_len(s) >= _MIN_ITEM1A_SECTION_CHARS
        ]
        s = (long_enough or candidates)[-1]
        return raw[s : s + _section_len(s)]

    # fallback: original (pre-fix) logic
    starts = [m.start() for m in re.finditer(r"item\s*1a", raw, re.I)]
    if not starts:
        m = re.search(r"risk factors", raw, re.I)
        return raw[m.start():] if m else raw
    s = starts[0]
    e = next((x for x in ends if x > s), len(raw))
    return raw[s:e]


def _truncate(text: Optional[str], max_chars: int) -> Optional[str]:
    """Truncate text to max_chars, preserving None."""
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... truncated at {max_chars} characters]"


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None for NaN / missing."""
    if value is None:
        return None
    try:
        f = float(value)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None


def _argmax_label(probs: Probabilities) -> str:
    """Return the label with highest probability."""
    mapping = {
        "increase": probs.increase,
        "hold": probs.hold,
        "reduce": probs.reduce,
    }
    return max(mapping, key=mapping.get)  # type: ignore[arg-type]
