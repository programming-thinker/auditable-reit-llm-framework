"""v3 corrected-feed robustness rerun of the LOCKED test window.

Protocol note (disclosed in thesis Section 5.3): prompts are UNCHANGED from the
v2 production run (same files, same SHAs, PROMPT_LOCK_TIMESTAMP honoured); the
only difference is the input pipeline, which now carries the two fidelity fixes
found by the 2026-07-05 replay audit (Item 1A anchor hardened; 8-K window keeps
the NEWEST five filings). v2 remains the headline run; v3 is a robustness run
answering "would a corrected feed have added signal?".

Isolation guarantees (CLAUDE.md tripwires):
  * audit_log/decisions.jsonl, predictions.csv, cost_ledger.jsonl are NOT touched;
    everything is written under audit_log/v3_corrected_feed/ (new, append-only).
  * The three non-disclosure specialists' prompts are byte-identical to v2, so
    their calls replay from the response cache at zero cost; only Disclosure and
    Aggregator hit the API (~2 x 575 calls).

Usage:
    python -m llm.run_v3_corrected_feed --smoke   # 1 REIT x 2 test months
    python -m llm.run_v3_corrected_feed           # full 25 x 2024-2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import structlog
import yaml

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
PANEL_PATH = ROOT / "data" / "processed" / "backtest_ready_panel_enriched.csv"
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"
V3_DIR = ROOT / "audit_log" / "v3_corrected_feed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="1 REIT x 2 test months only")
    args = parser.parse_args()

    with CONFIG_PATH.open("r") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("PROMPT_LOCK_TIMESTAMP") is None:
        raise RuntimeError("PROMPT_LOCK_TIMESTAMP not set; v3 run refused.")

    tickers = sorted(pd.read_csv(UNIVERSE_PATH)["ticker"].tolist())
    date_range = ("2024-01-01", "2025-11-30")
    if args.smoke:
        tickers = tickers[:1]
        date_range = ("2024-01-01", "2024-02-29")

    logger.info("v3_corrected_feed_run", n_tickers=len(tickers),
                date_range=date_range, out_dir=str(V3_DIR))

    from llm.orchestrator import Orchestrator
    orch = Orchestrator(
        config_path=CONFIG_PATH,
        model_config_name="deepseek_v4_flash",
        panel_path=PANEL_PATH,
        audit_log_dir=V3_DIR,
    )
    records = orch.run_batch(tickers=tickers, date_range=date_range)
    logger.info("v3_run_complete", n_records=len(records))

    from llm.postprocess import jsonl_to_predictions_csv
    jsonl_to_predictions_csv(
        decisions_jsonl=V3_DIR / "decisions.jsonl",
        output_csv=V3_DIR / "predictions_v3.csv",
    )
    print(f"[done] {len(records)} decisions -> {V3_DIR}/predictions_v3.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
