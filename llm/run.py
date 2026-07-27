"""CLI entry point for the multi-agent LLM pipeline.

Usage:
    python -m llm.run --mode dev          # 1 REIT x 2 months (smoke test)
    python -m llm.run --mode validation   # 25 REITs x 2022-2023
    python -m llm.run --mode test         # locked test-window run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import structlog
import yaml

logger = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
PANEL_PATH = ROOT / "data" / "processed" / "backtest_ready_panel_enriched.csv"
UNIVERSE_PATH = ROOT / "config" / "reit_universe.csv"


def _load_tickers() -> List[str]:
    """Load the 25-REIT universe from config."""
    df = pd.read_csv(UNIVERSE_PATH)
    return sorted(df["ticker"].tolist())


def _check_prompt_lock() -> str:
    """Check PROMPT_LOCK_TIMESTAMP; raise if not set."""
    with CONFIG_PATH.open("r") as f:
        cfg = yaml.safe_load(f)
    ts = cfg.get("PROMPT_LOCK_TIMESTAMP")
    if ts is None:
        raise RuntimeError(
            "PROMPT_LOCK_TIMESTAMP not set in config/config.yaml. "
            "Test run refused."
        )
    return str(ts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-agent LLM pipeline.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dev", "validation", "validation_mini", "test"],
        help="dev=1 REIT x 2 months, validation_mini=5 REITs x 2022-2023, validation=25 REITs x 2022-2023, test=locked",
    )
    parser.add_argument(
        "--model-config",
        default="qwen3_8b_non_thinking",
        help="Model config key in llm_model_configs.yaml",
    )
    args = parser.parse_args()

    # lazy import to avoid loading heavy deps at parse time
    from llm.orchestrator import Orchestrator

    if args.mode == "dev":
        tickers = _load_tickers()[:1]  # first ticker only
        date_range = ("2022-01-01", "2022-02-28")  # 2 months
        logger.info(
            "dev_run", tickers=tickers, date_range=date_range
        )

    elif args.mode == "validation_mini":
        tickers = _load_tickers()[:5]  # first 5 REITs only
        date_range = ("2022-01-01", "2023-12-31")
        logger.info(
            "validation_mini_run",
            n_tickers=len(tickers),
            date_range=date_range,
        )

    elif args.mode == "validation":
        tickers = _load_tickers()
        date_range = ("2022-01-01", "2023-12-31")
        logger.info(
            "validation_run",
            n_tickers=len(tickers),
            date_range=date_range,
        )

    elif args.mode == "test":
        ts = _check_prompt_lock()
        tickers = _load_tickers()
        date_range = ("2024-01-01", "2025-11-30")
        logger.info(
            "test_run",
            prompt_lock=ts,
            n_tickers=len(tickers),
            date_range=date_range,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    orch = Orchestrator(
        config_path=CONFIG_PATH,
        model_config_name=args.model_config,
        panel_path=PANEL_PATH,
    )

    records = orch.run_batch(tickers=tickers, date_range=date_range)

    logger.info("run_complete", mode=args.mode, n_records=len(records))

    # generate predictions.csv
    from llm.postprocess import jsonl_to_predictions_csv

    jsonl_to_predictions_csv()

    return 0


if __name__ == "__main__":
    sys.exit(main())
