"""SEC EDGAR retrieval client with filing-date enforcement.

Filing-date enforcement is critical: we must never use a filing whose
filing_date is after the decision_date_t, to prevent look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class EdgarClient:
    """Retrieve and cache SEC EDGAR filings with strict date enforcement.

    Parameters
    ----------
    filings_dir : Path
        Root directory containing clean_text/ and raw_html/ subdirectories.
    metadata_path : Path
        Path to data/interim/filing_metadata.csv.
    """

    def __init__(
        self,
        filings_dir: Path = Path("filings"),
        metadata_path: Path = Path("data/interim/filing_metadata.csv"),
    ) -> None:
        self._filings_dir = filings_dir
        self._clean_text_dir = filings_dir / "clean_text"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Filing metadata not found: {metadata_path}")
        if not self._clean_text_dir.is_dir():
            raise FileNotFoundError(
                f"clean_text directory not found: {self._clean_text_dir}"
            )

        df = pd.read_csv(metadata_path, dtype=str)
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        df = df.sort_values("filing_date")
        self._metadata = df

        logger.info(
            "edgar_client_init",
            n_filings=len(df),
            tickers=sorted(df["ticker"].unique().tolist()),
            date_range=(
                str(df["filing_date"].min().date()),
                str(df["filing_date"].max().date()),
            ),
        )

    # ── public API ────────────────────────────────────────────────────

    def get_latest_filing(
        self,
        ticker: str,
        filing_type: str,
        as_of_date: str,
    ) -> str | None:
        """Return the text of the most recent filing of the given type
        whose filing_date <= as_of_date.

        Parameters
        ----------
        ticker : str
            REIT ticker symbol (e.g. "EQIX").
        filing_type : str
            SEC filing type (e.g. "10-K", "10-Q", "8-K").
        as_of_date : str
            Decision date in YYYY-MM-DD format. Only filings on or before
            this date are eligible.

        Returns
        -------
        str | None
            Filing text, or None if no eligible filing found.
        """
        cutoff = pd.Timestamp(as_of_date)
        mask = (
            (self._metadata["ticker"] == ticker)
            & (self._metadata["form"] == filing_type)
            & (self._metadata["filing_date"] <= cutoff)
        )
        eligible = self._metadata.loc[mask]

        if eligible.empty:
            logger.debug(
                "no_eligible_filing",
                ticker=ticker,
                filing_type=filing_type,
                as_of_date=as_of_date,
            )
            return None

        latest = eligible.iloc[-1]  # already sorted by filing_date
        return self._read_clean_text(latest)

    def get_filings_in_window(
        self,
        ticker: str,
        filing_type: str,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, str]]:
        """Return all filings of the given type within a date window.

        Useful for gathering multiple 8-K filings in a period.

        Parameters
        ----------
        ticker : str
            REIT ticker symbol.
        filing_type : str
            SEC filing type (e.g. "8-K").
        start_date : str
            Window start (inclusive) in YYYY-MM-DD.
        end_date : str
            Window end (inclusive) in YYYY-MM-DD.  Must not exceed
            the decision date to avoid look-ahead bias.

        Returns
        -------
        list[dict]
            Each dict has keys: filing_date, accession_nodash, text.
            Ordered by filing_date ascending.
        """
        t_start = pd.Timestamp(start_date)
        t_end = pd.Timestamp(end_date)
        mask = (
            (self._metadata["ticker"] == ticker)
            & (self._metadata["form"] == filing_type)
            & (self._metadata["filing_date"] >= t_start)
            & (self._metadata["filing_date"] <= t_end)
        )
        eligible = self._metadata.loc[mask]

        results: list[dict[str, str]] = []
        for _, row in eligible.iterrows():
            text = self._read_clean_text(row)
            if text is not None:
                results.append(
                    {
                        "filing_date": str(row["filing_date"].date()),
                        "accession_nodash": row["accession_nodash"],
                        "text": text,
                    }
                )

        logger.debug(
            "filings_in_window",
            ticker=ticker,
            filing_type=filing_type,
            window=(start_date, end_date),
            n_found=len(results),
        )
        return results

    def get_latest_annual_and_quarterly(
        self,
        ticker: str,
        as_of_date: str,
    ) -> dict[str, str | None]:
        """Convenience method: get latest 10-K and 10-Q texts as of a date.

        Returns
        -------
        dict with keys "10-K" and "10-Q", each mapping to filing text or None.
        """
        return {
            "10-K": self.get_latest_filing(ticker, "10-K", as_of_date),
            "10-Q": self.get_latest_filing(ticker, "10-Q", as_of_date),
        }

    # ── internals ─────────────────────────────────────────────────────

    def _read_clean_text(self, row: pd.Series) -> str | None:
        """Read a clean_text file for a metadata row.

        File naming convention:
            {ticker}_{form}_{filing_date}_{accession_nodash}.txt
        """
        filing_date_str = str(row["filing_date"].date()) if isinstance(
            row["filing_date"], pd.Timestamp
        ) else row["filing_date"]
        filename = (
            f"{row['ticker']}_{row['form']}_{filing_date_str}"
            f"_{row['accession_nodash']}.txt"
        )
        path = self._clean_text_dir / filename

        if not path.exists():
            logger.warning("clean_text_missing", path=str(path))
            return None

        return path.read_text(encoding="utf-8")
