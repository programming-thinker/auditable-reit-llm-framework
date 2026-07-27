"""Tests for EdgarClient: filing-date enforcement, missing files, ordering.

All tests use a synthetic filings directory + metadata CSV in tmp_path;
no network and no real filings/ data are touched.
"""

from pathlib import Path

import pandas as pd

from llm.edgar_client import EdgarClient


def _filename(ticker: str, form: str, filing_date: str, accession: str) -> str:
    return f"{ticker}_{form}_{filing_date}_{accession}.txt"


def _make_client(
    tmp_path: Path,
    rows: list,
    texts: dict,
) -> EdgarClient:
    """Build an EdgarClient over a synthetic filings dir + metadata CSV.

    rows: list of dicts with keys ticker, form, filing_date, accession_nodash.
    texts: {filename: file content} written into clean_text/.
    """
    filings_dir = tmp_path / "filings"
    clean_text_dir = filings_dir / "clean_text"
    clean_text_dir.mkdir(parents=True)

    metadata_path = tmp_path / "filing_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)

    for filename, content in texts.items():
        (clean_text_dir / filename).write_text(content, encoding="utf-8")

    return EdgarClient(filings_dir=filings_dir, metadata_path=metadata_path)


def test_filing_dated_after_cutoff_excluded(tmp_path: Path) -> None:
    """A filing dated after as_of_date must never be returned (no look-ahead)."""
    rows = [
        {
            "ticker": "EQIX",
            "form": "10-K",
            "filing_date": "2023-02-01",
            "accession_nodash": "acc2023",
        },
        {
            "ticker": "EQIX",
            "form": "10-K",
            "filing_date": "2024-02-15",
            "accession_nodash": "acc2024",
        },
    ]
    texts = {
        _filename("EQIX", "10-K", "2023-02-01", "acc2023"): "OLD 10-K TEXT",
        _filename("EQIX", "10-K", "2024-02-15", "acc2024"): "NEW 10-K TEXT",
    }
    client = _make_client(tmp_path, rows, texts)

    # decision date between the two filings: only the 2023 filing is eligible
    assert client.get_latest_filing("EQIX", "10-K", "2023-12-31") == "OLD 10-K TEXT"
    # decision date after both: the 2024 filing wins
    assert client.get_latest_filing("EQIX", "10-K", "2024-12-31") == "NEW 10-K TEXT"
    # decision date before both: nothing eligible
    assert client.get_latest_filing("EQIX", "10-K", "2022-12-31") is None
    # filing dated exactly on the decision date is eligible (<= cutoff)
    assert client.get_latest_filing("EQIX", "10-K", "2023-02-01") == "OLD 10-K TEXT"
    # window queries must also exclude the post-cutoff filing
    window = client.get_filings_in_window("EQIX", "10-K", "2022-01-01", "2023-12-31")
    assert [f["filing_date"] for f in window] == ["2023-02-01"]


def test_missing_clean_text_file_handled(tmp_path: Path) -> None:
    """A metadata row whose clean_text file is absent must not raise."""
    rows = [
        {
            "ticker": "EQIX",
            "form": "10-Q",
            "filing_date": "2023-05-01",
            "accession_nodash": "accMISSING",
        },
        {
            "ticker": "EQIX",
            "form": "8-K",
            "filing_date": "2023-06-01",
            "accession_nodash": "acc8k1",
        },
        {
            "ticker": "EQIX",
            "form": "8-K",
            "filing_date": "2023-07-01",
            "accession_nodash": "acc8kMISSING",
        },
    ]
    texts = {
        # note: no file for accMISSING (10-Q) or acc8kMISSING (8-K)
        _filename("EQIX", "8-K", "2023-06-01", "acc8k1"): "8-K ONE",
    }
    client = _make_client(tmp_path, rows, texts)

    # latest filing with missing text file -> None, no exception
    assert client.get_latest_filing("EQIX", "10-Q", "2023-12-31") is None

    # window query silently skips the row whose file is missing
    window = client.get_filings_in_window("EQIX", "8-K", "2023-01-01", "2023-12-31")
    assert [f["accession_nodash"] for f in window] == ["acc8k1"]
    assert window[0]["text"] == "8-K ONE"


def test_filings_in_window_ascending_by_date(tmp_path: Path) -> None:
    """Returned filings must be ordered by filing_date ascending, even when
    the metadata CSV rows are written out of chronological order."""
    rows = [
        {
            "ticker": "EQIX",
            "form": "8-K",
            "filing_date": "2023-09-15",
            "accession_nodash": "accC",
        },
        {
            "ticker": "EQIX",
            "form": "8-K",
            "filing_date": "2023-03-10",
            "accession_nodash": "accA",
        },
        {
            "ticker": "EQIX",
            "form": "8-K",
            "filing_date": "2023-06-20",
            "accession_nodash": "accB",
        },
        {
            # outside the queried window: must be excluded
            "ticker": "EQIX",
            "form": "8-K",
            "filing_date": "2024-01-05",
            "accession_nodash": "accD",
        },
    ]
    texts = {
        _filename("EQIX", "8-K", "2023-09-15", "accC"): "C",
        _filename("EQIX", "8-K", "2023-03-10", "accA"): "A",
        _filename("EQIX", "8-K", "2023-06-20", "accB"): "B",
        _filename("EQIX", "8-K", "2024-01-05", "accD"): "D",
    }
    client = _make_client(tmp_path, rows, texts)

    window = client.get_filings_in_window("EQIX", "8-K", "2023-01-01", "2023-12-31")

    dates = [f["filing_date"] for f in window]
    assert dates == sorted(dates)
    assert dates == ["2023-03-10", "2023-06-20", "2023-09-15"]
    assert [f["text"] for f in window] == ["A", "B", "C"]
