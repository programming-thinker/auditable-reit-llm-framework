"""Unit tests for the fixed llm.orchestrator._extract_item1a.

Real-file cases use 10-Ks under filings/clean_text (read-only), including
empirically identified cross-reference cases where the OLD logic (anchor on
the FIRST 'item 1a' hit) landed on a cross-reference inside the Item 1
business description instead of the genuine Risk Factors section:

  - DLR 2024 10-K: first hit at ~99k is '... in Item 1A. Risk Factors for
    further discussion' (business text); genuine header 'ITEM 1A. RISK
    FACTORS For purposes of this section ...' is at ~133k.
  - ARE 2024 10-K: first hit at ~44k is a cross-reference; genuine header
    'ITEM 1A. RISK FACTORS Overview ...' is at ~75k.

Also covers OCR-split headers (MAA 'Ris k Factors'), regression equality
where the old anchor was already correct (EQIX), and fallback-to-old-logic
cases (AMT: no genuine section in clean text; VICI: no 'Item 1B' marker).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm.orchestrator import _extract_item1a

REPO = Path(__file__).resolve().parents[2]
CLEAN_TEXT = REPO / "filings" / "clean_text"

pytestmark = pytest.mark.skipif(
    not CLEAN_TEXT.is_dir(), reason="filings/clean_text not available"
)


def _original_extract_item1a(raw):
    """Reference replica of the pre-fix (as-run) extraction logic."""
    if not raw:
        return raw
    starts = [m.start() for m in re.finditer(r"item\s*1a", raw, re.I)]
    ends = [m.start() for m in re.finditer(r"item\s*1b", raw, re.I)]
    if not starts:
        m = re.search(r"risk factors", raw, re.I)
        return raw[m.start():] if m else raw
    s = starts[0]
    e = next((x for x in ends if x > s), len(raw))
    return raw[s:e]


def _read(fname: str) -> str:
    return (CLEAN_TEXT / fname).read_text(encoding="utf-8")


# ── real-file cases ───────────────────────────────────────────────────────


def test_dlr_2024_cross_reference_case() -> None:
    """DLR 10-K: OLD anchor is a cross-reference in Item 1 (business text);
    the fix must anchor on the genuine section header instead."""
    raw = _read("DLR_10-K_2024-02-23_000155837024001575.txt")

    old = _original_extract_item1a(raw)
    # old anchor is the cross-reference '... in Item 1A. Risk Factors for
    # further discussion ...' -> excerpt continues with business narrative
    assert old.lower().startswith("item 1a. risk factors for further discussion")

    new = _extract_item1a(raw)
    assert new.startswith("ITEM 1A. RISK FACTORS For purposes of this section")
    assert new != old
    # the returned section is substantial, not a trailing cross-ref fragment
    assert len(new) > 50_000


def test_are_2024_cross_reference_case() -> None:
    raw = _read("ARE_10-K_2024-01-29_000103544324000072.txt")

    old = _original_extract_item1a(raw)
    new = _extract_item1a(raw)
    assert new.startswith("ITEM 1A. RISK FACTORS Overview")
    # OLD anchored ~31k chars earlier, on a cross-reference
    assert old != new
    assert not old.startswith("ITEM 1A. RISK FACTORS Overview")


def test_maa_2024_ocr_split_header() -> None:
    """MAA's genuine header is OCR-split ('Ris k Factors'); the tolerant
    header pattern must still find it."""
    raw = _read("MAA_10-K_2024-02-09_000095017024013275.txt")
    new = _extract_item1a(raw)
    assert new.startswith("Item 1A. Ris k Factors.")
    assert len(new) > 50_000


def test_eqix_2024_unchanged_when_old_anchor_correct() -> None:
    """EQIX's first 'item 1a' hit IS the genuine header: fix must not
    change behaviour where the old logic was already right."""
    raw = _read("EQIX_10-K_2024-02-16_000162828024005350.txt")
    assert _extract_item1a(raw) == _original_extract_item1a(raw)
    assert _extract_item1a(raw).startswith("ITEM 1A. Risk Factors")


def test_amt_2024_fallback_no_genuine_section() -> None:
    """AMT clean text contains only cross-references to Item 1A (no genuine
    section survives cleaning): must fall back to the original logic."""
    raw = _read("AMT_10-K_2024-02-27_000105350724000011.txt")
    # no 'Item 1B' marker anywhere -> no candidate can be validated
    assert not re.search(r"item\s*1\s*b", raw, re.I)
    assert _extract_item1a(raw) == _original_extract_item1a(raw)


def test_vici_2024_fallback_no_item1b() -> None:
    """VICI 10-K has no 'Item 1B' marker at all: with no way to validate a
    section start, must fall back to the original logic."""
    raw = _read("VICI_10-K_2024-02-22_000170569624000033.txt")
    assert _extract_item1a(raw) == _original_extract_item1a(raw)


# ── synthetic cases ───────────────────────────────────────────────────────


def test_none_and_empty_passthrough() -> None:
    assert _extract_item1a(None) is None
    assert _extract_item1a("") == ""


def test_synthetic_prefers_section_over_toc_and_cross_reference() -> None:
    doc = (
        "TABLE OF CONTENTS Item 1. Business 3 Item 1A. Risk Factors 10 "
        "Item 1B. Unresolved Staff Comments 45 "
        "Item 1. Business We operate towers. As set forth in Item 1A. Risk "
        "Factors, demand may decline. "
        + "Business prose. " * 400
        + "Item 1A. Risk Factors The following risks could materially affect us. "
        + "Risk prose. " * 600
        + "Item 1B. Unresolved Staff Comments None."
    )
    out = _extract_item1a(doc)
    assert out.startswith("Item 1A. Risk Factors The following risks")
    assert "Business prose." not in out
    assert out.rstrip().endswith("Risk prose.")


def test_synthetic_fallback_first_hit_when_no_header_form() -> None:
    doc = (
        "Intro text. Item 1A of this report describes various matters. "
        "More narrative. Item 1B follows here."
    )
    out = _extract_item1a(doc)
    assert out.startswith("Item 1A of this report")
    assert "Item 1B" not in out


def test_synthetic_fallback_risk_factors_when_no_item1a() -> None:
    doc = "Nothing here about items. But risk factors are described anyway."
    out = _extract_item1a(doc)
    assert out.startswith("risk factors are described")
