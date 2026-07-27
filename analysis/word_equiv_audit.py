"""Word-equivalent audit of the typeset thesis under the school counting formula.

Rule (department guidance, Kalikman deck p.73, quoting school guidelines):
  "Tables must be in-text and all diagrams, maps, illustrations, graphs, symbols
   and pictures will be counted with an A4 page being assessed as equivalent to
   250 words, part pages being assessed on pro rata basis."
Operationalisation here:
  * body text words count as extracted (includes in-text tables, captions,
    footnotes, and equation tokens);
  * each figure's DISPLAYED area folds in at 250 words per full A4 page pro rata,
    while any text extracted from inside the figure is subtracted (not double-counted);
  * table of contents / lists are excluded (p.74); references assumed excluded
    (not stated); abstract reported separately;
  * appendices would count (p.67) — the thesis has none (moved to the Supplement).

Usage: python3 analysis/word_equiv_audit.py   (after building paper/main.pdf)
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "paper" / "main.pdf"
FIGDIR = REPO / "outputs" / "figures_v2"
A4_AREA = 595.32 * 841.92
TEXT_W = 450.8   # displayed width: all figures use width=\linewidth
LIMIT = 12_000

W = lambda s: len(re.findall(r"\S+", s))


def chapter_pages(doc: fitz.Document) -> dict[str, int]:
    """First physical page of each big-font heading (chapters, References)."""
    out: dict[str, int] = {}
    for i, p in enumerate(doc):
        for b in p.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    txt = s["text"].strip()
                    if s["size"] > 20 and txt and txt not in out:
                        out[txt] = i + 1
    return out


def main() -> None:
    doc = fitz.open(PDF)
    pt = [p.get_text() for p in doc]
    heads = chapter_pages(doc)
    body0 = heads["Chapter 1"]
    _bib = heads.get("Bibliography") or heads.get("References")
    refs0 = _bib
    body_pages = range(body0, refs0)          # 1-based, refs excluded

    body = sum(W(pt[i - 1]) for i in body_pages) - len(list(body_pages))  # -page numbers
    front = sum(W(pt[i]) for i in range(0, body0 - 1))
    refs = sum(W(pt[i - 1]) for i in range(refs0, len(doc) + 1))
    abstract = W(pt[1]) if len(doc) > 1 else 0

    # figures actually referenced by the build
    body_tex = (REPO / "paper" / "_body.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"outputs/figures_v2/(\w+)\.(?:pdf|png)", body_tex))
    fig_int, fig_fold = 0, 0.0
    for stem in sorted(used):
        fp = FIGDIR / f"{stem}.pdf"
        d = fitz.open(fp)
        pg = d[0]
        fig_int += W(pg.get_text())
        disp_h = pg.rect.height * (TEXT_W / pg.rect.width)
        fig_fold += 250 * (TEXT_W * disp_h) / A4_AREA
        print(f"  fig {stem:34s} fold {250*(TEXT_W*disp_h)/A4_AREA:5.0f}w  "
              f"internal {W(pg.get_text())}")
    # native TikZ figures: measure from top margin to their caption on-page
    for m in re.finditer(r"\\input\{_figures/(\w+)\}", body_tex):
        for i in body_pages:
            hits = doc[i - 1].search_for("Figure 1:")
            if hits:
                h = hits[0].y0 - 72
                fig_fold += 250 * (TEXT_W * h) / A4_AREA
                fig_int += W(doc[i - 1].get_text(clip=fitz.Rect(0, 0, 596, hits[0].y0)))
                print(f"  fig {m.group(1):34s} fold {250*(TEXT_W*h)/A4_AREA:5.0f}w  "
                      f"internal (page-top) counted")
                break

    countable = body - fig_int + fig_fold
    print(f"""
==== school-formula word equivalents ====
front matter (title/abstract/ToC/LoF, excluded): {front:6d}  (abstract alone: {abstract})
body text pp.{body0}-{refs0-1} (tables/captions/footnotes/equations in): {body:6d}
  - figure-internal text: -{fig_int}
  + figure area at 250w/A4 pro rata: +{fig_fold:.0f}
references pp.{refs0}- (assumed excluded): {refs:6d}
------------------------------------------------
COUNTABLE TOTAL: {countable:8.0f}  vs limit {LIMIT:,}
MARGIN: {LIMIT - countable:+8.0f}""")


if __name__ == "__main__":
    main()
