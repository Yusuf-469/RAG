"""Tests for PDF loading, boilerplate removal and de-hyphenation."""

from pathlib import Path

import pymupdf

from meddoc_rag.ingestion.cleaner import clean_page, find_boilerplate, join_lines, normalize_text
from meddoc_rag.ingestion.document import parse_document
from meddoc_rag.ingestion.loader import Line, RawPage, load_pdf
from meddoc_rag.manifest import SourceDocument


def page(number: int, *lines: str) -> RawPage:
    return RawPage(number=number, label=None, blocks=[[Line(t, 10.0, False)] for t in lines])


def test_normalize_text_fixes_unicode_spaces_and_ligatures() -> None:
    assert normalize_text("screen time  and ﬁve  ") == "screen time and five"


def test_running_footer_removed_but_body_kept() -> None:
    pages = [page(i, "Guideline Title", f"Body text {i} about hypertension.", f"Page {i} of 9") for i in range(1, 10)]
    boiler = find_boilerplate(pages, edge=1)
    cleaned = [clean_page(p, boiler, edge=1) for p in pages]
    texts = [ln.text for blk in cleaned[0].blocks for ln in blk]
    assert texts == ["Body text 1 about hypertension."]


def test_boilerplate_only_removed_at_page_edges() -> None:
    pages = [page(i, "Header", "Recommendations", "Body", "More body", "Footer") for i in range(1, 10)]
    boiler = find_boilerplate(pages, edge=1)
    assert "Recommendations" not in boiler  # it is never at an edge, so never flagged


def test_dehyphenation_rules() -> None:
    vocab = {"treatment", "alcohol-based", "alcohol", "based"}
    assert join_lines("antihypertensive treat-", "ment is", vocab) == "antihypertensive treatment is"
    assert join_lines("use alcohol-", "based rubs", vocab) == "use alcohol-based rubs"
    assert join_lines("moderate-", "to vigorous", vocab) == "moderate- to vigorous"
    assert join_lines("plain", "line", vocab) == "plain line"


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    for i in range(1, 3):
        p = doc.new_page()
        p.insert_text((72, 72), f"Section {i} Heading", fontsize=18, fontname="hebo")
        p.insert_text((72, 110), f"Body sentence on page {i}.", fontsize=10)
    doc.set_toc([[1, "Section 1 Heading", 1], [1, "Section 2 Heading", 2]])
    doc.save(path)


def test_load_and_parse_small_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf)
    raw = load_pdf(pdf)
    assert len(raw.pages) == 2 and len(raw.toc) == 2

    meta = SourceDocument(id="t", title="T", publisher="P", year=2024, license="l",
                          source_page="s", url="u", filename="t.pdf")
    doc = parse_document(meta, pdf)
    assert doc.num_pages == 2
    assert doc.page_at(doc.text.index("page 2")) == 2
    # Only 2 bookmarks (< MIN_TOC_ENTRIES), so headings come from font size.
    assert doc.heading_source == "font"
    assert [h.title for h in doc.headings] == ["Section 1 Heading", "Section 2 Heading"]
