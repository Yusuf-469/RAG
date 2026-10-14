"""Unit tests for the chunkers, using a word counter so they need no model download."""

import pytest

from meddoc_rag.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    SectionAwareChunker,
    WhitespaceCounter,
)
from meddoc_rag.schemas import Heading, ParsedDocument

WC = WhitespaceCounter()


def make_doc(pages: list[str], headings: list[tuple[str, int]] = ()) -> ParsedDocument:
    """Build a ParsedDocument from page texts; headings are (title, level) placed where the title occurs."""
    text = "\n\n".join(pages)
    starts, pos = [], 0
    for p in pages:
        starts.append(pos)
        pos += len(p) + 2
    hs = [Heading(offset=text.index(t), level=lvl, title=t) for t, lvl in headings]
    return ParsedDocument(
        doc_id="d", title="Doc", publisher="P", year=2020, text=text,
        page_starts=starts, page_labels=[None] * len(pages), headings=hs,
        heading_source="toc" if hs else "none",
    )


def sentences(n: int, prefix: str = "s") -> str:
    return " ".join(f"{prefix}{i} has five words here." for i in range(n))


def test_fixed_size_respects_limit_and_overlap() -> None:
    doc = make_doc([" ".join(f"w{i}" for i in range(100))])
    chunks = FixedSizeChunker(WC, max_tokens=20, overlap_tokens=5).chunk(doc)
    assert all(c.n_tokens <= 20 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-5:] == b.text.split()[:5]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(doc.text.split())


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(WC, max_tokens=10, overlap_tokens=10)


def test_recursive_prefers_sentence_boundaries() -> None:
    doc = make_doc([sentences(20)])
    chunks = RecursiveChunker(WC, max_tokens=12, overlap_tokens=0).chunk(doc)
    assert all(c.n_tokens <= 12 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_recursive_keeps_paragraphs_whole_when_they_fit() -> None:
    para_a, para_b = sentences(2, "a"), sentences(2, "b")
    doc = make_doc([f"{para_a}\n\n{para_b}"])
    chunks = RecursiveChunker(WC, max_tokens=12, overlap_tokens=0).chunk(doc)
    assert [c.text for c in chunks] == [para_a, para_b]


def test_recursive_hard_splits_unbreakable_text() -> None:
    doc = make_doc(["x" * 10 + " " + "y" * 10])  # no separators except one space; limit 1 word
    chunks = RecursiveChunker(WC, max_tokens=1, overlap_tokens=0).chunk(doc)
    assert [c.text for c in chunks] == ["x" * 10, "y" * 10]


def test_page_numbers_follow_offsets() -> None:
    doc = make_doc([sentences(3, "p1"), sentences(3, "p2")])
    chunks = FixedSizeChunker(WC, max_tokens=10, overlap_tokens=0).chunk(doc)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)


def test_section_aware_never_crosses_headings() -> None:
    doc = make_doc(
        [f"Intro Title\n\n{sentences(3, 'i')}\n\nMethods Title\n\n{sentences(3, 'm')}"],
        headings=[("Intro Title", 1), ("Methods Title", 1)],
    )
    chunks = SectionAwareChunker(WC, max_tokens=50, overlap_tokens=0, min_section_tokens=1).chunk(doc)
    heads = [h.offset for h in doc.headings]
    assert len(chunks) == 2
    assert not any(c.start < o < c.end for c in chunks for o in heads)
    assert [c.section for c in chunks] == ["Intro Title", "Methods Title"]


def test_recursive_does_cross_headings_when_it_fits() -> None:
    doc = make_doc(
        [f"Intro Title\n\n{sentences(3, 'i')}\n\nMethods Title\n\n{sentences(3, 'm')}"],
        headings=[("Intro Title", 1), ("Methods Title", 1)],
    )
    chunks = RecursiveChunker(WC, max_tokens=50, overlap_tokens=0).chunk(doc)
    assert len(chunks) == 1


def test_tiny_parent_heading_merges_into_child_and_breadcrumb_nests() -> None:
    doc = make_doc(
        [f"3 Recommendations\n\n3.1 Initiation\n\n{sentences(4)}"],
        headings=[("3 Recommendations", 1), ("3.1 Initiation", 2)],
    )
    chunks = SectionAwareChunker(WC, max_tokens=100, overlap_tokens=0, min_section_tokens=5).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("3 Recommendations")
    assert chunks[0].section == "3 Recommendations > 3.1 Initiation"


def test_chunk_ids_are_unique_and_stable() -> None:
    doc = make_doc([sentences(30)])
    a = RecursiveChunker(WC, max_tokens=15, overlap_tokens=3).chunk(doc)
    b = RecursiveChunker(WC, max_tokens=15, overlap_tokens=3).chunk(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
