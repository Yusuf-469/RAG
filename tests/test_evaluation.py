"""Tests for golden-set resolution and retrieval metrics."""

from pathlib import Path

import pytest

from meddoc_rag.config import PROJECT_ROOT
from meddoc_rag.evaluation.golden import GoldSpan, find_quote, load_golden
from meddoc_rag.evaluation.metrics import coverage, first_relevant_rank, hit_at_k, mrr, wrong_doc_at_1
from meddoc_rag.schemas import Chunk, ParsedDocument


def doc(text: str) -> ParsedDocument:
    return ParsedDocument(doc_id="d", title="D", publisher="P", year=2020, text=text,
                          page_starts=[0], page_labels=[None], headings=[], heading_source="none")


def chunk(start: int, end: int, doc_id: str = "d") -> Chunk:
    return Chunk(chunk_id=f"{doc_id}:{start}", doc_id=doc_id, doc_title="D", strategy="x", text="t",
                 start=start, end=end, page_start=1, page_end=1, section=None, n_tokens=1)


def test_find_quote_ignores_whitespace_differences_and_finds_all() -> None:
    text = "Adults should do 150– 300 minutes. Again: Adults should do 150–300\nminutes."
    spans = find_quote(doc(text), "Adults should do 150–300 minutes")
    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == "Adults should do 150– 300 minutes"


def test_find_quote_is_case_sensitive() -> None:
    # "Adults" must not match inside "Older adults" (population regression).
    assert find_quote(doc("Older adults should do X."), "Adults should do X") == []


def test_coverage_and_relevance() -> None:
    g = GoldSpan(doc_id="d", start=100, end=200, page_start=1, page_end=1)
    assert coverage(chunk(0, 150), g) == 0.5
    assert coverage(chunk(0, 300), g) == 1.0
    assert coverage(chunk(0, 300, doc_id="other"), g) == 0.0
    assert first_relevant_rank([chunk(0, 120), chunk(140, 260)], [g]) == 2   # first covers only 20%


def test_hit_and_mrr() -> None:
    ranks = [1, 3, None, 12]
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 5) == 0.5
    assert mrr(ranks, cutoff=10) == pytest.approx((1 + 1 / 3) / 4)


def test_wrong_doc_rate() -> None:
    g = [[GoldSpan(doc_id="a", start=0, end=1, page_start=1, page_end=1)]] * 2
    assert wrong_doc_at_1([chunk(0, 1, "a"), chunk(0, 1, "b")], g) == 0.5


def test_golden_set_file_is_well_formed() -> None:
    items = load_golden(PROJECT_ROOT / "eval" / "golden_set.jsonl")
    assert 40 <= sum(i.answerable for i in items) <= 60
    assert sum(not i.answerable for i in items) >= 8


def test_answerable_requires_evidence(tmp_path: Path) -> None:
    bad = tmp_path / "g.jsonl"
    bad.write_text('{"id": "x", "type": "fact", "answerable": true, "question": "q", "evidence": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="need evidence"):
        load_golden(bad)
