"""Tests for BM25 tokenisation/search and Reciprocal Rank Fusion."""

from pathlib import Path

import pytest

from meddoc_rag.indexing.bm25_index import BM25Index, tokenize
from meddoc_rag.retrieval.bm25 import BM25Retriever
from meddoc_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from meddoc_rag.schemas import Chunk, RetrievedChunk


def chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="d", doc_title="D", strategy="x", text=text,
                 start=0, end=len(text), page_start=1, page_end=1, section=None, n_tokens=len(text.split()))


CHUNKS = [
    chunk(0, "Before increasing total opioid dosage to ≥50 MME/day, clinicians should pause."),
    chunk(1, "Clinicians should avoid increasing dosage to ≥90 MME/day."),
    chunk(2, "Patients with chronic pain benefit from nonopioid therapies and exercise."),
    chunk(3, "Hand hygiene with alcohol-based hand rub before patient contact."),
]


def test_tokenize_splits_units_keeps_numbers_negations_and_stems() -> None:
    assert tokenize("≥50 MME/day") == ["50", "mme", "day"]
    assert tokenize("Do NOT exceed") == ["not", "exceed"]           # "do" is a stopword, "not" is kept
    assert tokenize("recommendations recommended") == ["recommend", "recommend"]
    assert tokenize("doing workouts") == ["workout"]                 # "doing" stems to stopword "do"


def test_bm25_prefers_exact_number() -> None:
    hits = BM25Index(CHUNKS).search("90 MME per day", k=4)
    assert hits[0][0] == 1  # the only chunk containing "90"


def test_bm25_drops_chunks_without_any_query_term() -> None:
    hits = BM25Index(CHUNKS).search("alcohol hand rub", k=4)
    assert [i for i, _ in hits] == [3]


def test_bm25_empty_query_after_stopwords() -> None:
    assert BM25Index(CHUNKS).search("the of and", k=3) == []


def test_bm25_save_load_roundtrip(tmp_path: Path) -> None:
    BM25Index(CHUNKS).save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.search("opioid dosage", 2) == BM25Index(CHUNKS).search("opioid dosage", 2)


def test_rrf_exact_scores() -> None:
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "c"]}, k=60)
    scores = {d: s for d, s, _ in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)   # rank 2 in dense, rank 1 in bm25
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [d for d, _, _ in fused] == ["b", "a", "c"]
    assert dict((d, r) for d, _, r in fused)["b"] == {"dense": 2, "bm25": 1}


def test_rrf_agreement_beats_single_top_rank() -> None:
    # "x" is 3rd in both lists; "y" is 1st in one list only. Agreement wins.
    fused = reciprocal_rank_fusion({"a": ["y", "p", "x"], "b": ["q", "r", "x"]})
    assert fused[0][0] == "x"


class StaticRetriever:
    def __init__(self, name: str, order: list[int]) -> None:
        self.name = name
        self.order = order

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=CHUNKS[i], score=1.0, rank=r, retriever=self.name)
                for r, i in enumerate(self.order[:k], start=1)]


def test_hybrid_dedupes_and_reports_component_ranks() -> None:
    hybrid = HybridRetriever([StaticRetriever("dense", [2, 0, 1]), StaticRetriever("bm25", [0, 1])], candidates=10)
    results = hybrid.retrieve("q", k=3)
    assert [r.chunk.chunk_id for r in results] == ["c0", "c1", "c2"]
    assert len({r.chunk.chunk_id for r in results}) == 3
    assert results[0].component_ranks == {"dense": 2, "bm25": 1}
    assert [r.rank for r in results] == [1, 2, 3]


def test_bm25_retriever_ranks_start_at_one() -> None:
    results = BM25Retriever(BM25Index(CHUNKS)).retrieve("opioid", 2)
    assert results[0].rank == 1 and results[0].retriever == "bm25"
