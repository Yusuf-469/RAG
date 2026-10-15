"""Reranking tests with a fake cross-encoder (no model download)."""

import numpy as np
import pytest

from meddoc_rag.retrieval.reranker import RerankingRetriever, sigmoid
from meddoc_rag.schemas import Chunk, RetrievedChunk


def chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"c{i}", doc_id="d", doc_title="D", strategy="x", text=text,
                 start=0, end=len(text), page_start=1, page_end=1, section=None, n_tokens=len(text.split()))


CHUNKS = [chunk(i, t) for i, t in enumerate(["alpha", "beta pregnant", "gamma", "delta pregnant women"])]


class FirstStage:
    name = "hybrid"

    def __init__(self) -> None:
        self.asked_k: int | None = None

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.asked_k = k
        return [RetrievedChunk(chunk=c, score=0.0, rank=r, retriever=self.name, component_ranks={"dense": r})
                for r, c in enumerate(CHUNKS[:k], start=1)]


class WordOverlapReranker:
    """Logit = number of query words present in the passage (deterministic stand-in)."""

    model_name = "fake"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        q = set(query.split())
        return np.array([len(q & set(p.split())) for p in passages], dtype=np.float32)


def test_rerank_reorders_by_reranker_score() -> None:
    r = RerankingRetriever(FirstStage(), WordOverlapReranker(), candidates=4)
    results = r.retrieve("pregnant women", k=2)
    assert [x.chunk.chunk_id for x in results] == ["c3", "c1"]
    assert [x.rank for x in results] == [1, 2]
    assert results[0].retriever == "hybrid+rerank"


def test_rerank_keeps_first_stage_ranks() -> None:
    results = RerankingRetriever(FirstStage(), WordOverlapReranker(), candidates=4).retrieve("pregnant women", k=1)
    assert results[0].component_ranks == {"dense": 4, "hybrid": 4}   # moved from #4 to #1


def test_rerank_asks_first_stage_for_candidates_not_k() -> None:
    first = FirstStage()
    RerankingRetriever(first, WordOverlapReranker(), candidates=3).retrieve("q", k=1)
    assert first.asked_k == 3


def test_scores_are_probabilities_and_sorted() -> None:
    results = RerankingRetriever(FirstStage(), WordOverlapReranker(), candidates=4).retrieve("pregnant women", k=4)
    scores = [x.score for x in results]
    assert all(0.0 < s < 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("x, expected", [(0.0, 0.5), (8.56, 0.99981), (-11.24, 1.3135e-5), (-800.0, 0.0), (800.0, 1.0)])
def test_sigmoid_values_and_no_overflow(x: float, expected: float) -> None:
    assert sigmoid(x) == pytest.approx(expected, rel=1e-3, abs=1e-12)
