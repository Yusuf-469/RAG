"""Hybrid retrieval: run several retrievers and merge their rankings with Reciprocal Rank Fusion.

RRF uses ranks only, never raw scores: cosine similarities (~0.7-0.9) and BM25
scores (unbounded, query-dependent) are on incomparable scales, and learning
a weight between them would overfit a small golden set.
"""

from collections import defaultdict

from meddoc_rag.retrieval.base import Retriever
from meddoc_rag.schemas import Chunk, RetrievedChunk

RRF_K = 60  # from Cormack, Clarke & Buettcher (SIGIR 2009); damps the gap between rank 1 and rank 2


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank).

    Returns (id, fused score, {list name: rank}) sorted best first; ties keep
    first-seen order so results are deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    order: dict[str, int] = {}
    for name, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            ranks[doc_id][name] = rank
            order.setdefault(doc_id, len(order))
    fused = sorted(scores, key=lambda d: (-scores[d], order[d]))
    return [(d, scores[d], ranks[d]) for d in fused]


class HybridRetriever:
    """Fetch `candidates` results from each retriever, fuse with RRF, return the top k."""

    def __init__(self, retrievers: list[Retriever], candidates: int = 50, rrf_k: int = RRF_K) -> None:
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.name = "hybrid"

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        chunks: dict[str, Chunk] = {}
        rankings: dict[str, list[str]] = {}
        for retriever in self.retrievers:
            results = retriever.retrieve(query, self.candidates)
            rankings[retriever.name] = [r.chunk.chunk_id for r in results]
            chunks.update({r.chunk.chunk_id: r.chunk for r in results})
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
        return [
            RetrievedChunk(chunk=chunks[cid], score=score, rank=i, retriever=self.name, component_ranks=ranks)
            for i, (cid, score, ranks) in enumerate(fused, start=1)
        ]
