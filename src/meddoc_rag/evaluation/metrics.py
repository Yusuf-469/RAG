"""Retrieval metrics computed against gold character spans.

A retrieved chunk is relevant if it comes from the same document and covers at
least `min_coverage` of a gold quote's characters. The same rule applies to
every chunking strategy, so strategies are compared fairly.
"""

import statistics

from meddoc_rag.evaluation.golden import GoldSpan
from meddoc_rag.schemas import Chunk


def coverage(chunk: Chunk, span: GoldSpan) -> float:
    """Fraction of the gold span's characters inside the chunk (0 if different document)."""
    if chunk.doc_id != span.doc_id:
        return 0.0
    overlap = min(chunk.end, span.end) - max(chunk.start, span.start)
    return max(0, overlap) / (span.end - span.start)


def is_relevant(chunk: Chunk, gold: list[GoldSpan], min_coverage: float = 0.5) -> bool:
    return any(coverage(chunk, g) >= min_coverage for g in gold)


def first_relevant_rank(chunks: list[Chunk], gold: list[GoldSpan], min_coverage: float = 0.5) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved."""
    return next((i for i, c in enumerate(chunks, start=1) if is_relevant(c, gold, min_coverage)), None)


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions with a relevant chunk in the top k."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks: list[int | None], cutoff: int = 10) -> float:
    """Mean reciprocal rank of the first relevant chunk (0 when missing or beyond cutoff)."""
    return statistics.mean(1 / r if r is not None and r <= cutoff else 0.0 for r in ranks)


def wrong_doc_at_1(top_chunks: list[Chunk], gold: list[list[GoldSpan]]) -> float:
    """Share of questions whose top result comes from a document with no gold evidence (confusion rate)."""
    misses = [c.doc_id not in {g.doc_id for g in spans} for c, spans in zip(top_chunks, gold)]
    return sum(misses) / len(misses)
