"""Keyword-only retriever over the BM25 index."""

from meddoc_rag.indexing.bm25_index import BM25Index
from meddoc_rag.schemas import RetrievedChunk


class BM25Retriever:
    name = "bm25"

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk=self.index.chunks[i], score=score, rank=r, retriever=self.name)
            for r, (i, score) in enumerate(self.index.search(query, k), start=1)
        ]
