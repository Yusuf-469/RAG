"""Baseline retriever: embed the query, return the k nearest chunks."""

from meddoc_rag.indexing.dense_index import DenseIndex
from meddoc_rag.indexing.embedder import TextEmbedder
from meddoc_rag.schemas import RetrievedChunk


class DenseRetriever:
    name = "dense"

    def __init__(self, index: DenseIndex, embedder: TextEmbedder) -> None:
        # A query embedded by a different model than the index lives in a different vector
        # space: search would "work" and return garbage. Fail loudly instead.
        if index.model_name != embedder.model_name:
            raise ValueError(f"index built with {index.model_name!r}, embedder is {embedder.model_name!r}")
        if index.dim != embedder.dim:
            raise ValueError(f"index dim {index.dim} != embedder dim {embedder.dim}")
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = self.index.search(self.embedder.embed_query(query), k)
        return [
            RetrievedChunk(chunk=self.index.chunks[i], score=score, rank=r, retriever=self.name)
            for r, (i, score) in enumerate(hits, start=1)
        ]
