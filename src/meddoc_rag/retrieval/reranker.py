"""Second-stage reranking with a cross-encoder.

A bi-encoder (dense search) embeds query and passage separately, so passages can
be pre-computed and search is fast but shallow. A cross-encoder reads
"[CLS] query [SEP] passage [SEP]" in one pass, letting attention compare them
token by token: far more accurate, but one model run per (query, passage) pair,
so it is only applied to a small candidate set from the first stage.
"""

import math
from typing import Protocol

import numpy as np

from meddoc_rag.retrieval.base import Retriever
from meddoc_rag.schemas import RetrievedChunk


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Raw relevance logits, one per passage (higher = more relevant)."""
        ...


def sigmoid(x: float) -> float:
    """Map a logit to (0, 1): a fixed scale across queries, used later for the refusal threshold."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder returning raw logits (activation pinned to Identity)."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        import torch

        return self.model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            activation_fn=torch.nn.Identity(),  # explicit: never depend on a library default
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class RerankingRetriever:
    """Retrieve `candidates` with a first-stage retriever, rescore them with a reranker, return top k.

    `score` on results is sigmoid(logit) in (0, 1). `component_ranks` keeps the
    first-stage ranks so you can see how far reranking moved each chunk.
    """

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 30) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{base.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        first = self.base.retrieve(query, self.candidates)
        logits = self.reranker.score(query, [r.chunk.text for r in first])
        order = np.argsort(-logits, kind="stable")[:k]
        return [
            RetrievedChunk(
                chunk=first[i].chunk,
                score=sigmoid(float(logits[i])),
                rank=new_rank,
                retriever=self.name,
                component_ranks={**first[i].component_ranks, self.base.name: first[i].rank},
            )
            for new_rank, i in enumerate(order, start=1)
        ]
