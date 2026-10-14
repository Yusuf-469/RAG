"""The interface every retriever implements, so eval and API can swap them by name."""

from typing import Protocol

from meddoc_rag.schemas import RetrievedChunk


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...
