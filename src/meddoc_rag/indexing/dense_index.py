"""Exact (brute-force) vector search with FAISS.

`IndexFlatIP` compares the query with every vector. At ~4k chunks this takes
well under a millisecond and is exact, so evaluation differences come from
the retrieval method, never from approximate-search error. HNSW/IVF only pay
off at millions of vectors.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from meddoc_rag.io import read_chunks, write_chunks
from meddoc_rag.schemas import Chunk


class DenseIndex:
    """FAISS inner-product index over unit vectors, aligned with a list of chunks."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    @property
    def dim(self) -> int:
        return self.index.d

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (chunk position, cosine score), best first."""
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        write_chunks(directory / "chunks.jsonl", self.chunks)
        meta = {"model_name": self.model_name, "dim": self.dim, "size": len(self.chunks)}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / "index.faiss"))
        obj = cls.__new__(cls)
        obj.chunks = read_chunks(directory / "chunks.jsonl")
        obj.model_name = meta["model_name"]
        obj.index = index
        if obj.index.ntotal != len(obj.chunks):
            raise ValueError(f"corrupt index in {directory}: {obj.index.ntotal} vectors, {len(obj.chunks)} chunks")
        return obj
