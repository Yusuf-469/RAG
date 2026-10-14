"""Dense retrieval tests with a deterministic fake embedder (no model download)."""

import re
from pathlib import Path

import numpy as np
import pytest

from meddoc_rag.indexing.dense_index import DenseIndex
from meddoc_rag.indexing.embedder import EmbeddingCache, SentenceTransformerEmbedder
from meddoc_rag.retrieval.dense import DenseRetriever
from meddoc_rag.schemas import Chunk

DIM = 64


class FakeEmbedder:
    """Bag-of-words hashed into DIM buckets, L2-normalised: shared words => higher cosine."""

    def __init__(self, model_name: str = "fake") -> None:
        self.model_name = model_name
        self.dim = DIM

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[sum(map(ord, w)) % DIM] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


def chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"d:x:{i:04d}", doc_id="d", doc_title="D", strategy="x", text=text,
                 start=0, end=len(text), page_start=i + 1, page_end=i + 1, section=None, n_tokens=len(text.split()))


CHUNKS = [
    chunk(0, "Hypertension treatment threshold is 140 mmHg systolic."),
    chunk(1, "Children should do 60 minutes of physical activity daily."),
    chunk(2, "Wash hands with alcohol based hand rub before patient contact."),
]


def build(embedder: FakeEmbedder) -> DenseIndex:
    return DenseIndex(CHUNKS, embedder.embed_documents([c.text for c in CHUNKS]), embedder.model_name)


def test_most_similar_chunk_ranks_first() -> None:
    emb = FakeEmbedder()
    results = DenseRetriever(build(emb), emb).retrieve("hand rub before patient contact", k=3)
    assert results[0].chunk.chunk_id == "d:x:0002"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score


def test_k_larger_than_index_is_clamped() -> None:
    emb = FakeEmbedder()
    assert len(DenseRetriever(build(emb), emb).retrieve("anything", k=50)) == 3


def test_save_load_roundtrip(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    build(emb).save(tmp_path)
    loaded = DenseIndex.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in CHUNKS]
    assert DenseRetriever(loaded, emb).retrieve("physical activity children", 1)[0].chunk.chunk_id == "d:x:0001"


def test_model_mismatch_fails_loudly() -> None:
    index = build(FakeEmbedder("model-a"))
    with pytest.raises(ValueError, match="model-a"):
        DenseRetriever(index, FakeEmbedder("model-b"))


def test_cache_key_depends_on_model_and_text() -> None:
    k = EmbeddingCache.key
    assert k("m1", "t") != k("m2", "t")
    assert k("m1", "t") != k("m1", "t2")
    assert k("m1", "t") == k("m1", "t")


def test_cached_embedder_only_encodes_new_texts(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class CountingEmbedder(SentenceTransformerEmbedder):
        dim = DIM  # type: ignore[assignment]

        def _encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
            calls.append(texts)
            return FakeEmbedder().embed_documents(texts)

    cache = EmbeddingCache(tmp_path / "c.sqlite")
    emb = CountingEmbedder("fake", cache=cache)
    first = emb.embed_documents(["a b", "c d"])
    second = emb.embed_documents(["c d", "e f", "a b"])
    cache.close()

    assert calls == [["a b", "c d"], ["e f"]]
    np.testing.assert_allclose(first[0], second[2])
