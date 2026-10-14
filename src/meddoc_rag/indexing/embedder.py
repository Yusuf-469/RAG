"""Text embeddings with an on-disk cache.

Vectors are L2-normalised, so inner product == cosine similarity.
"""

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    """Anything that maps text to unit-length float32 vectors."""

    model_name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class EmbeddingCache:
    """SQLite map from sha256(model, text) to vector bytes.

    Keying on the model name means switching models can never return stale
    vectors; keying on the text means re-chunking only embeds new chunks.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    @staticmethod
    def key(model_name: str, text: str) -> str:
        return hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):  # stay under SQLite's bound-parameter limit
            batch = keys[i:i + 500]
            rows = self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})", batch
            )
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class SentenceTransformerEmbedder:
    """sentence-transformers model with bge-style query instruction and caching of document vectors."""

    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.cache = cache
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: importing torch + loading weights takes seconds
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def _encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return vecs.astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.cache is None:
            return self._encode(texts, progress=len(texts) > 100)
        keys = [EmbeddingCache.key(self.model_name, t) for t in texts]
        cached = self.cache.get_many(keys)
        missing = sorted({k: i for i, k in enumerate(keys) if k not in cached}.values())
        log.info("embedding %d texts (%d cached, %d new)", len(texts), len(texts) - len(missing), len(missing))
        if missing:
            new = self._encode([texts[i] for i in missing], progress=len(missing) > 100)
            fresh = {keys[i]: v for i, v in zip(missing, new)}
            self.cache.put_many(fresh)
            cached.update(fresh)
        return np.stack([cached[k] for k in keys])

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_instruction + query])[0]
