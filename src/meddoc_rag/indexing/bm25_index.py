"""BM25 keyword index: catches exact terms (doses, drug names, acronyms) that embeddings blur."""

import json
import re
from pathlib import Path

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from meddoc_rag.io import read_chunks, write_chunks
from meddoc_rag.schemas import Chunk

# Small English stopword list. Negations ("no", "not", "nor") are deliberately kept:
# in clinical text they flip the meaning of a recommendation.
STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have
    how i if in into is it its may might more most must of on or our over shall should so
    such than that the their them then there these they this those through to under up
    us was we were what when where which while who whom why will with would you your""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics ("≥50 MME/day" -> 50, mme, day), drop stopwords, stem."""
    words = [w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]
    # Filter again after stemming: "doing" -> "do", "having" -> "have" are stopwords only once stemmed.
    return [s for s in _stemmer.stemWords(words) if s not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over chunk texts, aligned with `chunks` like the dense index."""

    def __init__(self, chunks: list[Chunk], tokens: list[list[str]] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.tokens = tokens if tokens is not None else [tokenize(c.text) for c in chunks]
        if len(self.tokens) != len(chunks):
            raise ValueError("tokens and chunks are misaligned")
        self.k1, self.b = k1, b
        self.bm25 = BM25Okapi(self.tokens, k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (chunk position, BM25 score), best first; chunks sharing no query term are excluded."""
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]           # O(n) top-k, then sort only k items
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, directory: Path) -> None:
        # Store tokens, not the BM25 object: rebuilding statistics from tokens takes
        # milliseconds, and JSON is inspectable and safe to load (unlike pickle).
        directory.mkdir(parents=True, exist_ok=True)
        write_chunks(directory / "chunks.jsonl", self.chunks)
        payload = {"k1": self.k1, "b": self.b, "tokens": self.tokens}
        (directory / "bm25_tokens.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        payload = json.loads((directory / "bm25_tokens.json").read_text(encoding="utf-8"))
        return cls(read_chunks(directory / "chunks.jsonl"), payload["tokens"], payload["k1"], payload["b"])
