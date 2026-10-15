"""Factory functions that wire components from `settings` (used by scripts, eval and API)."""

from meddoc_rag.config import Settings, settings
from meddoc_rag.indexing.bm25_index import BM25Index
from meddoc_rag.indexing.dense_index import DenseIndex
from meddoc_rag.indexing.embedder import EmbeddingCache, SentenceTransformerEmbedder
from meddoc_rag.retrieval.base import Retriever
from meddoc_rag.retrieval.bm25 import BM25Retriever
from meddoc_rag.retrieval.dense import DenseRetriever
from meddoc_rag.retrieval.hybrid import HybridRetriever

RETRIEVERS = ("dense", "bm25", "hybrid")


def make_embedder(cfg: Settings = settings, use_cache: bool = True) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        model_name=cfg.embedding_model,
        query_instruction=cfg.query_instruction,
        cache=EmbeddingCache(cfg.embedding_cache_path) if use_cache else None,
        batch_size=cfg.embed_batch_size,
        device=cfg.device,
    )


def _require(path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"no {what} at {path}; run scripts/build_index.py first")


def load_dense_retriever(strategy: str, embedder: SentenceTransformerEmbedder, cfg: Settings = settings) -> DenseRetriever:
    _require(cfg.index_dir(strategy) / "index.faiss", "dense index")
    return DenseRetriever(DenseIndex.load(cfg.index_dir(strategy)), embedder)


def load_bm25_retriever(strategy: str, cfg: Settings = settings) -> BM25Retriever:
    _require(cfg.index_dir(strategy) / "bm25_tokens.json", "BM25 index")
    return BM25Retriever(BM25Index.load(cfg.index_dir(strategy)))


def load_retriever(name: str, strategy: str, embedder: SentenceTransformerEmbedder | None = None, cfg: Settings = settings) -> Retriever:
    """Build a retriever by name: 'dense', 'bm25' or 'hybrid' (dense + BM25 fused with RRF)."""
    if name == "bm25":
        return load_bm25_retriever(strategy, cfg)
    embedder = embedder or make_embedder(cfg, use_cache=False)
    if name == "dense":
        return load_dense_retriever(strategy, embedder, cfg)
    if name == "hybrid":
        return HybridRetriever(
            [load_dense_retriever(strategy, embedder, cfg), load_bm25_retriever(strategy, cfg)],
            candidates=cfg.hybrid_candidates,
            rrf_k=cfg.rrf_k,
        )
    raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVERS}")
