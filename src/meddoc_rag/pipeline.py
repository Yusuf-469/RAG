"""Factory functions that wire components from `settings` (used by scripts, eval and API)."""

from meddoc_rag.config import Settings, settings
from meddoc_rag.indexing.dense_index import DenseIndex
from meddoc_rag.indexing.embedder import EmbeddingCache, SentenceTransformerEmbedder
from meddoc_rag.retrieval.dense import DenseRetriever


def make_embedder(cfg: Settings = settings, use_cache: bool = True) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        model_name=cfg.embedding_model,
        query_instruction=cfg.query_instruction,
        cache=EmbeddingCache(cfg.embedding_cache_path) if use_cache else None,
        batch_size=cfg.embed_batch_size,
        device=cfg.device,
    )


def load_dense_retriever(strategy: str, embedder: SentenceTransformerEmbedder, cfg: Settings = settings) -> DenseRetriever:
    index_dir = cfg.index_dir(strategy)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"no index for '{strategy}'; run scripts/build_index.py first")
    return DenseRetriever(DenseIndex.load(index_dir), embedder)
