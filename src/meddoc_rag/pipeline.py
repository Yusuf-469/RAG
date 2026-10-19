"""Factory functions that wire components from `settings` (used by scripts, eval and API)."""

from meddoc_rag.config import Settings, settings
from meddoc_rag.indexing.bm25_index import BM25Index
from meddoc_rag.indexing.dense_index import DenseIndex
from meddoc_rag.indexing.embedder import EmbeddingCache, SentenceTransformerEmbedder
from meddoc_rag.retrieval.base import Retriever
from meddoc_rag.retrieval.bm25 import BM25Retriever
from meddoc_rag.retrieval.dense import DenseRetriever
from meddoc_rag.retrieval.hybrid import HybridRetriever
from meddoc_rag.retrieval.reranker import CrossEncoderReranker, Reranker, RerankingRetriever

BASE_RETRIEVERS = ("dense", "bm25", "hybrid")
RETRIEVERS = (*BASE_RETRIEVERS, *(f"{b}+rerank" for b in BASE_RETRIEVERS))


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


def make_reranker(cfg: Settings = settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(cfg.reranker_model, device=cfg.device)


def load_retriever(
    name: str,
    strategy: str,
    embedder: SentenceTransformerEmbedder | None = None,
    cfg: Settings = settings,
    reranker: Reranker | None = None,
) -> Retriever:
    """Build a retriever by name: 'dense', 'bm25', 'hybrid', or any of these + '+rerank'."""
    if name.endswith("+rerank"):
        base = load_retriever(name.removesuffix("+rerank"), strategy, embedder, cfg)
        return RerankingRetriever(base, reranker or make_reranker(cfg), cfg.rerank_candidates)
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


def make_llm(cfg: Settings = settings, use_cache: bool = True):
    """LLM client for `cfg.llm_provider`, wrapped in the response cache by default."""
    from meddoc_rag.generation.llm import CachedLLM, GroqClient, OllamaClient

    if cfg.llm_provider == "groq":
        key = cfg.groq_api_key.get_secret_value() if cfg.groq_api_key else ""
        client = GroqClient(key, cfg.llm_model, cfg.groq_base_url)
    else:
        client = OllamaClient(cfg.llm_model, cfg.ollama_base_url)
    return CachedLLM(client, cfg.llm_cache_path) if use_cache else client


def make_answerer(retriever_name: str | None = None, strategy: str | None = None, cfg: Settings = settings):
    """Full question-answering pipeline. The refusal gate is only applied to reranked
    retrievers, because only the cross-encoder score is on a fixed scale across queries."""
    from meddoc_rag.generation.answerer import Answerer

    name = retriever_name or cfg.retriever
    retriever = load_retriever(name, strategy or cfg.chunk_strategy, cfg=cfg)
    return Answerer(
        retriever,
        make_llm(cfg),
        context_chunks=cfg.context_chunks,
        refusal_threshold=cfg.refusal_threshold if name.endswith("+rerank") else None,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        input_guard=make_input_guard(cfg),
        guard_threshold=cfg.input_guard_threshold,
        guard_fail_open=cfg.input_guard_fail_open,
    )


def make_input_guard(cfg: Settings = settings):
    """Prompt-injection classifier for user questions (Groq), or None when disabled / no key."""
    from meddoc_rag.safety import PromptGuard

    if not cfg.input_guard_enabled or not cfg.groq_api_key:
        return None
    return PromptGuard(cfg.groq_api_key.get_secret_value(), cfg.input_guard_model, cfg.groq_base_url)
