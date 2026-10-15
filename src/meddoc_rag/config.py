"""Central configuration, read from environment variables / a `.env` file.

Every tunable value lives here so that experiments are driven by config,
not by editing code. Later milestones add model names, top-k, thresholds, etc.
"""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Override any field with an env var prefixed `MEDDOC_`."""

    # Absolute path: scripts work from any working directory, not only the repo root.
    model_config = SettingsConfigDict(env_prefix="MEDDOC_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    # Embedding model; its tokenizer also defines chunk length (bge-small truncates at 512 tokens).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge is trained asymmetrically: queries get this instruction, passages get none.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 32
    device: str = "cpu"

    # Chunking
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32
    chunk_strategy: str = "section"

    # Retrieval
    top_k: int = 5
    # Measured (eval/results/retrieval_20261015-202416): dense+rerank 95.8% Hit@5 vs hybrid+rerank 93.8%;
    # the difference is not significant (bootstrap CI includes 0), so the simpler pipeline is the default.
    retriever: str = "dense+rerank"
    hybrid_candidates: int = 50   # results taken from each retriever before fusion
    rrf_k: int = 60

    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30   # first-stage results rescored by the cross-encoder

    # Generation
    llm_provider: Literal["groq", "ollama"] = "groq"
    # Groq's model list changes; checked against GET /models on 2026-10-15
    # (llama-3.3-70b-versatile was no longer offered). The eval judge will use a different family.
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0      # deterministic answers -> reproducible evaluation
    llm_max_tokens: int = 1024        # reasoning models spend part of this on hidden reasoning tokens
    context_chunks: int = 5           # chunks shown to the LLM
    # Refuse without calling the LLM when the top reranker score (sigmoid) is below this.
    # Initial value from a 4-question preview (answerable ~0.9998, out-of-scope <=0.09); tuned in eval.
    refusal_threshold: float = 0.5
    # Standard env var name, read with or without the MEDDOC_ prefix. SecretStr hides it in logs/reprs.
    groq_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("GROQ_API_KEY", "MEDDOC_GROQ_API_KEY"))
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434"

    @property
    def manifest_path(self) -> Path:
        """YAML list of source documents (URL, checksum, license)."""
        return self.data_dir / "manifest.yaml"

    @property
    def raw_dir(self) -> Path:
        """Downloaded PDFs (gitignored)."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Chunks, indexes and caches (gitignored)."""
        return self.data_dir / "processed"

    def chunks_path(self, strategy: str) -> Path:
        return self.processed_dir / "chunks" / f"{strategy}.jsonl"

    def index_dir(self, strategy: str) -> Path:
        return self.processed_dir / "index" / strategy

    @property
    def embedding_cache_path(self) -> Path:
        return self.processed_dir / "embedding_cache.sqlite"

    @property
    def llm_cache_path(self) -> Path:
        return self.processed_dir / "llm_cache.sqlite"


settings = Settings()
