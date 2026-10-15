"""Central configuration, read from environment variables / a `.env` file.

Every tunable value lives here so that experiments are driven by config,
not by editing code. Later milestones add model names, top-k, thresholds, etc.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings. Override any field with an env var prefixed `MEDDOC_`."""

    model_config = SettingsConfigDict(env_prefix="MEDDOC_", env_file=".env", extra="ignore")

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
    retriever: str = "hybrid"
    hybrid_candidates: int = 50   # results taken from each retriever before fusion
    rrf_k: int = 60

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


settings = Settings()
