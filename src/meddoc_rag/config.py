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

    # Chunking
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32

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


settings = Settings()
