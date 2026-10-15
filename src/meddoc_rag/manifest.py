"""Typed access to `data/manifest.yaml`, the list of source documents."""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PDF_MAGIC = b"%PDF-"


class SourceDocument(BaseModel):
    """One public guideline PDF in the corpus."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    publisher: str
    year: int
    license: str
    source_page: str
    url: str
    filename: str = Field(pattern=r"^[a-z0-9_]+\.pdf$")
    manual: bool = False
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Regexes matched against heading titles; matching sections (e.g. quizzes) are never indexed.
    exclude_sections: list[str] = []


def load_manifest(path: Path) -> list[SourceDocument]:
    """Parse and validate the manifest; raises if ids or filenames repeat."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    docs = [SourceDocument(**entry) for entry in raw["documents"]]
    for field in ("id", "filename"):
        values = [getattr(d, field) for d in docs]
        dupes = {v for v in values if values.count(v) > 1}
        if dupes:
            raise ValueError(f"Duplicate {field} in manifest: {sorted(dupes)}")
    return docs


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 (constant memory, works for large PDFs)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def looks_like_pdf(path: Path) -> bool:
    """True if the file starts with the PDF magic bytes.

    Catches servers that return an HTML page (login wall, JS app, 'Access
    Denied') with a 200 status instead of the PDF.
    """
    with path.open("rb") as f:
        return f.read(len(PDF_MAGIC)) == PDF_MAGIC
