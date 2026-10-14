"""Offline tests for the manifest and file-verification helpers (no network)."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from meddoc_rag.config import settings
from meddoc_rag.manifest import SourceDocument, load_manifest, looks_like_pdf, sha256_of


def test_real_manifest_is_valid() -> None:
    docs = load_manifest(settings.manifest_path)
    assert 5 <= len(docs) <= 10
    assert all(d.license for d in docs), "every document must declare its license"


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    entry = """
  - id: a
    title: t
    publisher: p
    year: 2020
    license: l
    source_page: https://example.org
    url: https://example.org/a.pdf
    filename: a.pdf
"""
    manifest = tmp_path / "m.yaml"
    manifest.write_text("documents:" + entry + entry, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate id"):
        load_manifest(manifest)


def test_bad_checksum_format_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceDocument(
            id="a", title="t", publisher="p", year=2020, license="l",
            source_page="s", url="u", filename="a.pdf", sha256="not-a-hash",
        )


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    payload = b"hello world" * 100_000  # > one read chunk
    f.write_bytes(payload)
    assert sha256_of(f, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


def test_html_error_page_is_not_a_pdf(tmp_path: Path) -> None:
    html = tmp_path / "fake.pdf"
    html.write_bytes(b"<HTML><HEAD><TITLE>Access Denied</TITLE>")
    real = tmp_path / "real.pdf"
    real.write_bytes(b"%PDF-1.7\n...")
    assert not looks_like_pdf(html)
    assert looks_like_pdf(real)
