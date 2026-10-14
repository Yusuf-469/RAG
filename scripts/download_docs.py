"""Download the corpus PDFs listed in data/manifest.yaml and verify them.

Usage:
    python scripts/download_docs.py            # download missing files, verify all
    python scripts/download_docs.py --force    # re-download everything

Exit code is non-zero if any document is missing, is not a PDF, or fails its
checksum, so CI can rely on it.
"""

import argparse
import sys
from pathlib import Path

import requests

from meddoc_rag.config import settings
from meddoc_rag.manifest import SourceDocument, load_manifest, looks_like_pdf, sha256_of

USER_AGENT = "meddoc-rag/0.1 (educational research project)"
TIMEOUT_S = 120


def download(doc: SourceDocument, dest: Path) -> None:
    """Stream `doc.url` to `dest` via a temp file, so a failed download never leaves a partial PDF."""
    tmp = dest.with_suffix(".part")
    with requests.get(doc.url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_content(chunk_size=1 << 16):
                f.write(block)
    tmp.replace(dest)


def verify(doc: SourceDocument, path: Path) -> bool:
    """Check PDF magic bytes and (if pinned) the SHA-256. Prints the outcome."""
    if not looks_like_pdf(path):
        print(f"  FAIL  {doc.id}: not a PDF (server probably returned an HTML page)")
        return False
    actual = sha256_of(path)
    if doc.sha256 is None:
        print(f"  WARN  {doc.id}: checksum not pinned yet -> sha256: {actual}")
        return True
    if actual != doc.sha256:
        print(f"  FAIL  {doc.id}: checksum mismatch (publisher may have changed the file)")
        print(f"        expected {doc.sha256}\n        actual   {actual}")
        return False
    print(f"  OK    {doc.id}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    docs = load_manifest(settings.manifest_path)
    ok = True

    for doc in docs:
        dest = settings.raw_dir / doc.filename
        if dest.exists() and not args.force:
            pass
        elif doc.manual:
            print(f"  TODO  {doc.id}: publisher blocks scripted downloads.")
            print(f"        Open {doc.source_page} in a browser and save the PDF as {dest}")
            ok = False
            continue
        else:
            print(f"  GET   {doc.id}")
            try:
                download(doc, dest)
            except requests.RequestException as exc:
                print(f"  FAIL  {doc.id}: {exc}")
                ok = False
                continue
        ok = verify(doc, dest) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
