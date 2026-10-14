"""Parse every PDF in the manifest and chunk it with all three strategies.

Usage:
    python scripts/build_chunks.py

Writes data/processed/parsed/<doc_id>.json, data/processed/chunks/<strategy>.jsonl
and prints chunk-quality statistics per strategy.
"""

import json
import re
import statistics
import sys
import time

from meddoc_rag.config import settings
from meddoc_rag.ingestion.chunkers import STRATEGIES, HFTokenCounter
from meddoc_rag.ingestion.document import parse_document
from meddoc_rag.io import write_chunks
from meddoc_rag.manifest import load_manifest
from meddoc_rag.schemas import Chunk, ParsedDocument

# A chunk "ends cleanly" if it stops at a sentence end or a paragraph/line boundary.
CLEAN_END = re.compile(r"[.!?:;)\]\"']$")


def chunk_stats(chunks: list[Chunk], docs: dict[str, ParsedDocument], limit: int) -> dict[str, float]:
    """Structure-quality proxies we can measure before any retrieval eval."""
    tokens = sorted(c.n_tokens for c in chunks)
    mixed = 0
    for c in chunks:
        offsets = (h.offset for h in docs[c.doc_id].headings)
        mixed += any(c.start < o < c.end for o in offsets)
    clean_end = sum(bool(CLEAN_END.search(c.text)) or docs[c.doc_id].text[c.end:c.end + 2] == "\n\n" for c in chunks)
    return {
        "chunks": len(chunks),
        "tokens_total": sum(tokens),
        "tokens_mean": statistics.mean(tokens),
        "tokens_p50": tokens[len(tokens) // 2],
        "tokens_p95": tokens[int(0.95 * (len(tokens) - 1))],
        "over_model_limit_pct": 100 * sum(t > limit for t in tokens) / len(tokens),
        "clean_end_pct": 100 * clean_end / len(chunks),
        "crosses_heading_pct": 100 * mixed / len(chunks),
    }


def main() -> int:
    parsed_dir = settings.processed_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    docs: dict[str, ParsedDocument] = {}
    print(f"{'doc':32} {'pages':>5} {'chars':>8} {'headings':>8} source")
    for meta in load_manifest(settings.manifest_path):
        pdf = settings.raw_dir / meta.filename
        if not pdf.exists():
            print(f"missing {pdf}; run scripts/download_docs.py first")
            return 1
        doc = parse_document(meta, pdf)
        docs[doc.doc_id] = doc
        (parsed_dir / f"{doc.doc_id}.json").write_text(doc.model_dump_json(), encoding="utf-8")
        print(f"{doc.doc_id:32} {doc.num_pages:5} {len(doc.text):8} {len(doc.headings):8} {doc.heading_source}")

    counter = HFTokenCounter(settings.embedding_model)
    rows = {}
    for name, cls in STRATEGIES.items():
        chunker = cls(counter, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        t0 = time.perf_counter()
        chunks = [c for d in docs.values() for c in chunker.chunk(d)]
        elapsed = time.perf_counter() - t0
        write_chunks(settings.chunks_path(name), chunks)
        rows[name] = chunk_stats(chunks, docs, limit=512) | {"seconds": elapsed}

    keys = list(next(iter(rows.values())))
    print(f"\nChunking (max={settings.chunk_max_tokens}, overlap={settings.chunk_overlap_tokens} tokens, {settings.embedding_model} tokenizer)")
    print(f"{'metric':22}" + "".join(f"{n:>12}" for n in rows))
    for k in keys:
        print(f"{k:22}" + "".join(f"{rows[n][k]:>12.1f}" if isinstance(rows[n][k], float) else f"{rows[n][k]:>12}" for n in rows))
    (settings.processed_dir / "chunk_stats.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
