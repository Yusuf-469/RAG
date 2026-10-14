"""Embed chunks and build a FAISS index per chunking strategy.

Usage:
    python scripts/build_index.py                      # all strategies
    python scripts/build_index.py --strategy section

Embeddings are cached in data/processed/embedding_cache.sqlite, so re-runs
only embed chunks whose text changed.
"""

import argparse
import logging
import sys
import time

from meddoc_rag.config import settings
from meddoc_rag.indexing.dense_index import DenseIndex
from meddoc_rag.ingestion.chunkers import STRATEGIES
from meddoc_rag.io import read_chunks
from meddoc_rag.pipeline import make_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("meddoc_rag").setLevel(logging.INFO)

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    embedder = make_embedder()
    for strategy in strategies:
        path = settings.chunks_path(strategy)
        if not path.exists():
            print(f"missing {path}; run scripts/build_chunks.py first")
            return 1
        chunks = read_chunks(path)
        t0 = time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in chunks])
        elapsed = time.perf_counter() - t0
        index = DenseIndex(chunks, vectors, embedder.model_name)
        index.save(settings.index_dir(strategy))
        print(f"{strategy:10} {len(chunks):5} chunks  dim={index.dim}  embed {elapsed:6.1f}s ({len(chunks) / elapsed:5.1f} chunks/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
