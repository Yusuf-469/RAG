"""Ask a question from the command line and print the retrieved chunks.

Usage:
    python scripts/ask.py "What blood pressure threshold should trigger drug treatment?"
    python scripts/ask.py "..." --strategy fixed --k 3
"""

import argparse
import sys
import textwrap
import time

from meddoc_rag.config import settings
from meddoc_rag.ingestion.chunkers import STRATEGIES
from meddoc_rag.pipeline import load_dense_retriever, make_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=settings.chunk_strategy)
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    retriever = load_dense_retriever(args.strategy, make_embedder(use_cache=False))
    retriever.retrieve("warm-up", 1)  # load model weights outside the timed call

    t0 = time.perf_counter()
    results = retriever.retrieve(args.question, args.k)
    ms = (time.perf_counter() - t0) * 1000

    print(f"\nQ: {args.question}\n[{retriever.name}/{args.strategy}, top-{args.k}, {ms:.0f} ms]\n")
    for r in results:
        c = r.chunk
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        print(f"#{r.rank}  score={r.score:.3f}  {c.doc_id}  {pages}")
        if c.section:
            print(f"    section: {c.section[:110]}")
        print(textwrap.indent(textwrap.fill(c.text[:400].replace("\n", " "), 100), "    ") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
