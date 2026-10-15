"""Ask a question from the command line.

Usage:
    python scripts/ask.py "What blood pressure threshold should trigger drug treatment?"
    python scripts/ask.py "..." --retrieve-only            # show retrieved chunks, no LLM call
    python scripts/ask.py "..." --retriever hybrid --strategy fixed --k 3
"""

import argparse
import sys
import textwrap
import time

from meddoc_rag.config import settings
from meddoc_rag.ingestion.chunkers import STRATEGIES
from meddoc_rag.pipeline import RETRIEVERS, load_retriever, make_answerer


def show_chunks(question: str, retriever_name: str, strategy: str, k: int) -> None:
    retriever = load_retriever(retriever_name, strategy)
    retriever.retrieve("warm-up", 1)  # load model weights outside the timed call
    t0 = time.perf_counter()
    results = retriever.retrieve(question, k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\nQ: {question}\n[{retriever.name}/{strategy}, top-{k}, {ms:.0f} ms]\n")
    for r in results:
        c = r.chunk
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
        via = "  " + " ".join(f"{k}#{v}" for k, v in r.component_ranks.items()) if r.component_ranks else ""
        print(f"#{r.rank}  score={r.score:.4f}  {c.doc_id}  {pages}{via}")
        if c.section:
            print(f"    section: {c.section[:110]}")
        print(textwrap.indent(textwrap.fill(c.text[:400].replace("\n", " "), 100), "    ") + "\n")


def show_answer(question: str, retriever_name: str, strategy: str) -> None:
    answerer = make_answerer(retriever_name, strategy)
    answerer.retriever.retrieve("warm-up", 1)  # load embedding/reranker weights outside the timed call
    ans = answerer.answer(question)
    print(f"\nQ: {question}\n")
    print(textwrap.fill(ans.answer, 100))
    if ans.refused:
        print(f"\n[refused: {ans.refusal_reason}; top retrieval score {ans.top_score:.4f}]" if ans.top_score is not None
              else f"\n[refused: {ans.refusal_reason}]")
    for c in ans.citations:
        print(f"  [{c.number}] {c.label}  (score {c.score:.3f})")
    if ans.invalid_citations:
        print(f"  ! removed citations to sources that were never provided: {ans.invalid_citations}")
    t = ans.latency_ms
    print(f"\n[{ans.model or 'no LLM call'} | retrieval {t.get('retrieval', 0):.0f} ms, "
          f"generation {t.get('generation', 0):.0f} ms, total {t.get('total', 0):.0f} ms]")
    print(f"\n{ans.disclaimer}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=settings.chunk_strategy)
    parser.add_argument("--retriever", choices=RETRIEVERS, default=settings.retriever)
    parser.add_argument("--k", type=int, default=settings.top_k, help="chunks to show with --retrieve-only")
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM; print retrieved chunks")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    if args.retrieve_only:
        show_chunks(args.question, args.retriever, args.strategy, args.k)
    else:
        show_answer(args.question, args.retriever, args.strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
