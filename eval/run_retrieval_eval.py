"""Retrieval evaluation: every chunking strategy x retriever on the golden set.

Usage:
    python eval/run_retrieval_eval.py

Writes eval/results/retrieval_<timestamp>.{json,md}. Only answerable questions
are used (retrieval quality is undefined for out-of-scope questions).
"""

import json
import statistics
import sys
import time
from datetime import datetime

from meddoc_rag.config import PROJECT_ROOT, settings
from meddoc_rag.evaluation.golden import load_golden, load_parsed_docs, resolve
from meddoc_rag.evaluation.metrics import first_relevant_rank, hit_at_k, mrr, wrong_doc_at_1
from meddoc_rag.indexing.embedder import SentenceTransformerEmbedder
from meddoc_rag.ingestion.chunkers import STRATEGIES
from meddoc_rag.pipeline import load_retriever, make_embedder, make_reranker

RETRIEVERS = ["dense", "bm25", "hybrid", "dense+rerank", "hybrid+rerank"]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def evaluate(retriever, items, gold, k: int) -> dict:
    ranks, tops, times = [], [], []
    per_question = []
    for item in items:
        t0 = time.perf_counter()
        results = retriever.retrieve(item.question, k)
        times.append((time.perf_counter() - t0) * 1000)
        chunks = [r.chunk for r in results]
        rank = first_relevant_rank(chunks, gold[item.id])
        ranks.append(rank)
        tops.append(chunks[0])
        per_question.append({"id": item.id, "type": item.type, "rank": rank,
                             "top1": f"{chunks[0].doc_id} p.{chunks[0].page_start}"})
    by_type = {}
    for t in sorted({i.type for i in items}):
        sub = [r for r, i in zip(ranks, items) if i.type == t]
        by_type[t] = {"n": len(sub), "hit@5": hit_at_k(sub, 5)}
    return {
        "hit@1": hit_at_k(ranks, 1), "hit@3": hit_at_k(ranks, 3), "hit@5": hit_at_k(ranks, 5),
        "hit@10": hit_at_k(ranks, 10), "hit@30": hit_at_k(ranks, 30) if k >= 30 else None,
        "mrr@10": mrr(ranks, 10), "wrong_doc@1": wrong_doc_at_1(tops, [gold[i.id] for i in items]),
        "latency_ms_p50": statistics.median(times), "by_type": by_type, "per_question": per_question,
    }


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items = [i for i in load_golden(PROJECT_ROOT / "eval" / "golden_set.jsonl") if i.answerable]
    docs = load_parsed_docs(settings.processed_dir / "parsed")
    gold = {i.id: resolve(i, docs) for i in items}

    embedder = make_embedder(use_cache=False)
    reranker = make_reranker()
    no_instruction = SentenceTransformerEmbedder(settings.embedding_model, query_instruction="")
    no_instruction._model = embedder.model  # same weights, only the query prefix differs

    configs = [(s, r) for s in STRATEGIES for r in RETRIEVERS] + [("section", "dense (no query instruction)")]
    rows = []
    for strategy, name in configs:
        if name.startswith("dense (no"):
            retriever = load_retriever("dense", strategy, no_instruction)
        else:
            retriever = load_retriever(name, strategy, embedder, reranker=reranker)
        retriever.retrieve("warm-up", 1)
        k = 10 if name.endswith("+rerank") else 30
        res = evaluate(retriever, items, gold, k)
        rows.append({"strategy": strategy, "retriever": name, **res})
        print(f"{strategy:9} {name:30} Hit@1 {pct(res['hit@1']):>5}  Hit@5 {pct(res['hit@5']):>5}  "
              f"MRR {res['mrr@10']:.3f}  Hit@30 {pct(res['hit@30']):>5}  p50 {res['latency_ms_p50']:.0f} ms")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"timestamp": stamp, "n_questions": len(items), "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model, "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens, "rerank_candidates": settings.rerank_candidates,
            "relevance": "chunk covers >=50% of a gold quote's characters"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"retrieval_{stamp}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")

    md = [f"# Retrieval evaluation ({stamp})", "",
          f"{len(items)} answerable golden questions · relevance = {meta['relevance']} · "
          f"chunks {settings.chunk_max_tokens} tokens / {settings.chunk_overlap_tokens} overlap · "
          f"rerank candidates {settings.rerank_candidates}", "",
          "| Chunking | Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Hit@30 | Wrong doc @1 | p50 latency |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['strategy']} | {r['retriever']} | {pct(r['hit@1'])} | {pct(r['hit@3'])} | {pct(r['hit@5'])} | "
                  f"{r['mrr@10']:.3f} | {pct(r['hit@30'])} | {pct(r['wrong_doc@1'])} | {r['latency_ms_p50']:.0f} ms |")
    types = sorted(rows[0]["by_type"])
    md += ["", "## Hit@5 by question type (section-aware chunking)", "",
           "| Retriever | " + " | ".join(f"{t} (n={rows[0]['by_type'][t]['n']})" for t in types) + " |",
           "|---|" + "---|" * len(types)]
    for r in rows:
        if r["strategy"] == "section":
            md.append(f"| {r['retriever']} | " + " | ".join(pct(r["by_type"][t]["hit@5"]) for t in types) + " |")
    (RESULTS_DIR / f"retrieval_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote eval/results/retrieval_{stamp}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
