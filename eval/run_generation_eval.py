"""End-to-end evaluation: refusal-threshold sweep, answer generation, LLM-as-judge.

Usage:
    python eval/run_generation_eval.py                 # configured refusal_threshold (default)
    python eval/run_generation_eval.py --threshold 0.3 # explicit threshold
    python eval/run_generation_eval.py --tune          # pick the threshold from the sweep (overfits!)

The sweep is always reported. Choosing the threshold from it uses the same questions the
results are reported on, so `--tune` numbers are optimistic: a 0.95 threshold "achieved" 100%
refusal by a 0.001 margin on one question while refusing 3 answerable questions.

Steps:
1. Top reranker score for all golden questions -> sweep thresholds, report
   answer rate on answerable vs refusal rate on out-of-scope questions.
2. Answer every question with the chosen threshold (below it: no LLM call).
3. Judge answered questions: faithfulness to sources, correctness vs reference.

LLM responses are cached, so re-runs cost no API quota.
Writes eval/results/generation_<timestamp>.{json,md}.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime

from meddoc_rag.config import PROJECT_ROOT, settings
from meddoc_rag.evaluation.golden import load_golden
from meddoc_rag.evaluation.judge import Judge
from meddoc_rag.generation.answerer import Answerer
from meddoc_rag.generation.llm import CachedLLM, GroqClient
from meddoc_rag.pipeline import load_retriever, make_llm

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def sweep(scores: dict[str, float], answerable: dict[str, bool]) -> list[dict]:
    rows = []
    pos = [q for q, a in answerable.items() if a]
    neg = [q for q, a in answerable.items() if not a]
    for t in THRESHOLDS:
        answered = sum(scores[q] >= t for q in pos) / len(pos)
        refused = sum(scores[q] < t for q in neg) / len(neg)
        rows.append({"threshold": t, "answer_rate_answerable": answered, "refusal_rate_oos": refused,
                     "balanced_accuracy": (answered + refused) / 2})
    return rows


def choose_threshold(rows: list[dict]) -> float:
    """Best balanced accuracy; among ties take the middle threshold (largest margin both ways)."""
    best = max(r["balanced_accuracy"] for r in rows)
    tied = [r["threshold"] for r in rows if abs(r["balanced_accuracy"] - best) < 1e-9]
    return tied[len(tied) // 2]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tune", action="store_true", help="choose threshold from the sweep (optimistic)")
    args = parser.parse_args()

    items = load_golden(PROJECT_ROOT / "eval" / "golden_set.jsonl")
    retriever = load_retriever(settings.retriever, settings.chunk_strategy)
    retriever.retrieve("warm-up", 1)

    # 1) threshold sweep on top reranker scores
    scores = {i.id: retriever.retrieve(i.question, 1)[0].score for i in items}
    sweep_rows = sweep(scores, {i.id: i.answerable for i in items})
    if args.tune:
        threshold = choose_threshold(sweep_rows)
        print(f"WARNING: threshold {threshold} tuned on the evaluation set itself; results are optimistic")
    else:
        threshold = args.threshold if args.threshold is not None else settings.refusal_threshold
    print("threshold  answered(answerable)  refused(out-of-scope)  balanced")
    for r in sweep_rows:
        mark = "  <- chosen" if r["threshold"] == threshold else ""
        print(f"{r['threshold']:9}  {100 * r['answer_rate_answerable']:19.1f}  {100 * r['refusal_rate_oos']:21.1f}  "
              f"{100 * r['balanced_accuracy']:8.1f}{mark}")

    # 2) answers + 3) judgments
    answerer = Answerer(retriever, make_llm(), settings.context_chunks, threshold,
                        settings.llm_temperature, settings.llm_max_tokens)
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    judge = Judge(CachedLLM(GroqClient(key, settings.judge_model, settings.groq_base_url), settings.llm_cache_path))

    records = []
    for n, item in enumerate(items, start=1):
        ans = answerer.answer(item.question)
        rec = {"id": item.id, "type": item.type, "answerable": item.answerable, "question": item.question,
               "top_score": ans.top_score, "refused": ans.refused, "refusal_reason": ans.refusal_reason,
               "answer": ans.answer, "citations": [c.label for c in ans.citations],
               "invalid_citations": ans.invalid_citations, "latency_ms": ans.latency_ms,
               "reference_answer": item.reference_answer}
        if not ans.refused:
            f = judge.faithfulness(ans.answer, ans.sources)
            rec["faithfulness"] = f.score
            rec["unsupported_claims"] = [c.get("claim") for c in f.claims if not c.get("supported")]
            rec["faithfulness_parse_error"] = f.parse_error
            if item.answerable:
                c = judge.correctness(item.question, item.reference_answer or "", ans.answer)
                rec["correctness"] = c.verdict
                rec["correctness_reason"] = c.reason
        records.append(rec)
        status = "REFUSED " + (ans.refusal_reason or "") if ans.refused else f"faith={rec.get('faithfulness')} {rec.get('correctness', '')}"
        print(f"[{n:2}/{len(items)}] {item.id:8} top={ans.top_score:.3f} {status}")

    ans_items = [r for r in records if r["answerable"]]
    oos_items = [r for r in records if not r["answerable"]]
    answered = [r for r in records if not r["refused"]]
    faith = [r["faithfulness"] for r in answered if r.get("faithfulness") is not None]
    verdicts = [r.get("correctness") for r in ans_items if not r["refused"]]
    gen_ms = [r["latency_ms"]["generation"] for r in answered if r["latency_ms"].get("generation")]
    summary = {
        "threshold": threshold,
        "n_answerable": len(ans_items), "n_out_of_scope": len(oos_items),
        "answer_rate_answerable": sum(not r["refused"] for r in ans_items) / len(ans_items),
        "refusal_rate_out_of_scope": sum(r["refused"] for r in oos_items) / len(oos_items),
        "correct_rate_answerable": sum(v == "correct" for v in verdicts) / len(ans_items),
        "correct_or_partial_rate_answerable": sum(v in ("correct", "partial") for v in verdicts) / len(ans_items),
        "incorrect_count": sum(v == "incorrect" for v in verdicts),
        "faithfulness_mean": statistics.mean(faith) if faith else None,
        "fully_faithful_rate": sum(f == 1.0 for f in faith) / len(faith) if faith else None,
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in answered),
        "judge_parse_errors": sum(bool(r.get("faithfulness_parse_error")) for r in answered),
        "generation_ms_p50": statistics.median(gen_ms) if gen_ms else None,
        "refusal_reasons": {k: sum(r["refusal_reason"] == k for r in records if r["refused"])
                            for k in ("low_retrieval_confidence", "llm_insufficient_context", "no_valid_citations")},
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"timestamp": stamp, "threshold_tuned_on_eval_set": args.tune, "retriever": settings.retriever, "chunking": settings.chunk_strategy,
            "generator": settings.llm_model, "judge": settings.judge_model, "context_chunks": settings.context_chunks}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"generation_{stamp}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "sweep": sweep_rows, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    p = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    md = [f"# Generation evaluation ({stamp})", "",
          f"Retriever `{settings.retriever}` on `{settings.chunk_strategy}` chunks · generator `{settings.llm_model}` · "
          f"judge `{settings.judge_model}` · refusal threshold {threshold}", "",
          "| Metric | Value |", "|---|---|",
          f"| Answerable questions answered | {p(summary['answer_rate_answerable'])} ({len(ans_items)} questions) |",
          f"| Out-of-scope questions refused | {p(summary['refusal_rate_out_of_scope'])} ({len(oos_items)} questions) |",
          f"| Correct (judge) among all answerable | {p(summary['correct_rate_answerable'])} |",
          f"| Correct or partial among all answerable | {p(summary['correct_or_partial_rate_answerable'])} |",
          f"| Incorrect answers | {summary['incorrect_count']} |",
          f"| Mean faithfulness (supported claims) | {p(summary['faithfulness_mean'])} |",
          f"| Answers with every claim supported | {p(summary['fully_faithful_rate'])} |",
          f"| Answers that cited a non-provided source | {summary['answers_with_invalid_citations']} |",
          f"| Generation latency p50 | {summary['generation_ms_p50']:.0f} ms |" if summary["generation_ms_p50"] else "| Generation latency p50 | – |",
          "", "## Refusal threshold sweep (top reranker score)", "",
          "| Threshold | Answerable answered | Out-of-scope refused | Balanced accuracy |", "|---|---|---|---|"]
    md += [f"| {r['threshold']} | {p(r['answer_rate_answerable'])} | {p(r['refusal_rate_oos'])} | {p(r['balanced_accuracy'])} |"
           for r in sweep_rows]
    md += ["", "## Per question", "", "| id | type | top score | outcome | faithfulness | correctness |", "|---|---|---|---|---|---|"]
    for r in records:
        outcome = f"refused ({r['refusal_reason']})" if r["refused"] else "answered"
        md.append(f"| {r['id']} | {r['type']} | {r['top_score']:.3f} | {outcome} | {p(r.get('faithfulness'))} | {r.get('correctness') or '–'} |")
    (RESULTS_DIR / f"generation_{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote eval/results/generation_{stamp}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
