"""Check that every golden-set quote exists in its document, and print a review table.

Usage:
    python eval/verify_golden.py            # fails (exit 1) if any quote is missing
"""

import sys
from collections import Counter

from meddoc_rag.config import PROJECT_ROOT, settings
from meddoc_rag.evaluation.golden import load_golden, load_parsed_docs, resolve


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items = load_golden(PROJECT_ROOT / "eval" / "golden_set.jsonl")
    docs = load_parsed_docs(settings.processed_dir / "parsed")
    errors = 0
    for item in items:
        if not item.answerable:
            print(f"{item.id:8} [out_of_scope] {item.question}")
            continue
        try:
            spans = resolve(item, docs)
        except ValueError as exc:
            print(f"{item.id:8} MISSING  {exc}")
            errors += 1
            continue
        where = ", ".join(sorted({f"{s.doc_id} p.{s.page_start}" for s in spans}))
        print(f"{item.id:8} [{item.type}] {item.question}\n          gold: {where}")
    types = Counter(i.type for i in items)
    print(f"\n{len(items)} items ({sum(i.answerable for i in items)} answerable): {dict(types)}")
    print("all quotes verified" if not errors else f"{errors} quote(s) missing")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
