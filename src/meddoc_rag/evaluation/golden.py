"""Golden question set with strategy-independent gold labels.

Gold evidence is an exact quote from the source document, not a chunk id:
chunk ids differ between chunking strategies, quotes don't. Each quote is
located in the parsed document text (ignoring whitespace differences such as
"150– 300" vs "150–300"); every occurrence counts as gold, since many
recommendations appear in both the executive summary and the main body.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from meddoc_rag.schemas import ParsedDocument

QuestionType = Literal["fact", "number", "paraphrase", "confusable", "regression", "out_of_scope"]


class Evidence(BaseModel):
    doc_id: str
    quote: str
    # True when added after inspecting retrieved results (pooled relevance judgment):
    # the passage was missing from the original labels but genuinely answers the question.
    pooled: bool = False


class GoldenItem(BaseModel):
    id: str
    type: QuestionType
    answerable: bool
    question: str
    evidence: list[Evidence] = []
    reference_answer: str | None = None
    notes: str | None = None


class GoldSpan(BaseModel):
    doc_id: str
    start: int
    end: int
    page_start: int
    page_end: int


def load_golden(path: Path) -> list[GoldenItem]:
    items = [GoldenItem.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden item ids")
    for item in items:
        if item.answerable != bool(item.evidence):
            raise ValueError(f"{item.id}: answerable items need evidence, unanswerable items must have none")
    return items


@lru_cache(maxsize=None)
def _compact(text: str) -> tuple[str, tuple[int, ...]]:
    """Text with all whitespace removed, plus a map from compact index to original index."""
    keep = [i for i, ch in enumerate(text) if not ch.isspace()]
    return "".join(text[i] for i in keep), tuple(keep)


def find_quote(doc: ParsedDocument, quote: str) -> list[tuple[int, int]]:
    """All (start, end) original offsets where `quote` occurs, whitespace-insensitive, case-sensitive."""
    compact, index = _compact(doc.text)
    needle = re.sub(r"\s+", "", quote)
    spans, pos = [], compact.find(needle)
    while pos != -1:
        spans.append((index[pos], index[pos + len(needle) - 1] + 1))
        pos = compact.find(needle, pos + 1)
    return spans


def resolve(item: GoldenItem, docs: dict[str, ParsedDocument]) -> list[GoldSpan]:
    """Locate every evidence quote; raise if a quote is missing or only exists in excluded text."""
    spans: list[GoldSpan] = []
    for ev in item.evidence:
        doc = docs[ev.doc_id]
        found = [(s, e) for s, e in find_quote(doc, ev.quote) if not any(a <= s < b for a, b in doc.excluded)]
        if not found:
            raise ValueError(f"{item.id}: quote not found in {ev.doc_id}: {ev.quote[:70]!r}")
        spans.extend(GoldSpan(doc_id=ev.doc_id, start=s, end=e, page_start=doc.page_at(s), page_end=doc.page_at(e - 1))
                     for s, e in found)
    return spans


def load_parsed_docs(parsed_dir: Path) -> dict[str, ParsedDocument]:
    return {
        p.stem: ParsedDocument.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(parsed_dir.glob("*.json"))
    }


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
