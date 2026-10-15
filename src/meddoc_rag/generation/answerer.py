"""Retrieve -> gate on confidence -> generate -> verify citations.

Three layers keep answers grounded:
1. Refuse before calling the LLM when the best retrieval score is below a threshold.
2. A prompt that allows only numbered sources and a fixed refusal sentence.
3. Code-side verification: citations must refer to sources actually shown to the
   model; an answer with no valid citation is treated as unsupported and refused.
"""

import re
import time
import unicodedata

from pydantic import BaseModel

from meddoc_rag.generation.llm import LLMClient
from meddoc_rag.generation.prompts import REFUSAL, build_messages
from meddoc_rag.retrieval.base import Retriever
from meddoc_rag.schemas import RetrievedChunk

DISCLAIMER = (
    "For educational purposes only. Not medical advice. "
    "Consult a qualified healthcare professional and the original guideline."
)
_CITATION = re.compile(r"\[(\d+)\]")
# gpt-oss cites as 【1】 or 【1†L3-L5】; others write [1, 2] or [1-3]. All become [1][2]...
_LENTICULAR = re.compile(r"【\s*([\d\s,–-]+?)\s*(?:†[^】]*)?】")
_GROUPED = re.compile(r"\[\s*(\d+(?:\s*[,–-]\s*\d+)+)\s*\]")


class Citation(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section: str | None
    score: float

    @property
    def label(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"{self.doc_title}, {pages}"


class Answer(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None   # low_retrieval_confidence | llm_insufficient_context | no_valid_citations
    citations: list[Citation] = []
    invalid_citations: list[int] = []   # numbers the model cited that it was never given
    sources: list[RetrievedChunk] = []
    top_score: float | None = None
    model: str | None = None
    latency_ms: dict[str, float] = {}
    disclaimer: str = DISCLAIMER


def _expand(group: str) -> str:
    """'1, 3' -> '[1][3]'; '2-4' -> '[2][3][4]' (ranges capped to avoid abuse)."""
    nums: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        if m := re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part):
            lo, hi = int(m.group(1)), int(m.group(2))
            nums.extend(range(lo, hi + 1) if 0 <= hi - lo <= 20 else [lo, hi])
        elif part.isdigit():
            nums.append(int(part))
    return "".join(f"[{n}]" for n in nums)


def normalize_answer(text: str) -> str:
    """NFKC (narrow no-break spaces, full-width ［1］ -> [1]) and one canonical citation style."""
    text = unicodedata.normalize("NFKC", text)
    text = _LENTICULAR.sub(lambda m: _expand(m.group(1)), text)
    return _GROUPED.sub(lambda m: _expand(m.group(1)), text)


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) cited source numbers in first-seen order, without duplicates."""
    seen = list(dict.fromkeys(int(m) for m in _CITATION.findall(text)))
    return [n for n in seen if 1 <= n <= n_sources], [n for n in seen if not 1 <= n <= n_sources]


def strip_invalid_citations(text: str, invalid: list[int]) -> str:
    for n in invalid:
        text = text.replace(f"[{n}]", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(REFUSAL.lower().rstrip("."))


class Answerer:
    """End-to-end question answering over a retriever and an LLM."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        context_chunks: int = 5,
        refusal_threshold: float | None = 0.5,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.context_chunks = context_chunks
        # Only meaningful for calibrated scores (reranker sigmoid); None disables the gate.
        self.refusal_threshold = refusal_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _refuse(self, question: str, reason: str, sources: list[RetrievedChunk], timings: dict, top: float | None, model: str | None = None) -> Answer:
        return Answer(question=question, answer=REFUSAL, refused=True, refusal_reason=reason,
                      sources=sources, top_score=top, model=model, latency_ms=timings)

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        sources = self.retriever.retrieve(question, self.context_chunks)
        timings = {"retrieval": (time.perf_counter() - t0) * 1000}
        top = sources[0].score if sources else None

        if not sources or (self.refusal_threshold is not None and top < self.refusal_threshold):
            timings["total"] = (time.perf_counter() - t0) * 1000
            return self._refuse(question, "low_retrieval_confidence", sources, timings, top)

        resp = self.llm.generate(build_messages(question, sources), self.temperature, self.max_tokens)
        timings["generation"] = resp.latency_ms
        timings["total"] = (time.perf_counter() - t0) * 1000

        text = normalize_answer(resp.text)
        if is_refusal(text):
            return self._refuse(question, "llm_insufficient_context", sources, timings, top, resp.model)

        valid, invalid = parse_citations(text, len(sources))
        if not valid:
            return self._refuse(question, "no_valid_citations", sources, timings, top, resp.model)

        citations = []
        for n in valid:
            r = sources[n - 1]
            c = r.chunk
            citations.append(Citation(number=n, chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title,
                                      page_start=c.page_start, page_end=c.page_end, section=c.section, score=r.score))
        return Answer(
            question=question,
            answer=strip_invalid_citations(text, invalid),
            refused=False,
            citations=citations,
            invalid_citations=invalid,
            sources=sources,
            top_score=top,
            model=resp.model,
            latency_ms=timings,
        )
