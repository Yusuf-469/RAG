"""FastAPI service exposing the question-answering pipeline.

Run:
    uvicorn app.api:app --port 8000
    # interactive docs: http://127.0.0.1:8000/docs

Models are loaded once at startup (lifespan), not per request.
"""

import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from meddoc_rag.config import settings
from meddoc_rag.generation.answerer import Answer, Answerer
from meddoc_rag.generation.highlight import highlight
from meddoc_rag.generation.llm import LLMError


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["How much physical activity should older adults do?"])


class CitationOut(BaseModel):
    number: int
    label: str
    doc_id: str
    page_start: int
    page_end: int
    section: str | None
    score: float


class SourceOut(BaseModel):
    number: int                                  # the [n] the model saw
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section: str | None
    score: float                                 # reranker sigmoid score
    cited: bool
    text: str
    highlights: list[tuple[int, int]] = []       # approximate supporting sentences (cited sources only)


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    confidence: float | None                     # top retrieval score in (0, 1)
    citations: list[CitationOut]
    invalid_citations: list[int]
    sources: list[SourceOut]
    model: str | None
    latency_ms: dict[str, float]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    retriever: str
    chunk_strategy: str
    generator: str
    refusal_threshold: float | None
    indexed_chunks: int | None


def to_response(ans: Answer, server_ms: float) -> AskResponse:
    cited = {c.number for c in ans.citations}
    sources = [
        SourceOut(
            number=i, chunk_id=r.chunk.chunk_id, doc_id=r.chunk.doc_id, doc_title=r.chunk.doc_title,
            page_start=r.chunk.page_start, page_end=r.chunk.page_end, section=r.chunk.section,
            score=r.score, cited=i in cited, text=r.chunk.text,
            highlights=highlight(r.chunk.text, ans.answer) if i in cited else [],
        )
        for i, r in enumerate(ans.sources, start=1)
    ]
    return AskResponse(
        question=ans.question, answer=ans.answer, refused=ans.refused, refusal_reason=ans.refusal_reason,
        confidence=ans.top_score,
        citations=[CitationOut(number=c.number, label=c.label, doc_id=c.doc_id, page_start=c.page_start,
                               page_end=c.page_end, section=c.section, score=c.score) for c in ans.citations],
        invalid_citations=ans.invalid_citations, sources=sources, model=ans.model,
        latency_ms={**ans.latency_ms, "server_total": server_ms}, disclaimer=ans.disclaimer,
    )


def _default_answerer() -> Answerer:
    from meddoc_rag.pipeline import make_answerer

    answerer = make_answerer()
    answerer.retriever.retrieve("warm-up", 1)  # load embedding and reranker weights now, not on the first request
    return answerer


def _index_size(answerer: Answerer) -> int | None:
    base = getattr(answerer.retriever, "base", answerer.retriever)
    index = getattr(base, "index", None)
    return len(index.chunks) if index is not None and hasattr(index, "chunks") else None


def create_app(answerer_factory: Callable[[], Answerer] = _default_answerer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.answerer = answerer_factory()
        yield

    app = FastAPI(
        title="MedDoc RAG",
        description="Citation-grounded answers from public WHO/CDC/NIH guidelines. Educational use only; not medical advice.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        a: Answerer = request.app.state.answerer
        return HealthResponse(status="ok", retriever=a.retriever.name, chunk_strategy=settings.chunk_strategy,
                              generator=getattr(a.llm, "model", "unknown"), refusal_threshold=a.refusal_threshold,
                              indexed_chunks=_index_size(a))

    # Plain `def`: FastAPI runs it in a worker thread, so the CPU-bound model calls don't block the event loop.
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        t0 = time.perf_counter()
        try:
            ans = request.app.state.answerer.answer(body.question.strip())
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=f"Language model unavailable: {exc}") from exc
        return to_response(ans, (time.perf_counter() - t0) * 1000)

    return app


app = create_app()
