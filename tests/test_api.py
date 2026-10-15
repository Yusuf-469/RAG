"""API tests with a fake answerer: no models, no network, no API key."""

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from meddoc_rag.generation.answerer import Answer, Citation
from meddoc_rag.generation.highlight import highlight, sentence_spans
from meddoc_rag.generation.llm import LLMError
from meddoc_rag.generation.prompts import REFUSAL
from meddoc_rag.schemas import Chunk, RetrievedChunk

TEXT = "Unrelated intro sentence here. WHO recommends treatment at systolic 140 mmHg or higher. Follow up monthly."
CHUNK = Chunk(chunk_id="htn:section:0001", doc_id="htn", doc_title="Hypertension guideline", strategy="section",
              text=TEXT, start=0, end=len(TEXT), page_start=19, page_end=19, section="Recommendations", n_tokens=20)
SOURCE = RetrievedChunk(chunk=CHUNK, score=0.99, rank=1, retriever="dense+rerank")


class FakeRetriever:
    name = "dense+rerank"

    def retrieve(self, q, k):
        return [SOURCE]


class FakeAnswerer:
    retriever = FakeRetriever()
    llm = type("L", (), {"model": "fake-model"})()
    refusal_threshold = 0.5

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode

    def answer(self, question: str) -> Answer:
        if self.mode == "error":
            raise LLMError("rate limited")
        if self.mode == "refuse":
            return Answer(question=question, answer=REFUSAL, refused=True, refusal_reason="low_retrieval_confidence",
                          sources=[SOURCE], top_score=0.02, latency_ms={"retrieval": 5.0, "total": 5.0})
        cit = Citation(number=1, chunk_id=CHUNK.chunk_id, doc_id="htn", doc_title=CHUNK.doc_title,
                       page_start=19, page_end=19, section="Recommendations", score=0.99)
        return Answer(question=question, answer="Start treatment at systolic 140 mmHg or higher [1].", refused=False,
                      citations=[cit], sources=[SOURCE], top_score=0.99, model="fake-model",
                      latency_ms={"retrieval": 5.0, "generation": 10.0, "total": 15.0})


def client(mode: str = "ok") -> TestClient:
    return TestClient(create_app(lambda: FakeAnswerer(mode)))


def test_health() -> None:
    with client() as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["retriever"] == "dense+rerank" and body["refusal_threshold"] == 0.5


def test_ask_returns_answer_citations_sources_and_latency() -> None:
    with client() as c:
        r = c.post("/ask", json={"question": "When should treatment start?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["refused"] and body["citations"][0]["label"] == "Hypertension guideline, p.19"
    src = body["sources"][0]
    assert src["cited"] and src["number"] == 1
    start, end = src["highlights"][0]
    assert "140 mmHg" in src["text"][start:end]          # the supporting sentence is highlighted
    assert {"retrieval", "generation", "server_total"} <= set(body["latency_ms"])
    assert "Not medical advice" in body["disclaimer"]


def test_refusal_passes_through() -> None:
    with client("refuse") as c:
        body = c.post("/ask", json={"question": "capital of France?"}).json()
    assert body["refused"] and body["refusal_reason"] == "low_retrieval_confidence"
    assert body["citations"] == [] and body["sources"][0]["highlights"] == []


@pytest.mark.parametrize("question", ["", "hi", "x" * 501])
def test_invalid_questions_rejected(question: str) -> None:
    with client() as c:
        assert c.post("/ask", json={"question": question}).status_code == 422


def test_llm_failure_is_503_not_500() -> None:
    with client("error") as c:
        r = c.post("/ask", json={"question": "When should treatment start?"})
    assert r.status_code == 503 and "rate limited" in r.json()["detail"]


def test_highlight_picks_overlapping_sentence_only() -> None:
    spans = highlight(TEXT, "Treatment starts at systolic 140 mmHg [1].")
    assert [TEXT[s:e] for s, e in spans] == ["WHO recommends treatment at systolic 140 mmHg or higher."]
    assert highlight(TEXT, "Completely different words entirely.") == []


def test_sentence_spans_trim_and_cover() -> None:
    assert [TEXT[s:e] for s, e in sentence_spans(TEXT)][-1] == "Follow up monthly."


def test_request_id_header_and_metrics() -> None:
    with client() as c:
        r = c.post("/ask", json={"question": "When should treatment start?"}, headers={"X-Request-ID": "abc123"})
        assert r.headers["X-Request-ID"] == "abc123"
        assert len(c.get("/health").headers["X-Request-ID"]) == 12      # generated when absent
        c.post("/ask", json={"question": "When should treatment start?"})
        m = c.get("/metrics").json()
    assert m["requests"] == 2 and m["refusal_rate"] == 0.0
    assert {"retrieval", "generation", "server_total"} <= set(m["latency_ms"])
    assert m["latency_ms"]["retrieval"]["p50"] == 5.0


def test_logs_never_contain_question_text(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    secret = "Is my rare condition XYZ-123 treatable?"
    with client() as c:
        logging.getLogger().addHandler(caplog.handler)   # configure_logging replaced root handlers at startup
        c.post("/ask", json={"question": secret})
    ask_logs = [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "ask"]
    assert ask_logs and ask_logs[0].fields["question_chars"] == len(secret)
    assert all(secret not in str(getattr(r, "fields", "")) and secret not in r.getMessage() for r in caplog.records)
