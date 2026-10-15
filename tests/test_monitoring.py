"""Tests for JSON logging and the latency tracker."""

import json
import logging

from meddoc_rag.monitoring import JsonFormatter, LatencyTracker, question_fingerprint


def test_json_formatter_merges_fields() -> None:
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "ask", None, None)
    rec.fields = {"event": "ask", "refused": False}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "ask" and out["event"] == "ask" and out["refused"] is False and out["level"] == "INFO"


def test_fingerprint_hides_text_but_is_stable() -> None:
    a, b = question_fingerprint("secret question"), question_fingerprint("secret question")
    assert a == b and "secret" not in json.dumps(a) and a["question_chars"] == 15


def test_tracker_percentiles_and_rates() -> None:
    t = LatencyTracker(window=100)
    for ms in range(1, 101):
        t.record({"total": float(ms)}, refused=ms % 10 == 0, llm_cached=ms % 2 == 0)
    s = t.summary()
    assert s["requests"] == 100 and s["refusal_rate"] == 0.1 and s["llm_cache_hit_rate"] == 0.5
    assert s["latency_ms"]["total"]["p50"] in (50.0, 51.0) and s["latency_ms"]["total"]["p95"] >= 95.0


def test_tracker_window_drops_old_events() -> None:
    t = LatencyTracker(window=3)
    for ms in (1000.0, 1.0, 1.0, 1.0):
        t.record({"total": ms}, refused=False, llm_cached=None)
    s = t.summary()
    assert s["requests"] == 3 and s["latency_ms"]["total"]["max"] == 1.0 and s["llm_cache_hit_rate"] is None
