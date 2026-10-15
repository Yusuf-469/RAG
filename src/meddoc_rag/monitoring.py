"""Structured (JSON) logging and in-process latency tracking.

Logs are one JSON object per line so they can be searched and aggregated.
Question text is never logged (medical questions can be sensitive): only its
length and a short hash, which still lets repeated questions be spotted.
"""

import hashlib
import json
import logging
import sys
import threading
import time
from collections import deque


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Route all logs to stdout (the container convention), as JSON by default."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_format else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def question_fingerprint(question: str) -> dict:
    """Loggable stand-in for a question: length and a short hash, never the text."""
    return {"question_chars": len(question), "question_sha": hashlib.sha256(question.encode()).hexdigest()[:12]}


def _percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


class LatencyTracker:
    """Rolling window of recent requests; thread-safe because requests run in a thread pool."""

    def __init__(self, window: int = 1000) -> None:
        self._events: deque[dict] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, latency_ms: dict[str, float], refused: bool, llm_cached: bool | None) -> None:
        with self._lock:
            self._events.append({"latency": dict(latency_ms), "refused": refused, "cached": llm_cached})

    def summary(self) -> dict:
        with self._lock:
            events = list(self._events)
        if not events:
            return {"requests": 0}
        stages = sorted({k for e in events for k in e["latency"]})
        latency = {}
        for stage in stages:
            vals = [e["latency"][stage] for e in events if stage in e["latency"]]
            latency[stage] = {"p50": round(_percentile(vals, 0.5), 1), "p95": round(_percentile(vals, 0.95), 1),
                              "max": round(max(vals), 1)}
        generated = [e for e in events if e["cached"] is not None]
        return {
            "requests": len(events),
            "refusal_rate": round(sum(e["refused"] for e in events) / len(events), 3),
            "llm_cache_hit_rate": round(sum(e["cached"] for e in generated) / len(generated), 3) if generated else None,
            "latency_ms": latency,
        }
