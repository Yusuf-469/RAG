"""Pluggable LLM clients (Groq, Ollama) behind one interface, plus a response cache.

Both providers are called over plain HTTP: Groq exposes the OpenAI-compatible
chat-completions API, Ollama its local /api/chat endpoint. No vendor SDK is
needed, and the exact request is visible in the code.
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import requests

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    cached: bool = False


class LLMError(RuntimeError):
    """The provider failed after retries (bad key, network down, quota exhausted...)."""


class LLMClient(Protocol):
    provider: str
    model: str

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 600) -> LLMResponse: ...


def _post_with_retry(url: str, payload: dict, headers: dict, timeout: float, retries: int = 4) -> dict:
    """POST JSON, retrying on rate limits (429) and transient server errors with exponential backoff."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise LLMError(f"request to {url} failed: {exc}") from exc
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
                # Never include request headers here: they carry the API key.
                raise LLMError(f"{url} returned HTTP {r.status_code}: {r.text[:300]}")
            retry_after = r.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else delay
            log.warning("HTTP %s from LLM provider, retrying in %.1fs", r.status_code, delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise LLMError("unreachable")


class GroqClient:
    """Groq's OpenAI-compatible chat completions endpoint (free tier)."""

    provider = "groq"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.groq.com/openai/v1", timeout: float = 60) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set; add it to .env (see .env.example)")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 600) -> LLMResponse:
        t0 = time.perf_counter()
        data = _post_with_retry(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            {"Authorization": f"Bearer {self._api_key}"},
            self.timeout,
        )
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        # Reasoning models (e.g. gpt-oss) spend hidden "reasoning" tokens from the same budget;
        # if the budget runs out before the answer, fail loudly instead of returning "".
        if not content.strip() and choice.get("finish_reason") == "length":
            raise LLMError(f"{self.model} used all {max_tokens} tokens before answering; raise llm_max_tokens")
        return LLMResponse(
            text=content,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OllamaClient:
    """Local models served by Ollama (fully offline, free, slower on CPU)."""

    provider = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 300) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 600) -> LLMResponse:
        t0 = time.perf_counter()
        data = _post_with_retry(
            f"{self.base_url}/api/chat",
            {"model": self.model, "messages": messages, "stream": False,
             "options": {"temperature": temperature, "num_predict": max_tokens}},
            {},
            self.timeout,
        )
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class CachedLLM:
    """Wraps any client; identical requests are answered from SQLite instead of the API.

    Key = sha256(provider, model, messages, temperature, max_tokens). Re-running the
    evaluation therefore costs no API quota, and results stay reproducible.
    """

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, response TEXT NOT NULL)")

    def _key(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        blob = json.dumps([self.provider, self.model, messages, temperature, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 600) -> LLMResponse:
        key = self._key(messages, temperature, max_tokens)
        row = self._db.execute("SELECT response FROM llm WHERE key = ?", (key,)).fetchone()
        if row:
            return LLMResponse(**{**json.loads(row[0]), "cached": True, "latency_ms": 0.0})
        resp = self.inner.generate(messages, temperature, max_tokens)
        self._db.execute("INSERT OR REPLACE INTO llm (key, response) VALUES (?, ?)", (key, json.dumps(asdict(resp))))
        self._db.commit()
        return resp
