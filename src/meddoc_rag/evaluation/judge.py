"""LLM-as-judge for generated answers.

The judge is a different model family from the generator (self-preference bias:
models tend to rate their own outputs higher). It returns JSON so scores are
machine-readable; unparseable output is recorded as a failure, never guessed.

- Faithfulness: split the answer into factual claims, mark each supported or
  not by the provided sources. Score = supported / total claims.
- Correctness: compare the answer with a human-written reference answer.
"""

import json
import re

from pydantic import BaseModel

from meddoc_rag.generation.llm import LLMClient
from meddoc_rag.generation.prompts import format_sources
from meddoc_rag.schemas import RetrievedChunk

FAITHFULNESS_PROMPT = """You are a strict fact-checker for medical answers.
Split the ANSWER into its individual factual claims. For each claim, decide whether it is fully
supported by the SOURCES (explicitly stated or directly implied). Ignore citation markers like [1].
A claim with a wrong number, dose, population or condition is NOT supported.

Respond with JSON only, no other text:
{"claims": [{"claim": "<short claim>", "supported": true}]}"""

CORRECTNESS_PROMPT = """You grade answers to medical guideline questions against a reference answer.
- "correct": conveys the key facts of the reference with no contradictions.
- "partial": some key facts are right but something important is missing or imprecise.
- "incorrect": wrong, contradicts the reference, or does not answer the question.
Extra correct detail is fine. Respond with JSON only, no other text:
{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}"""


class Faithfulness(BaseModel):
    claims: list[dict] = []
    score: float | None = None      # None when the judge output could not be parsed or had no claims
    parse_error: bool = False


class Correctness(BaseModel):
    verdict: str | None = None      # correct | partial | incorrect
    reason: str | None = None
    parse_error: bool = False


def extract_json(text: str) -> dict | None:
    """First JSON object in the text, tolerating ```json fences, <think> blocks and surrounding prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


class Judge:
    def __init__(self, llm: LLMClient, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def _ask(self, system: str, user: str) -> dict | None:
        resp = self.llm.generate([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 temperature=0.0, max_tokens=self.max_tokens)
        return extract_json(resp.text)

    def faithfulness(self, answer: str, sources: list[RetrievedChunk]) -> Faithfulness:
        data = self._ask(FAITHFULNESS_PROMPT, f"SOURCES:\n{format_sources(sources)}\n\nANSWER:\n{answer}")
        if not data or not isinstance(data.get("claims"), list):
            return Faithfulness(parse_error=True)
        claims = [c for c in data["claims"] if isinstance(c, dict) and "supported" in c]
        if not claims:
            return Faithfulness(claims=[], score=None)
        return Faithfulness(claims=claims, score=sum(bool(c["supported"]) for c in claims) / len(claims))

    def correctness(self, question: str, reference: str, answer: str) -> Correctness:
        data = self._ask(CORRECTNESS_PROMPT, f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\nCANDIDATE ANSWER: {answer}")
        verdict = (data or {}).get("verdict")
        if verdict not in ("correct", "partial", "incorrect"):
            return Correctness(parse_error=True)
        return Correctness(verdict=verdict, reason=data.get("reason"))
