"""Input safety: prompt-injection/jailbreak detection and emergency notices.

- PromptGuard: Meta's Llama Prompt Guard 2 (86M) on Groq returns the probability
  that a text is a prompt attack. The 22M variant was measured to miss a "DAN"
  jailbreak (0.027) that the 86M model catches (0.997), so 86M is the default.
- Only the user's question is screened. Screening every retrieved chunk would
  cost one call per chunk (the free tier allows 1,000 requests/day); indirect
  injection from documents is handled by prompt design and measured by the
  adversarial evaluation instead.
- emergency_notice: transparent keyword rules (never a model) that attach an
  urgent-care message to questions describing a medical emergency.
"""

import logging
import re
from typing import Protocol

from meddoc_rag.generation.llm import LLMError, _post_with_retry

log = logging.getLogger(__name__)


class InputGuard(Protocol):
    def attack_probability(self, text: str) -> float: ...


class PromptGuard:
    """Groq-hosted Llama Prompt Guard 2; the model's whole output is a probability string."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-prompt-guard-2-86m",
                 base_url: str = "https://api.groq.com/openai/v1", timeout: float = 20) -> None:
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set; the input guard needs it")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def attack_probability(self, text: str) -> float:
        data = _post_with_retry(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": text}]},
            {"Authorization": f"Bearer {self._api_key}"},
            self.timeout,
            retries=2,
        )
        raw = (data["choices"][0]["message"].get("content") or "").strip()
        try:
            return min(1.0, max(0.0, float(raw)))
        except ValueError as exc:
            raise LLMError(f"unexpected guard output {raw[:40]!r}") from exc


def screen_input(guard: InputGuard | None, question: str, threshold: float, fail_open: bool) -> tuple[bool, float | None]:
    """(blocked, score). On guard failure: allow (fail open) or block (fail closed), and log either way."""
    if guard is None:
        return False, None
    try:
        score = guard.attack_probability(question)
    except LLMError as exc:
        log.warning("input guard unavailable (%s); %s", str(exc)[:120], "allowing" if fail_open else "blocking")
        return (not fail_open), None
    return score >= threshold, score


# Each pattern names a situation where reading a guideline is the wrong next step.
_EMERGENCY = [
    r"chest (pain|pressure|tightness)",
    r"(can'?t|cannot|unable to|struggling to|hard to) breathe|difficulty breathing|short(ness)? of breath",
    r"overdos(e|ed|ing)|took too (many|much)",
    r"suicid\w*|kill (myself|himself|herself)|end (my|his|her) life|self[- ]harm",
    r"unconscious|unresponsive|passed out|not breathing|seizure",
    r"(severe|heavy|uncontrolled) bleeding|bleeding (a lot|heavily|won'?t stop)",
    r"stroke|face (is )?droop\w*|slurred speech",
    r"anaphyla\w*|throat (is )?(swelling|closing)",
]
_EMERGENCY_RE = re.compile("|".join(f"(?:{p})" for p in _EMERGENCY), re.IGNORECASE)

EMERGENCY_NOTICE = (
    "If this describes a current emergency (e.g. chest pain, trouble breathing, overdose, thoughts of "
    "self-harm, stroke signs), call your local emergency number now (112 in India, 911 in the US) or go "
    "to the nearest emergency department. This tool cannot help in an emergency."
)


def emergency_notice(question: str) -> str | None:
    return EMERGENCY_NOTICE if _EMERGENCY_RE.search(question) else None
