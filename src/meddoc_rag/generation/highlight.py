"""Approximate evidence highlighting for the UI.

Within a cited source, mark the sentences that share the most content words
(and numbers) with the answer. This is a lexical heuristic to help a reader find
the supporting sentence quickly; it is not a verification step.
"""

import re

from meddoc_rag.indexing.bm25_index import STOPWORDS

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n{2,}|•")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if (len(w) > 2 or w.isdigit()) and w not in STOPWORDS}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence-like segment, whitespace-trimmed."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        seg = text[s:e]
        s += len(seg) - len(seg.lstrip())
        e -= len(seg) - len(seg.rstrip())
        if e > s:
            out.append((s, e))
    return out


def highlight(source_text: str, answer: str, max_sentences: int = 2, min_shared: int = 3) -> list[tuple[int, int]]:
    """Spans of the source sentences most lexically similar to the answer, in reading order."""
    answer_words = _content_words(re.sub(r"\[\d+\]", " ", answer))
    scored = []
    for s, e in sentence_spans(source_text):
        words = _content_words(source_text[s:e])
        shared = len(words & answer_words)
        if shared >= min_shared:
            scored.append((shared / len(words), shared, s, e))
    best = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[:max_sentences]
    return sorted((s, e) for _, _, s, e in best)
