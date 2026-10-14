"""Three chunking strategies behind one interface.

Every chunker returns character spans over `ParsedDocument.text`; page numbers
and section breadcrumbs are derived from offsets by the shared base class, so
strategies differ *only* in where they cut.
"""

import re
from abc import ABC, abstractmethod
from typing import Protocol

from meddoc_rag.schemas import Chunk, ParsedDocument

Span = tuple[int, int]


class TokenCounter(Protocol):
    """Measures length in the embedding model's tokens (what actually gets truncated)."""

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[Span]: ...


class WhitespaceCounter:
    """Word-level counter; deterministic and dependency-free, used in tests."""

    def count(self, text: str) -> int:
        return len(text.split())

    def offsets(self, text: str) -> list[Span]:
        return [m.span() for m in re.finditer(r"\S+", text)]


class HFTokenCounter:
    """Counts tokens with a Hugging Face fast tokenizer (e.g. the bge-small WordPiece vocab)."""

    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_pretrained(model_name)
        self._tok.no_truncation()   # tokenizer.json ships with max_length=512 truncation
        self._tok.no_padding()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def offsets(self, text: str) -> list[Span]:
        return list(self._tok.encode(text, add_special_tokens=False).offsets)


class Chunker(ABC):
    """Base class: subclasses implement `spans`; `chunk` attaches citation metadata."""

    name: str

    def __init__(self, counter: TokenCounter, max_tokens: int = 256, overlap_tokens: int = 32) -> None:
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("need 0 <= overlap_tokens < max_tokens")
        self.counter = counter
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @abstractmethod
    def spans(self, doc: ParsedDocument) -> list[Span]:
        """Character spans to cut the document into."""

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks = []
        for start, end in self.spans(doc):
            raw = doc.text[start:end]
            text = raw.strip()
            if not text:
                continue
            start += len(raw) - len(raw.lstrip())
            end = start + len(text)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{self.name}:{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    strategy=self.name,
                    text=text,
                    start=start,
                    end=end,
                    page_start=doc.page_at(start),
                    page_end=doc.page_at(end - 1),
                    # Midpoint, not start: a chunk that begins with a parent heading and
                    # continues into its first subsection is labelled with the subsection.
                    section=doc.section_at((start + end) // 2),
                    n_tokens=self.counter.count(text),
                )
            )
        return chunks


class FixedSizeChunker(Chunker):
    """Sliding window of `max_tokens` tokens with `overlap_tokens` overlap; ignores all structure."""

    name = "fixed"

    def spans(self, doc: ParsedDocument) -> list[Span]:
        offs = self.counter.offsets(doc.text)
        step = self.max_tokens - self.overlap_tokens
        out = []
        for i in range(0, len(offs), step):
            j = min(i + self.max_tokens, len(offs))
            out.append((offs[i][0], offs[j - 1][1]))
            if j == len(offs):
                break
        return out


class RecursiveChunker(Chunker):
    """Split on the coarsest boundary that works (paragraph > line > sentence > clause > word), then pack greedily."""

    name = "recursive"
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")

    def _pieces(self, text: str, start: int, end: int, seps: tuple[str, ...]) -> list[Span]:
        """Atomic spans of at most `max_tokens`, cut at the coarsest separator available."""
        if self.counter.count(text[start:end]) <= self.max_tokens:
            return [(start, end)]
        for k, sep in enumerate(seps):
            cuts = [m.end() for m in re.finditer(re.escape(sep), text[start:end])]
            cuts = [start + c for c in cuts if start + c < end]
            if not cuts:
                continue
            out: list[Span] = []
            for s, e in zip([start, *cuts], [*cuts, end]):
                out.extend(self._pieces(text, s, e, seps[k + 1:]))
            return out
        # No separator left (e.g. a giant token run): hard-cut on token boundaries.
        offs = self.counter.offsets(text[start:end])
        return [
            (start + offs[i][0], start + offs[min(i + self.max_tokens, len(offs)) - 1][1])
            for i in range(0, len(offs), self.max_tokens)
        ]

    def _pack(self, text: str, pieces: list[Span]) -> list[Span]:
        """Greedily merge consecutive pieces up to `max_tokens`, carrying ~`overlap_tokens` of tail pieces forward."""
        sized = [(p, self.counter.count(text[p[0]:p[1]])) for p in pieces]
        out: list[Span] = []
        cur: list[tuple[Span, int]] = []
        cur_tokens = 0
        for piece, n in sized:
            if cur and cur_tokens + n > self.max_tokens:
                out.append((cur[0][0][0], cur[-1][0][1]))
                tail: list[tuple[Span, int]] = []
                t = 0
                for item in reversed(cur):
                    if t + item[1] > self.overlap_tokens:
                        break
                    tail.insert(0, item)
                    t += item[1]
                cur, cur_tokens = tail, t
                while cur and cur_tokens + n > self.max_tokens:
                    cur_tokens -= cur.pop(0)[1]
            cur.append((piece, n))
            cur_tokens += n
        if cur:
            out.append((cur[0][0][0], cur[-1][0][1]))
        return out

    def _split(self, text: str, start: int, end: int) -> list[Span]:
        return self._pack(text, self._pieces(text, start, end, self.separators))

    def spans(self, doc: ParsedDocument) -> list[Span]:
        return self._split(doc.text, 0, len(doc.text))


class SectionAwareChunker(RecursiveChunker):
    """Never cross a heading: split the document into sections, then recursively chunk inside each."""

    name = "section"

    def __init__(self, counter: TokenCounter, max_tokens: int = 256, overlap_tokens: int = 32, min_section_tokens: int = 32) -> None:
        super().__init__(counter, max_tokens, overlap_tokens)
        self.min_section_tokens = min_section_tokens

    def sections(self, doc: ParsedDocument) -> list[Span]:
        """Heading-to-heading spans; a tiny section (usually a bare parent heading) is merged into the next one."""
        bounds = sorted({0, len(doc.text), *(h.offset for h in doc.headings)})
        raw = [(s, e) for s, e in zip(bounds, bounds[1:]) if doc.text[s:e].strip()]
        merged: list[Span] = []
        pending_start: int | None = None
        for s, e in raw:
            s = pending_start if pending_start is not None else s
            if self.counter.count(doc.text[s:e]) < self.min_section_tokens:
                pending_start = s
                continue
            pending_start = None
            merged.append((s, e))
        if pending_start is not None:
            if merged:
                merged[-1] = (merged[-1][0], len(doc.text))
            else:
                merged.append((pending_start, len(doc.text)))
        return merged

    def spans(self, doc: ParsedDocument) -> list[Span]:
        out: list[Span] = []
        for s, e in self.sections(doc):
            out.extend(self._split(doc.text, s, e))
        return out


STRATEGIES: dict[str, type[Chunker]] = {
    c.name: c for c in (FixedSizeChunker, RecursiveChunker, SectionAwareChunker)
}
