"""Assemble cleaned pages into one `ParsedDocument` with page and heading offsets."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from meddoc_rag.ingestion.cleaner import build_vocabulary, clean_page, find_boilerplate, join_lines
from meddoc_rag.ingestion.loader import RawDocument, TocEntry, load_pdf
from meddoc_rag.manifest import SourceDocument
from meddoc_rag.schemas import Heading, ParsedDocument

BLOCK_SEP = "\n\n"      # paragraph boundary: the recursive chunker's first split point
MIN_TOC_ENTRIES = 5     # fewer bookmarks than this -> fall back to font-based headings
# A heading line ending in one of these continues on the next line ("EVIDENCE TO" / "DECISION").
_CONNECTORS = {"and", "or", "of", "to", "for", "the", "in", "on", "with", "a", "an"}


@dataclass(frozen=True)
class _PlacedLine:
    """A line's position in the assembled text, kept for font-based heading detection."""

    offset: int
    text: str
    size: float
    bold: bool
    block_key: tuple[int, int]


def _assemble(raw: RawDocument) -> tuple[str, list[int], list[_PlacedLine]]:
    boilerplate = find_boilerplate(raw.pages)
    pages = [clean_page(p, boilerplate) for p in raw.pages]
    vocab = build_vocabulary(pages)

    parts: list[str] = []
    page_starts: list[int] = []
    placed: list[_PlacedLine] = []
    cursor = 0
    for page in pages:
        if parts:
            parts.append(BLOCK_SEP)
            cursor += len(BLOCK_SEP)
        page_starts.append(cursor)
        for b, block in enumerate(page.blocks):
            if b:
                parts.append(BLOCK_SEP)
                cursor += len(BLOCK_SEP)
            text = ""
            for line in block:
                # Offset is approximate after de-hyphenation joins, which is fine for locating headings.
                placed.append(_PlacedLine(cursor + len(text) + (1 if text else 0), line.text, line.size, line.bold, (page.number, b)))
                text = join_lines(text, line.text, vocab) if text else line.text
            parts.append(text)
            cursor += len(text)
    return "".join(parts), page_starts, placed


def _headings_from_toc(toc: list[TocEntry], text: str, page_starts: list[int]) -> list[Heading]:
    """Place each bookmark at the first match of its title on its page (page start if not found)."""
    headings = []
    for entry in toc:
        if entry.page > len(page_starts):
            continue
        lo = page_starts[entry.page - 1]
        hi = page_starts[entry.page] if entry.page < len(page_starts) else len(text)
        words = re.findall(r"\w+", entry.title)[:8]
        offset = lo
        if words:
            m = re.compile(r"\W*".join(map(re.escape, words)), re.IGNORECASE).search(text, lo, hi)
            if m:
                offset = m.start()
        headings.append(Heading(offset=offset, level=entry.level, title=entry.title))
    return sorted(headings, key=lambda h: h.offset)


def _is_candidate(line: _PlacedLine, body_size: float) -> bool:
    # Relative to the document's own body size: WHO sugars headings are 12pt on an 11pt body,
    # WHO physical-activity headings are 16pt on a 10pt body.
    return line.bold and line.size >= body_size + 0.5 and bool(re.search(r"[A-Za-z]", line.text))


def _is_heading_text(text: str) -> bool:
    words = [w for w in text.split() if re.search(r"[A-Za-z]", w)]
    if not 1 <= len(words) <= 12 or not text[0].isupper() or text.endswith((".", ":")):
        return False  # bold recommendation sentences end in punctuation
    return not (len(words) == 1 and text.isupper())  # infographic labels like "LIMIT"


def _headings_from_fonts(placed: list[_PlacedLine]) -> list[Heading]:
    """Fallback for PDFs without bookmarks: bold lines larger than body text.

    Consecutive candidate lines of similar size are merged first, so a heading
    wrapped over several lines ("EVIDENCE TO" / "DECISION") becomes one title.
    """
    sizes: Counter[float] = Counter()
    for ln in placed:
        sizes[round(ln.size, 1)] += len(ln.text)
    if not sizes:
        return []
    body = sizes.most_common(1)[0][0]

    merged: list[_PlacedLine] = []
    prev_idx = -2
    for idx, ln in enumerate(placed):
        if not _is_candidate(ln, body):
            continue
        prev = merged[-1] if merged else None
        continues = prev is not None and (
            prev.block_key == ln.block_key
            or prev.text.split()[-1].lower() in _CONNECTORS
            or ln.text[0].islower()    # "...adolescents should" / "do at least 60 minutes..."
        )
        if prev and continues and idx == prev_idx + 1 and abs(prev.size - ln.size) < 1.0:
            merged[-1] = _PlacedLine(prev.offset, f"{prev.text} {ln.text}", prev.size, True, prev.block_key)
        else:
            merged.append(ln)
        prev_idx = idx

    kept = [m for m in merged if _is_heading_text(m.text)]
    levels = {s: i + 1 for i, s in enumerate(sorted({round(m.size) for m in kept}, reverse=True)[:3])}
    return [Heading(offset=m.offset, level=levels.get(round(m.size), 3), title=m.text) for m in kept]


def parse_document(meta: SourceDocument, pdf_path: Path) -> ParsedDocument:
    """Full ingestion for one PDF: load -> clean -> assemble -> locate headings."""
    raw = load_pdf(pdf_path)
    text, page_starts, placed = _assemble(raw)
    if len(raw.toc) >= MIN_TOC_ENTRIES:
        headings, source = _headings_from_toc(raw.toc, text, page_starts), "toc"
    else:
        headings = _headings_from_fonts(placed)
        source = "font" if headings else "none"
    return ParsedDocument(
        doc_id=meta.id,
        title=meta.title,
        publisher=meta.publisher,
        year=meta.year,
        text=text,
        page_starts=page_starts,
        page_labels=[p.label for p in raw.pages],
        headings=headings,
        heading_source=source,
    )
