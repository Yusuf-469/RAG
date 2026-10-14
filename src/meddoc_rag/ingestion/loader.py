"""Read PDFs with PyMuPDF, keeping layout signals (blocks, font size, bold).

PyMuPDF was chosen over pypdf after measuring both on the corpus: ~9x faster
(2.2s vs 19.5s for 1,077 pages), equal text quality, it exposes font metadata
needed for heading detection, and it only extracts text inside the visible
page area (pypdf pulled in off-page text from a spread layout).
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

BOLD_FLAG = 1 << 4


@dataclass(frozen=True)
class Line:
    """One visual line of text with the layout signals used for heading detection."""

    text: str
    size: float
    bold: bool


@dataclass(frozen=True)
class RawPage:
    """A physical page: blocks (≈ paragraphs) of lines, in content-stream order."""

    number: int               # 1-based physical page
    label: str | None         # printed page label, None if absent or corrupt
    blocks: list[list[Line]]


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int                 # 1-based physical page


@dataclass(frozen=True)
class RawDocument:
    pages: list[RawPage]
    toc: list[TocEntry]


def _clean_label(label: str) -> str | None:
    # Some PDFs (e.g. the USDA guidelines) store labels as raw UTF-16 hex like "<FEFF0043...>1".
    label = label.strip()
    return label if label and "<" not in label else None


def _line_from_spans(spans: list[dict]) -> Line | None:
    text = "".join(s["text"] for s in spans)
    visible = [s for s in spans if s["text"].strip()]
    if not visible:
        return None
    return Line(
        text=text,
        size=max(s["size"] for s in visible),
        bold=all(s["flags"] & BOLD_FLAG or "bold" in s["font"].lower() for s in visible),
    )


def load_pdf(path: Path) -> RawDocument:
    """Extract text blocks with font info, page labels and the PDF outline (bookmarks)."""
    with pymupdf.open(path) as doc:
        pages = []
        for page in doc:
            blocks = []
            for block in page.get_text("dict")["blocks"]:
                lines = [ln for raw in block.get("lines", []) if (ln := _line_from_spans(raw["spans"]))]
                if lines:
                    blocks.append(lines)
            pages.append(RawPage(number=page.number + 1, label=_clean_label(page.get_label()), blocks=blocks))
        toc = [TocEntry(level, title.strip(), page) for level, title, page in doc.get_toc() if page >= 1 and title.strip()]
    return RawDocument(pages=pages, toc=toc)
