"""Text cleaning that operates on lines *before* they are joined into a string.

Cleaning at line level keeps character offsets consistent: nothing is deleted
from the final text after offsets (pages, headings) have been recorded.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import replace

from meddoc_rag.ingestion.loader import Line, RawPage

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
# A trailing hyphen before these words is a "suspended" hyphen ("moderate- to vigorous-intensity").
_SUSPENDED_NEXT = {"to", "and", "or", "nor"}


def normalize_text(text: str) -> str:
    """NFKC (non-breaking/hair/figure spaces -> space, ligatures -> letters) + collapse whitespace."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _signature(text: str) -> str:
    # "Page 3 of 206" and "Page 4 of 206" are the same running footer.
    return re.sub(r"\d+", "#", text)


def _edge_lines(page: RawPage, edge: int) -> list[str]:
    lines = [normalize_text(ln.text) for block in page.blocks for ln in block]
    lines = [ln for ln in lines if ln]
    return lines[:edge] + lines[-edge:] if len(lines) > 2 * edge else lines


def find_boilerplate(pages: list[RawPage], edge: int = 3, min_fraction: float = 0.2) -> set[str]:
    """Signatures of lines that repeat at the top/bottom of many pages (running headers/footers, page numbers)."""
    if len(pages) < 5:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_signature(t) for t in _edge_lines(page, edge)})
    return {sig for sig, n in counts.items() if n >= min_fraction * len(pages)}


def clean_page(page: RawPage, boilerplate: set[str], edge: int = 3) -> RawPage:
    """Normalize every line and drop boilerplate lines that sit within `edge` lines of the page top/bottom."""
    flat = [(b, i) for b, block in enumerate(page.blocks) for i in range(len(block))]
    n = len(flat)
    keep_blocks: list[list[Line]] = [[] for _ in page.blocks]
    for pos, (b, i) in enumerate(flat):
        line = page.blocks[b][i]
        text = normalize_text(line.text)
        if not text:
            continue
        at_edge = pos < edge or pos >= n - edge
        if at_edge and _signature(text) in boilerplate:
            continue
        keep_blocks[b].append(replace(line, text=text))
    return replace(page, blocks=[blk for blk in keep_blocks if blk])


def build_vocabulary(pages: list[RawPage]) -> set[str]:
    """Lower-cased words (incl. hyphenated compounds) seen in the document; used to decide de-hyphenation."""
    vocab: set[str] = set()
    for page in pages:
        for block in page.blocks:
            for line in block:
                vocab.update(w.lower() for w in _WORD.findall(line.text))
    return vocab


def join_lines(left: str, right: str, vocab: set[str]) -> str:
    """Join two wrapped lines of the same paragraph, repairing line-break hyphenation.

    "treat-" + "ment"         -> "treatment"       (joined word is a known word)
    "alcohol-" + "based"      -> "alcohol-based"   (hyphenated compound is a known word)
    "moderate-" + "to vig..." -> "moderate- to"    (suspended hyphen, keep as is)
    unknown fragments         -> joined            (soft hyphens are the common case)
    """
    m_left = re.search(r"([A-Za-z]+)-$", left)
    m_right = re.match(r"([a-z]+)", right)
    if not (m_left and m_right):
        return f"{left} {right}"
    a, b = m_left.group(1), m_right.group(1)
    if b in _SUSPENDED_NEXT:
        return f"{left} {right}"
    if (a + b).lower() in vocab:
        return left[:-1] + right
    if f"{a}-{b}".lower() in vocab:
        return left + right
    return left[:-1] + right
