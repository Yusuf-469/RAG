"""Core data models shared across ingestion, retrieval and the API."""

from bisect import bisect_right
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Heading(BaseModel):
    """A section heading located at a character offset in `ParsedDocument.text`."""

    offset: int
    level: int
    title: str


class ParsedDocument(BaseModel):
    """One cleaned PDF as a single string, plus offset-based page and section maps.

    Chunkers only decide *where to cut* (character spans); page numbers and
    section titles are then looked up from the offsets, identically for every
    chunking strategy.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    publisher: str
    year: int
    text: str
    page_starts: list[int]          # page_starts[i] = offset where physical page i+1 begins
    page_labels: list[str | None]   # printed label ("iv", "12") where the PDF has a clean one
    headings: list[Heading]         # sorted by offset
    heading_source: Literal["toc", "font", "none"]

    @property
    def num_pages(self) -> int:
        return len(self.page_starts)

    def page_at(self, offset: int) -> int:
        """1-based physical page number containing `offset`."""
        return max(1, bisect_right(self.page_starts, offset))

    @cached_property
    def _breadcrumbs(self) -> list[str]:
        """'Parent > Child' path for each heading, built with a level stack."""
        stack: list[Heading] = []
        crumbs = []
        for h in self.headings:
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
            crumbs.append(" > ".join(s.title for s in stack))
        return crumbs

    @cached_property
    def _heading_offsets(self) -> list[int]:
        return [h.offset for h in self.headings]

    def section_at(self, offset: int) -> str | None:
        """Breadcrumb of the nearest heading at or before `offset`."""
        i = bisect_right(self._heading_offsets, offset) - 1
        return self._breadcrumbs[i] if i >= 0 else None


class Chunk(BaseModel):
    """A retrievable unit of text with everything needed to cite it."""

    chunk_id: str
    doc_id: str
    doc_title: str
    strategy: str
    text: str
    start: int
    end: int
    page_start: int
    page_end: int
    section: str | None
    n_tokens: int
