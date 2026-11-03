# ==================================================================================================
# 📜 ailab-nlp :: documents
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/documents.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Retrieval works on passages, not whole documents. This module splits long text (annual
#    reports, policies, clinical guidelines) into overlapping chunks that keep their exact
#    position in the source, so every answer can be traced back to the original words.
#
#  🔧 WHAT IT DOES
#    - Chunk      : a passage with document id, sequence number and character offsets.
#    - chunk_text : splits text into chunks of at most `max_chars`, preferring paragraph, line
#                   and sentence boundaries, with `overlap` characters shared between chunks.
#    - html_to_text : readable text from HTML filings, without third-party parsers.
#
#  🧠 DESIGN NOTES
#    - Offsets are exact: text[chunk.start:chunk.end] == chunk.text is a tested invariant. Citations
#      therefore point at real characters, not approximate locations.
#    - Overlap keeps a sentence that straddles a boundary retrievable from either side.
#    - Boundaries are only "preferred": a single 10,000-character paragraph is still split, so
#      no chunk ever exceeds the limit a model's context window was planned for.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: Chunk, chunk_text, html_to_text.
# ==================================================================================================
"""Split documents into traceable, overlapping chunks.

Example:
    >>> text = "First paragraph about revenue.\\n\\nSecond paragraph about debt."
    >>> chunks = chunk_text(text, "report", max_chars=40, overlap=0)
    >>> [c.text for c in chunks]
    ['First paragraph about revenue.', 'Second paragraph about debt.']
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser


# ========================================
# 📂 Chunk
# ========================================
@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int  # character offset of text in the source document
    end: int

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def chunk_text(text: str, doc_id: str, *, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``max_chars`` characters."""
    if max_chars < 20 or not 0 <= overlap < max_chars // 2:
        raise ValueError("need max_chars >= 20 and 0 <= overlap < max_chars / 2")
    n = len(text)
    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text[start:end]
            # Prefer a paragraph, then a line, then a sentence, then a word boundary,
            # as long as it keeps the chunk at least half full.
            for separator in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(separator)
                if cut > max_chars // 2:
                    end = start + cut + len(separator)
                    break
        piece = text[start:end]
        stripped = piece.strip()
        if stripped:
            lead = len(piece) - len(piece.lstrip())
            begin = start + lead
            chunks.append(Chunk(doc_id, len(chunks), stripped, begin, begin + len(stripped)))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        while nxt < end and not text[nxt - 1].isspace():  # start the overlap on a word
            nxt += 1
        start = nxt
    return chunks


# ========================================
# 📂 HTML to Text
# ========================================
class _TextExtractor(HTMLParser):
    _BLOCK = frozenset({"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"})
    _SKIP = frozenset({"script", "style", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    """Readable text from HTML: block tags become line breaks, scripts and styles are dropped."""
    parser = _TextExtractor()
    parser.feed(markup)
    text = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
