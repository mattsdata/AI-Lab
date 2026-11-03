# ==================================================================================================
# 📜 ailab-nlp :: rag
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/rag.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Answer questions about documents with citations to the exact passages used, and say "not
#    found" instead of guessing. In finance, healthcare or law an uncited answer is not usable,
#    and a confident wrong answer is worse than none.
#
#  🔧 WHAT IT DOES
#    - RagPipeline        : retrieve -> answer -> check citations -> Answer.
#    - LlmAnswerer        : prompts any LanguageModel to answer only from numbered sources.
#    - ExtractiveAnswerer : no-model baseline that quotes the best-matching sentences.
#    - Answer, Citation   : the answer text, valid citations (with character offsets), invalid
#                           citation numbers, the sources shown, and whether it abstained.
#
#  🧠 DESIGN NOTES
#    - Citations are verified, not trusted: a "[7]" when only 5 sources were shown is recorded
#      as invalid. An answer with no valid citation and no abstention is flagged ungrounded.
#    - Abstention is explicit: the model is told to reply NOT FOUND, and the pipeline also
#      abstains when retrieval finds nothing, without calling the model at all.
#    - The extractive baseline makes the whole pipeline testable offline and gives a floor
#      that an LLM answerer must beat on the evaluation set.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: RagPipeline, LlmAnswerer, ExtractiveAnswerer, citation checks.
# ==================================================================================================
"""Retrieval-augmented question answering with verified citations."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ailab_nlp.documents import Chunk
from ailab_nlp.llm import LanguageModel
from ailab_nlp.retrieval import Retriever, tokenize

NOT_FOUND = "NOT FOUND"
SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. After every factual claim, "
    "cite its source as [n]. Do not use outside knowledge. If the sources do not contain the "
    f"answer, reply exactly: {NOT_FOUND}"
)
_CITATION = re.compile(r"\[(\d+)\]")
# Split after . ! ? followed by whitespace, so numbers such as "1.5 billion" stay intact.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


# ========================================
# 📂 Result Types
# ========================================
@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    chunk_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    citations: list[Citation]
    invalid_citations: list[int]
    sources: list[Chunk]
    abstained: bool

    @property
    def grounded(self) -> bool:
        """True if it abstained, or cites at least one real source and no invented ones."""
        return self.abstained or (bool(self.citations) and not self.invalid_citations)


# ========================================
# 📂 Answerers
# ========================================
class Answerer(Protocol):
    def answer(self, question: str, sources: Sequence[Chunk]) -> str: ...


def build_prompt(question: str, sources: Sequence[Chunk]) -> str:
    numbered = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(sources, start=1))
    return f"Sources:\n\n{numbered}\n\nQuestion: {question}\nAnswer with citations:"


class LlmAnswerer:
    """Ask a LanguageModel to answer strictly from the numbered sources."""

    def __init__(self, llm: LanguageModel, *, system: str = SYSTEM_PROMPT) -> None:
        self._llm = llm
        self._system = system

    def answer(self, question: str, sources: Sequence[Chunk]) -> str:
        return self._llm.complete(build_prompt(question, sources), system=self._system)


class ExtractiveAnswerer:
    """Baseline without a model: quote the sentences that share most terms with the question.

    A sentence must share at least ``min_overlap`` question terms (default: 1 for questions of
    up to 3 terms, otherwise 2), so one common word such as "risk" is not mistaken for an
    answer. A matched heading ("Debt.") is quoted together with the sentence that follows it.
    """

    def __init__(self, *, max_sentences: int = 2, min_overlap: int | None = None) -> None:
        self._max = max_sentences
        self._min_overlap = min_overlap

    def answer(self, question: str, sources: Sequence[Chunk]) -> str:
        wanted = set(tokenize(question))
        required = self._min_overlap or (1 if len(wanted) <= 3 else 2)
        scored = []
        for number, chunk in enumerate(sources, start=1):
            sentences = [s.strip() for s in _SENTENCE_BREAK.split(chunk.text) if s.strip()]
            for i, sentence in enumerate(sentences):
                overlap = len(wanted & set(tokenize(sentence)))
                if overlap < required:
                    continue
                if len(sentence.split()) < 6 and i + 1 < len(sentences):  # a heading
                    sentence = f"{sentence} {sentences[i + 1]}"
                scored.append((overlap, -number, -i, sentence, number))
        if not scored:
            return NOT_FOUND
        best = sorted(scored, reverse=True)[: self._max]
        return " ".join(f"{sentence} [{number}]" for *_, sentence, number in best)


# ========================================
# 📂 Pipeline
# ========================================
class RagPipeline:
    """retrieve top-k chunks -> answer from them -> verify citations."""

    def __init__(self, retriever: Retriever, answerer: Answerer, *, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._retriever = retriever
        self._answerer = answerer
        self._k = k

    def ask(self, question: str) -> Answer:
        sources = [hit.chunk for hit in self._retriever.search(question, self._k)]
        if not sources:  # nothing relevant retrieved: abstain without calling the model
            return Answer(question, NOT_FOUND, [], [], [], abstained=True)
        text = self._answerer.answer(question, sources)
        abstained = NOT_FOUND in text.upper()
        citations: list[Citation] = []
        invalid: list[int] = []
        for number in dict.fromkeys(int(n) for n in _CITATION.findall(text)):  # unique, ordered
            if 1 <= number <= len(sources):
                chunk = sources[number - 1]
                citations.append(Citation(number, chunk.id, chunk.start, chunk.end))
            else:
                invalid.append(number)
        return Answer(question, text, citations, invalid, sources, abstained)
