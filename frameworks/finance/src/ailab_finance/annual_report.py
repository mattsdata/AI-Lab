# ==================================================================================================
# 📜 ailab-finance :: annual_report
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/annual_report.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    An analyst reads a 200-page annual report to answer the same dozen questions every time:
#    what does the business do, how did revenue move, how much debt, what liquidity, which risks,
#    any litigation, any climate exposure. This module asks those questions of the report with
#    RAG and returns answers that cite the exact passages, or say "not found".
#
#  🔧 WHAT IT DOES
#    - ANALYST_QUESTIONS : the standard question set (topic, question), editable per team.
#    - analyse_report    : chunks the text, builds a BM25 (or hybrid) retriever, answers every
#                          question with the chosen answerer (an LLM, or the extractive baseline).
#    - findings_markdown : a review-ready memo: each answer with its quoted sources, plus a list
#                          of topics the report did not cover.
#
#  🧠 DESIGN NOTES
#    - Built entirely from ailab-nlp components; this module only adds finance knowledge (which
#      questions matter), exactly the split ADR-0001 describes.
#    - "Not found" is a finding, not a failure: a 10-K that says nothing about climate risk is
#      itself useful information for a credit or ESG review.
#    - Quotes are cut from the source by character offsets, so the memo shows the report's own
#      words, never the model's paraphrase of them.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: analyst question set, analyse_report, findings_markdown.
# ==================================================================================================
"""Annual report analyser: standard analyst questions answered with cited passages."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ailab_nlp import (
    Answer,
    Bm25Retriever,
    EmbeddingRetriever,
    ExtractiveAnswerer,
    HybridRetriever,
    RagPipeline,
    Retriever,
    chunk_text,
)
from ailab_nlp.rag import Answerer
from ailab_nlp.retrieval import Embedder

ANALYST_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Business", "What are the company's main products, services and markets?"),
    ("Revenue", "How did total revenue change compared with the prior year, and why?"),
    ("Debt", "How much long-term debt does the company have and when does it mature?"),
    ("Liquidity", "What credit facilities, cash or other liquidity sources are available?"),
    ("Risks", "What are the most significant risk factors?"),
    ("Litigation", "Are there material legal proceedings or regulatory investigations?"),
    ("Climate", "What climate-related risks or emissions targets are disclosed?"),
)


# ========================================
# 📂 Analysis
# ========================================
@dataclass(frozen=True)
class Finding:
    topic: str
    answer: Answer


def analyse_report(
    text: str,
    doc_id: str,
    *,
    answerer: Answerer | None = None,
    questions: Sequence[tuple[str, str]] = ANALYST_QUESTIONS,
    embed: Embedder | None = None,
    k: int = 5,
    max_chars: int = 1500,
    overlap: int = 200,
) -> list[Finding]:
    """Answer each (topic, question) from the report text, with citations."""
    chunks = chunk_text(text, doc_id, max_chars=max_chars, overlap=overlap)
    retriever: Retriever = Bm25Retriever(chunks)
    if embed is not None:
        retriever = HybridRetriever(retriever, EmbeddingRetriever(chunks, embed))
    pipeline = RagPipeline(retriever, answerer or ExtractiveAnswerer(), k=k)
    return [Finding(topic, pipeline.ask(question)) for topic, question in questions]


def findings_markdown(findings: Sequence[Finding], source: str, *, quote_chars: int = 240) -> str:
    """Review memo: answers with quoted sources; uncovered topics listed at the end."""
    lines = [f"# Annual report review: {source}", ""]
    missing = []
    for finding in findings:
        answer = finding.answer
        if answer.abstained:
            missing.append(finding.topic)
            continue
        lines += [f"## {finding.topic}", f"*{answer.question}*", "", answer.text, ""]
        if not answer.grounded:
            lines += [
                "> ⚠️ Check: answer has no valid citation or cites a source that was not shown.",
                "",
            ]
        by_number = {i: chunk for i, chunk in enumerate(answer.sources, start=1)}
        for citation in answer.citations:
            quote = by_number[citation.number].text[:quote_chars].replace("\n", " ")
            lines.append(
                f'> [{citation.number}] chars {citation.start}-{citation.end}: "{quote}..."'
            )
        lines.append("")
    if missing:
        lines += ["## Not found in the report", "", ", ".join(missing), ""]
    lines.append("Generated with retrieval-augmented answering. Verify against the source filing.")
    return "\n".join(lines)
