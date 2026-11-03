# ==================================================================================================
# 📜 ailab-nlp :: evaluation
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/evaluation.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A RAG system is only trustworthy if it is measured. This module scores a pipeline on a small,
#    hand-written test set: did retrieval find the evidence, did the answer cite it, did the
#    system refuse questions the documents cannot answer, and did it use the expected facts.
#
#  🔧 WHAT IT DOES
#    - EvalCase     : question, a phrase that the right passage contains (`evidence`), whether
#                     the question is answerable, and terms a correct answer should mention.
#    - evaluate_rag : runs the pipeline and returns a RagReport.
#    - RagReport    : hit rate@k and MRR (retrieval), citation precision, abstention accuracy,
#                     groundedness and term coverage (answers), plus per-question failures.
#
#  🧠 DESIGN NOTES
#    - Evidence is a phrase, not a chunk id, so test cases survive changes to chunk size.
#    - Unanswerable questions are part of the set on purpose: refusing correctly is a skill, and
#      a system that never says "not found" is hallucinating somewhere.
#    - Every metric comes with the failing questions, because an average hides the case that
#      matters (one wrong debt figure in a credit memo).
#    - Re-run the same set whenever the retriever, chunk size, prompt or model changes.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: EvalCase, evaluate_rag, RagReport.
# ==================================================================================================
"""Evaluate a RAG pipeline: retrieval, citations, abstention and answer content."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ailab_nlp.rag import RagPipeline


# ========================================
# 📂 Test Cases and Report
# ========================================
@dataclass(frozen=True)
class EvalCase:
    question: str
    evidence: str = ""  # phrase found in the passage that answers it ("" if unanswerable)
    answerable: bool = True
    expected_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagReport:
    cases: int
    hit_rate: float  # answerable questions whose evidence was retrieved
    mrr: float  # mean reciprocal rank of the first passage containing the evidence
    citation_precision: float  # cited passages that contain the evidence
    abstention_accuracy: float  # answered when answerable, refused when not
    grounded_rate: float
    term_coverage: float  # expected terms present in answers
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{self.cases} questions · retrieval hit rate {self.hit_rate:.0%} · MRR {self.mrr:.2f}",
            f"citation precision {self.citation_precision:.0%} · abstention accuracy "
            f"{self.abstention_accuracy:.0%} · grounded {self.grounded_rate:.0%} · "
            f"term coverage {self.term_coverage:.0%}",
        ]
        lines += [f"  FAIL {f}" for f in self.failures]
        return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ========================================
# 📂 Evaluation
# ========================================
def evaluate_rag(pipeline: RagPipeline, cases: Sequence[EvalCase]) -> RagReport:
    if not cases:
        raise ValueError("no evaluation cases")
    hits, ranks, precision, abstention, grounded, coverage = [], [], [], [], [], []
    failures: list[str] = []
    for case in cases:
        answer = pipeline.ask(case.question)
        grounded.append(float(answer.grounded))
        correct_refusal = answer.abstained != case.answerable
        abstention.append(float(correct_refusal))
        if not correct_refusal:
            what = "answered an unanswerable question" if answer.abstained is False else "refused"
            failures.append(f"{case.question!r}: {what}")
        if not case.answerable:
            continue

        evidence = case.evidence.lower()
        positions = [i for i, c in enumerate(answer.sources, 1) if evidence in c.text.lower()]
        hits.append(float(bool(positions)))
        ranks.append(1 / positions[0] if positions else 0.0)
        if not positions:
            failures.append(f"{case.question!r}: evidence not retrieved")
        if answer.citations:
            cited = {c.chunk_id for c in answer.citations}
            good = sum(evidence in c.text.lower() for c in answer.sources if c.id in cited)
            precision.append(good / len(cited))
        if case.expected_terms:
            text = answer.text.lower()
            found = sum(term.lower() in text for term in case.expected_terms)
            coverage.append(found / len(case.expected_terms))
            if found < len(case.expected_terms):
                failures.append(f"{case.question!r}: missing expected terms")
    return RagReport(
        cases=len(cases),
        hit_rate=_mean(hits),
        mrr=_mean(ranks),
        citation_precision=_mean(precision),
        abstention_accuracy=_mean(abstention),
        grounded_rate=_mean(grounded),
        term_coverage=_mean(coverage),
        failures=failures,
    )
