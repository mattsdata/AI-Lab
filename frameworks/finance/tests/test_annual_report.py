# ==================================================================================================
# 📜 ailab-finance :: tests :: annual_report
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/tests/test_annual_report.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the analyser answers the analyst questions from the right passages, reports
#    uncovered topics as "not found", and flags answers whose citations cannot be verified.
#
#  🧠 TESTING APPROACH
#    A short synthetic 10-K with distinct sections and no climate disclosure; the extractive
#    answerer (no model) and a fake LLM that invents a citation.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for analyse_report and findings_markdown.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
from collections.abc import Sequence

import numpy as np
from ailab_nlp import Chunk, LlmAnswerer

from ailab_finance.annual_report import ANALYST_QUESTIONS, analyse_report, findings_markdown

TEN_K = (
    "Item 1. Business. Acme designs industrial sensors and sells monitoring software services "
    "to utilities in North America and Europe.\n\n"
    "Item 7. Revenue. Total revenue increased 9 percent compared with the prior year to 2.1 "
    "billion dollars, mainly because software subscriptions grew.\n\n"
    "Debt. Long-term debt was 800 million dollars at year end, consisting of senior notes that "
    "mature in 2030.\n\n"
    "Liquidity. We held 300 million dollars of cash and an undrawn revolving credit facility of "
    "400 million dollars.\n\n"
    "Item 1A. Risk factors. Significant risks include dependence on a single chip supplier and "
    "cyber attacks on customer networks.\n\n"
    "Item 3. Legal proceedings. A patent lawsuit filed by a competitor is pending; we believe "
    "it is without merit."
)


def findings() -> list:  # type: ignore[type-arg]
    return analyse_report(TEN_K, "acme-2025", max_chars=300, overlap=0, k=3)


# ========================================
# 📂 Analyser Tests
# ========================================
def test_answers_come_from_the_right_passages() -> None:
    by_topic = {f.topic: f.answer for f in findings()}
    assert [t for t, _ in ANALYST_QUESTIONS] == list(by_topic)
    assert "800 million" in by_topic["Debt"].text and by_topic["Debt"].grounded
    assert "revolving credit facility" in by_topic["Liquidity"].text
    assert "patent lawsuit" in by_topic["Litigation"].text
    debt_source = by_topic["Debt"].citations[0]
    assert TEN_K[debt_source.start : debt_source.end].startswith("Debt.")


def test_memo_quotes_sources_and_lists_gaps() -> None:
    memo = findings_markdown(findings(), "ACME 10-K 2025")
    assert memo.startswith("# Annual report review: ACME 10-K 2025")
    assert "## Debt" in memo and '"Debt. Long-term debt was 800 million' in memo
    assert "## Not found in the report" in memo and "Climate" in memo.split("Not found")[1]
    assert "Verify against the source filing" in memo


def test_unverifiable_citations_are_flagged() -> None:
    class InventsSources:
        def complete(self, prompt: str, *, system: str | None = None) -> str:
            return "Debt is large [7]."

    result = analyse_report(
        TEN_K, "acme", answerer=LlmAnswerer(InventsSources()), questions=[("Debt", "debt")]
    )
    assert "Check: answer has no valid citation" in findings_markdown(result, "acme")


def test_hybrid_retrieval_option() -> None:
    def embed(texts: Sequence[str]) -> np.ndarray:
        words = ["debt", "revenue", "risk", "cash", "lawsuit", "sensors"]
        return np.array([[t.lower().count(w) for w in words] for t in texts], dtype=float) + 0.01

    result = analyse_report(TEN_K, "acme", embed=embed, max_chars=300, overlap=0, k=2)
    debt = {f.topic: f.answer for f in result}["Debt"]
    assert any(isinstance(c, Chunk) and c.text.startswith("Debt.") for c in debt.sources)
