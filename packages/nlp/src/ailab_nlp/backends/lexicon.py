# ==================================================================================================
# 📜 ailab-nlp :: backends :: lexicon
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/backends/lexicon.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A transparent, dependency-free sentiment baseline. It runs anywhere in microseconds, every
#    decision can be explained word by word, and it is the bar a neural model must clearly beat
#    to justify its cost.
#
#  🔧 WHAT IT DOES
#    - LexiconAnalyzer : counts positive and negative words, handles simple negation
#                        ("not strong" counts as negative), and returns smoothed probabilities.
#                        Registered as the "lexicon" backend.
#
#  🧠 DESIGN NOTES
#    - Laplace smoothing (+1 to each class): one matched word gives a mild score, five give a
#      confident one. Raw ratios would call a single word "100% positive".
#    - Negation flips words within a short window after "not", "no", "never" or "...n't".
#    - The default word lists are small and general. Pass domain lists for real use; finance has
#      published dictionaries (e.g. Loughran-McDonald), where words like "liability" are neutral.
#    - Known limits: sarcasm, comparisons and context ("beat" in sport vs earnings). These are
#      exactly the gaps a model like FinBERT is meant to close, and they make good test cases.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: word-list analyzer with negation and smoothing.
# ==================================================================================================
"""Word-list sentiment baseline with negation handling."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from ailab_nlp.sentiment import SentimentResult, register_analyzer

# ========================================
# 📂 Default Word Lists
# ========================================
DEFAULT_POSITIVE = frozenset(
    "gain gains growth grow grew grows strong stronger beat beats exceeded improve improved "
    "improvement improving profit profits profitable record surge surged rise rises rose "
    "upgrade upgraded optimistic success successful robust expand expanded recovery".split()
)
DEFAULT_NEGATIVE = frozenset(
    "loss losses decline declined declining fall fell falls drop dropped weak weaker miss missed "
    "cut cuts downgrade downgraded lawsuit fraud bankruptcy default warning concern concerns "
    "plunge plunged layoffs slump slowdown shortfall investigation recall".split()
)
NEGATORS = frozenset({"not", "no", "never", "without", "neither", "nor", "hardly"})

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


# ========================================
# 📂 Lexicon Analyzer
# ========================================
@register_analyzer("lexicon")
class LexiconAnalyzer:
    """Counts sentiment words with simple negation; fast, transparent, dependency-free."""

    def __init__(
        self,
        positive: Iterable[str] | None = None,
        negative: Iterable[str] | None = None,
        *,
        negation_window: int = 3,
    ) -> None:
        self._positive = frozenset(w.lower() for w in (positive or DEFAULT_POSITIVE))
        self._negative = frozenset(w.lower() for w in (negative or DEFAULT_NEGATIVE))
        if self._positive & self._negative:
            raise ValueError("a word cannot be both positive and negative")
        self._window = negation_window

    def _counts(self, text: str) -> tuple[int, int]:
        positive = negative = 0
        negate_left = 0  # how many following tokens are still negated
        for token in _TOKEN.findall(text.lower()):
            if token in NEGATORS or token.endswith("n't"):
                negate_left = self._window
                continue
            polarity = (token in self._positive) - (token in self._negative)
            if negate_left:
                polarity = -polarity
                negate_left -= 1
            positive += polarity > 0
            negative += polarity < 0
        return positive, negative

    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]:
        results = []
        for text in texts:
            pos, neg = self._counts(text)
            # Laplace smoothing; neutral keeps a fixed prior weight of 1.
            results.append(
                SentimentResult.from_probabilities(positive=pos + 1, negative=neg + 1, neutral=1)
            )
        return results
