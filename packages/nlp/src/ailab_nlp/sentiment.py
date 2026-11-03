# ==================================================================================================
# 📜 ailab-nlp :: sentiment
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/sentiment.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Sentiment is needed in many domains: market news in finance, journal entries in wellbeing
#    tools, survey comments in healthcare, public reaction in climate communication. This module
#    gives all of them one result type and one interface, whatever model sits behind it.
#
#  🔧 WHAT IT DOES
#    - SentimentResult    : label (positive / negative / neutral), score in [-1, 1], confidence.
#    - SentimentAnalyzer  : the interface: analyze(texts) -> list of results, in input order.
#    - create_analyzer    : builds a backend by name ("lexicon", "finbert", or your own).
#    - register_analyzer  : decorator to add a backend.
#
#  🧠 DESIGN NOTES
#    - Score = P(positive) - P(negative). It is a single signed number that averages and plots
#      naturally, while the label and confidence keep the categorical view for reports.
#    - Ties between positive and negative resolve to neutral, never to an arbitrary side.
#    - The interface is batch-first (a list of texts): transformer models are many times faster
#      on batches than one text at a time, and callers rarely have a single text.
#    - Domain-free on purpose: nothing here knows about stocks or patients (ADR-0001).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: SentimentResult, SentimentAnalyzer, analyzer registry.
# ==================================================================================================
"""Model-independent sentiment analysis interface.

Example:
    >>> r = SentimentResult.from_probabilities(positive=0.7, negative=0.1, neutral=0.2)
    >>> r.label, round(r.score, 2), round(r.confidence, 2)
    ('positive', 0.6, 0.7)
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, get_args, runtime_checkable

from mattslib import Registry

Label = Literal["positive", "negative", "neutral"]
LABELS: tuple[str, ...] = get_args(Label)


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Sentiment of one text."""

    label: Label
    score: float  # -1 (very negative) .. +1 (very positive)
    confidence: float  # probability of the chosen label, 0..1

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1, 1], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @classmethod
    def from_probabilities(
        cls, *, positive: float, negative: float, neutral: float
    ) -> SentimentResult:
        """Build a result from class probabilities (normalised if they do not sum to 1)."""
        total = positive + negative + neutral
        if min(positive, negative, neutral) < 0 or total <= 0:
            raise ValueError("probabilities must be non-negative and not all zero")
        pos, neg, neu = positive / total, negative / total, neutral / total
        if pos > neg and pos > neu:
            label: Label = "positive"
        elif neg > pos and neg > neu:
            label = "negative"
        else:
            label = "neutral"  # includes positive/negative ties
        confidence = {"positive": pos, "negative": neg, "neutral": neu}[label]
        return cls(label, pos - neg, confidence)


# ========================================
# 📂 Interface and Registry
# ========================================
@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Anything that scores a batch of texts, returning results in the same order."""

    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]: ...


F = TypeVar("F", bound=Callable[..., SentimentAnalyzer])

ANALYZERS: Registry[SentimentAnalyzer] = Registry(
    "analyzer",
    builtins={
        "lexicon": "ailab_nlp.backends.lexicon",
        "finbert": "ailab_nlp.backends.finbert",
    },
)


def register_analyzer(name: str) -> Callable[[F], F]:
    """Register a sentiment backend under ``name``."""
    return ANALYZERS.register(name)


def available_analyzers() -> list[str]:
    return ANALYZERS.available()


def create_analyzer(name: str, /, **options: Any) -> SentimentAnalyzer:
    """Create the analyzer registered as ``name``, e.g. create_analyzer("finbert")."""
    return ANALYZERS.create(name, **options)
