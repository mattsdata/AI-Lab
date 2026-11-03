# ==================================================================================================
# 📜 ailab-nlp :: aggregate
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/aggregate.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Decisions are rarely made on one text. An analyst wants "how has news about this company
#    felt over the last month?", where yesterday's story matters more than one from 3 weeks ago.
#    This module turns many per-text results into one weighted summary.
#
#  🔧 WHAT IT DOES
#    - time_decay_weights : exponential weights from text age, with a configurable half-life.
#    - aggregate_sentiment: weighted average score, overall label, count and label shares.
#    - SentimentSummary   : the result.
#
#  🧠 DESIGN NOTES
#    - Exponential decay with a half-life is easy to explain ("a week-old story counts half")
#      and has one parameter to tune, unlike arbitrary windows that cliff-edge at day 30.
#    - A negative age (a text dated after the decision time) raises an error. Silently using
#      future text is look-ahead bias and makes any back-test look better than reality.
#    - No texts is a normal state, not an error: the summary is neutral with count 0, so
#      downstream models can treat "no news" as its own signal.
#    - A small neutral band stops a score of +0.01 being reported as "positive".
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: time_decay_weights, aggregate_sentiment.
# ==================================================================================================
"""Combine many sentiment results into one time-weighted summary.

Example:
    >>> [round(w, 2) for w in time_decay_weights([0, 7, 14], half_life_days=7)]
    [1.0, 0.5, 0.25]
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ailab_nlp.sentiment import Label, SentimentResult


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True, slots=True)
class SentimentSummary:
    score: float  # weighted mean score, -1..1
    label: Label
    count: int
    positive_share: float  # fraction of texts labelled positive (unweighted)
    negative_share: float


# ========================================
# 📂 Weights
# ========================================
def time_decay_weights(ages_days: Sequence[float], half_life_days: float = 7.0) -> list[float]:
    """Weight 1.0 for age 0, 0.5 at one half-life, 0.25 at two, and so on."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if any(age < 0 for age in ages_days):
        raise ValueError("negative age: a text is dated after the decision time (look-ahead)")
    return [0.5 ** (age / half_life_days) for age in ages_days]


# ========================================
# 📂 Aggregation
# ========================================
def aggregate_sentiment(
    results: Sequence[SentimentResult],
    weights: Sequence[float] | None = None,
    *,
    neutral_band: float = 0.05,
) -> SentimentSummary:
    """Weighted average of scores; label is neutral when |score| <= ``neutral_band``."""
    if not results:
        return SentimentSummary(0.0, "neutral", 0, 0.0, 0.0)
    w = [1.0] * len(results) if weights is None else list(weights)
    if len(w) != len(results):
        raise ValueError("weights and results must have the same length")
    if any(x < 0 for x in w) or sum(w) == 0:
        raise ValueError("weights must be non-negative and not all zero")

    score = sum(r.score * x for r, x in zip(results, w, strict=True)) / sum(w)
    label: Label = (
        "positive" if score > neutral_band else "negative" if score < -neutral_band else "neutral"
    )
    n = len(results)
    return SentimentSummary(
        score=score,
        label=label,
        count=n,
        positive_share=sum(r.label == "positive" for r in results) / n,
        negative_share=sum(r.label == "negative" for r in results) / n,
    )
