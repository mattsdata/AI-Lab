# ==================================================================================================
# 📜 ailab-ml :: metrics
# --------------------------------------------------------------------------------------------------
#  File         : packages/ml/src/ailab_ml/metrics.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    The metrics a model risk reviewer asks for before a probability model goes live: can it rank
#    (AUC, Gini, KS), are its probabilities right (Brier, calibration table), and is today's
#    population still like the one it was built on (PSI).
#
#  🔧 WHAT IT DOES
#    - roc_auc, gini              : ranking power (Gini = 2 x AUC - 1, the banking convention).
#    - ks_statistic               : largest gap between the score distributions of the classes.
#    - brier_score                : mean squared error of probabilities.
#    - calibration_table          : predicted vs observed rate in equal-sized bands.
#    - population_stability_index : drift of a score distribution between two samples.
#
#  🧠 DESIGN NOTES
#    - AUC uses the rank (Mann-Whitney) formula with averaged ties, so it is exact, O(n log n),
#      and equals the probability that a random positive outranks a random negative.
#    - PSI bins come from the reference sample's quantiles and use a small floor, so an empty bin
#      gives a large but finite value instead of infinity. Common reading: < 0.10 stable,
#      0.10-0.25 monitor, > 0.25 significant shift.
#    - Every function validates its input and fails clearly on a single-class target, where
#      ranking metrics are undefined.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: AUC, Gini, KS, Brier, calibration table, PSI.
# ==================================================================================================
"""Validation metrics for probability models.

Example:
    >>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    0.75
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _binary(targets: npt.ArrayLike, scores: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(targets, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("targets and scores must be 1-D arrays of equal length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("targets must be 0 or 1")
    if len(np.unique(y)) < 2:
        raise ValueError("both classes must be present")
    return y, s


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average rank for ties, 1-based
        i = j + 1
    return ranks


# ========================================
# 📂 Ranking Power
# ========================================
def roc_auc(targets: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve: P(score of a random positive > score of a random negative)."""
    y, s = _binary(targets, scores)
    ranks = _average_ranks(s)
    positives = float(y.sum())
    negatives = len(y) - positives
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def gini(targets: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Gini coefficient (accuracy ratio) = 2 x AUC - 1."""
    return 2 * roc_auc(targets, scores) - 1


def ks_statistic(targets: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Kolmogorov-Smirnov: max distance between the classes' cumulative score distributions."""
    y, s = _binary(targets, scores)
    thresholds = np.unique(s)
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    cdf_pos = np.searchsorted(pos, thresholds, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, thresholds, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


# ========================================
# 📂 Probability Accuracy
# ========================================
def brier_score(targets: npt.ArrayLike, probabilities: npt.ArrayLike) -> float:
    """Mean squared error between predicted probabilities and 0/1 outcomes (lower is better)."""
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("targets and probabilities must be non-empty 1-D arrays of equal length")
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    predicted: float  # mean predicted probability in the band
    observed: float  # actual event rate in the band
    count: int


def calibration_table(
    targets: npt.ArrayLike, probabilities: npt.ArrayLike, *, bins: int = 10
) -> list[CalibrationBin]:
    """Sort by prediction, split into ``bins`` equal-sized bands, compare predicted vs observed."""
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or len(y) < bins or bins < 1:
        raise ValueError("need equal-length inputs with at least `bins` rows")
    order = np.argsort(p, kind="mergesort")
    return [
        CalibrationBin(float(p[band].mean()), float(y[band].mean()), len(band))
        for band in np.array_split(order, bins)
    ]


# ========================================
# 📂 Stability
# ========================================
def population_stability_index(
    reference: npt.ArrayLike, current: npt.ArrayLike, *, bins: int = 10, floor: float = 1e-4
) -> float:
    """PSI between a reference distribution (e.g. build sample) and a current one."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if len(ref) < bins or len(cur) == 0:
        raise ValueError("reference needs at least `bins` values and current must not be empty")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.maximum(np.histogram(ref, edges)[0] / len(ref), floor)
    actual = np.maximum(np.histogram(cur, edges)[0] / len(cur), floor)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
