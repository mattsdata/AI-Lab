# ==================================================================================================
# 📜 ailab-ml :: tests
# --------------------------------------------------------------------------------------------------
#  File         : packages/ml/tests/test_ml.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the model recovers known coefficients and every metric matches its definition.
#
#  🧠 TESTING APPROACH
#    Fast formulas are checked against slow, obviously-correct reference implementations (pair
#    counting for AUC, explicit CDF comparison for KS) on random data, plus hand-worked cases.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for ailab-ml.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import itertools

import numpy as np
import pytest

from ailab_ml import (
    LogisticRegression,
    brier_score,
    calibration_table,
    gini,
    ks_statistic,
    logistic_irls,
    population_stability_index,
    roc_auc,
    sigmoid,
)


def synthetic(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (rng.random(n) < sigmoid(-1.0 + 1.5 * x[:, 0] - 0.5 * x[:, 1])).astype(float)
    return x, y


# ========================================
# 📂 Logistic Regression Tests
# ========================================
def test_recovers_true_coefficients() -> None:
    x, y = synthetic()
    model = LogisticRegression(l2=0.0, standardize=False).fit(x, y)
    assert model.intercept == pytest.approx(-1.0, abs=0.15)
    assert model.coefficients == pytest.approx([1.5, -0.5], abs=0.15)


def test_regularisation_shrinks_effects_not_base_rate() -> None:
    x, y = synthetic()
    loose = LogisticRegression(l2=0.0).fit(x, y)
    tight = LogisticRegression(l2=5000.0).fit(x, y)
    assert np.all(np.abs(tight.coefficients) < np.abs(loose.coefficients))
    assert sigmoid(tight.intercept) == pytest.approx(y.mean(), abs=0.03)


def test_logistic_validation() -> None:
    with pytest.raises(ValueError):
        LogisticRegression(l2=-1)
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba([[0.0]])
    with pytest.raises(ValueError, match="one class"):
        LogisticRegression().fit([[0.0], [1.0]], [1, 1])
    with pytest.raises(ValueError, match="0 or 1"):
        LogisticRegression().fit([[0.0], [1.0]], [0, 2])
    with pytest.raises(ValueError):
        LogisticRegression().fit([0.0, 1.0], [0, 1])


def test_irls_and_sigmoid_are_stable() -> None:
    assert sigmoid([-1000.0, 0.0, 1000.0]) == pytest.approx([0.0, 0.5, 1.0])
    design = np.hstack([np.ones((4, 1)), [[0.0], [0.0], [1.0], [1.0]]])
    weights = logistic_irls(design, [0, 1, 1, 1], l2=0.0)
    assert sigmoid(design @ weights) == pytest.approx([0.5, 0.5, 1.0, 1.0], abs=1e-3)


# ========================================
# 📂 Metric Tests
# ========================================
def test_auc_matches_pair_counting() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.3).astype(float)
    s = np.round(rng.random(300) + 0.3 * y, 1)  # rounding creates ties on purpose
    pairs = [(a, b) for a, b in itertools.product(s[y == 1], s[y == 0])]
    reference = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert roc_auc(y, s) == pytest.approx(reference)
    assert gini(y, s) == pytest.approx(2 * reference - 1)


def test_auc_edge_cases() -> None:
    assert roc_auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert roc_auc([0, 0, 1, 1], [4, 3, 2, 1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [5, 5, 5, 5]) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.2])
    with pytest.raises(ValueError, match="0 or 1"):
        roc_auc([0, 2], [0.2, 0.3])


def test_ks_matches_definition() -> None:
    assert ks_statistic([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert ks_statistic([0, 1, 0, 1], [1, 2, 3, 4]) == pytest.approx(0.5)
    rng = np.random.default_rng(5)
    y = (rng.random(200) < 0.5).astype(float)
    s = rng.random(200) + 0.2 * y
    grid = np.unique(s)
    reference = max(abs(np.mean(s[y == 1] <= t) - np.mean(s[y == 0] <= t)) for t in grid)
    assert ks_statistic(y, s) == pytest.approx(reference)


def test_brier_and_calibration() -> None:
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    assert brier_score([0, 1], [0.5, 0.5]) == 0.25
    with pytest.raises(ValueError):
        brier_score([], [])
    rng = np.random.default_rng(7)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)  # perfectly calibrated by construction
    table = calibration_table(y, p, bins=5)
    assert len(table) == 5 and sum(b.count for b in table) == 20000
    assert all(abs(b.predicted - b.observed) < 0.03 for b in table)
    with pytest.raises(ValueError):
        calibration_table([0, 1], [0.1, 0.2], bins=5)


def test_population_stability_index() -> None:
    rng = np.random.default_rng(9)
    reference = rng.normal(size=5000)
    assert population_stability_index(reference, rng.normal(size=5000)) < 0.02
    assert population_stability_index(reference, rng.normal(1.0, 1.0, 5000)) > 0.25
    assert np.isfinite(population_stability_index(reference, np.full(100, 99.0)))  # empty bins
    with pytest.raises(ValueError):
        population_stability_index([1.0, 2.0], [1.0])
