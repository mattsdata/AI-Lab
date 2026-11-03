# ==================================================================================================
# 📜 ailab-finance :: tests :: scorecard
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/tests/test_scorecard.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the scorecard behaves as a credit risk reviewer expects: correct WoE and IV, exact
#    points scaling (600 at 50:1, +20 doubles the odds), uninformative features dropped, good
#    ranking on a hold-out, stability and drift detected, decisions explained, fairness measured.
#
#  🧠 TESTING APPROACH
#    A synthetic loan book with known drivers (income, debt ratio, delinquencies, credit history),
#    a pure-noise feature and 5% missing incomes; build and hold-out samples use different seeds.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for WoeBinning, Scorecard, validate and group_outcomes.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import math
from typing import Any

import numpy as np
import pytest
from ailab_ml import sigmoid

from ailab_finance.scorecard import Scorecard, WoeBinning, group_outcomes, validate


def loan_book(n: int, seed: int, *, stressed: bool = False) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.3, 0.5, n)
    debt_ratio = rng.uniform(0.5 if stressed else 0.0, 0.9, n)
    delinquencies = rng.poisson(0.4, n).astype(float)
    history = rng.uniform(0, 25, n)
    z = (
        -3.2
        - 0.8 * np.log(income / 30000)
        + 2.5 * debt_ratio
        + 0.7 * delinquencies
        - 0.05 * history
    )
    defaults = (rng.random(n) < sigmoid(z)).astype(float)
    income[rng.random(n) < 0.05] = np.nan
    features = {
        "income": income,
        "debt_ratio": debt_ratio,
        "delinquencies": delinquencies,
        "history_years": history,
        "noise": rng.normal(size=n),
    }
    return features, defaults


@pytest.fixture(scope="module")
def fitted() -> tuple[Scorecard, dict[str, Any], np.ndarray]:
    features, defaults = loan_book(8000, 1)
    return Scorecard().fit(features, defaults), features, defaults


# ========================================
# 📂 WoE Binning Tests
# ========================================
def test_woe_and_iv_match_hand_calculation() -> None:
    x = [1.0, 1.0, 2.0, 2.0]
    y = [1, 0, 0, 0]
    b = WoeBinning.fit("x", x, y, bins=2, smoothing=0.5)
    bad, good = 1.5, 3.5  # totals with smoothing
    low = math.log((1.5 / good) / (1.5 / bad))  # bin x<=1: 1 good, 1 bad (+0.5 each)
    high = math.log((2.5 / good) / (0.5 / bad))  # bin x>1: 2 good, 0 bad
    assert b.woe == pytest.approx((low, high))
    iv = (1.5 / good - 1.5 / bad) * low + (2.5 / good - 0.5 / bad) * high
    assert b.information_value == pytest.approx(iv)
    assert b.labels() == ["(-inf, 1.5]", "(1.5, inf]"]  # median of 1, 1, 2, 2 is 1.5


def test_missing_values_get_their_own_bin() -> None:
    x = np.array([np.nan] * 20 + list(range(80)), dtype=float)
    y = np.array([1] * 15 + [0] * 5 + [0] * 80, dtype=float)  # missing income is risky here
    b = WoeBinning.fit("income", x, y, bins=4)
    assert b.missing_count == 20 and b.missing_woe < -1
    assert b.transform([np.nan])[0] == b.missing_woe
    clean = WoeBinning.fit("income", np.arange(100.0), np.r_[np.ones(10), np.zeros(90)], bins=4)
    assert clean.transform([np.nan])[0] == 0.0  # never seen missing: neutral evidence
    with pytest.raises(ValueError, match="too few"):
        WoeBinning.fit("x", [np.nan, 1.0], [0, 1], bins=5)


def test_monotonicity_flag() -> None:
    x = np.arange(300.0)
    rising = (x > 150).astype(float)
    assert WoeBinning.fit("x", x, rising, bins=3).is_monotonic
    u_shape = ((x < 100) | (x > 200)).astype(float)
    assert not WoeBinning.fit("x", x, u_shape, bins=3).is_monotonic


# ========================================
# 📂 Scorecard Tests
# ========================================
def test_points_scaling_is_exact(fitted: tuple[Scorecard, dict[str, Any], np.ndarray]) -> None:
    scorecard, features, _ = fitted
    pd_ = scorecard.probability_of_default(features)
    expected = 600 + 20 / math.log(2) * (np.log((1 - pd_) / pd_) - math.log(50))
    assert scorecard.score(features) == pytest.approx(expected)  # 600 at 50:1, +20 doubles odds
    assert np.corrcoef(scorecard.score(features), pd_)[0, 1] < -0.9  # higher score, lower risk


def test_uninformative_features_are_dropped(
    fitted: tuple[Scorecard, dict[str, Any], np.ndarray],
) -> None:
    scorecard, _, _ = fitted
    assert "noise" in scorecard.dropped and "noise" not in scorecard.binnings
    assert scorecard.binnings["debt_ratio"].information_value > 0.1
    table = scorecard.table()
    assert "Base points" in table and "noise  dropped" in table and "missing" in table


def test_holdout_ranking_stability_and_drift(
    fitted: tuple[Scorecard, dict[str, Any], np.ndarray],
) -> None:
    scorecard, features, _ = fitted
    reference = scorecard.score(features)
    holdout, defaults = loan_book(4000, 2)
    report = validate(scorecard, holdout, defaults, reference_scores=reference)
    assert report.auc > 0.7 and report.gini == pytest.approx(2 * report.auc - 1)
    assert report.psi is not None and report.psi < 0.10
    # Within 3 standard errors of the observed rate in every decile (400 loans per decile).
    assert all(
        abs(b.predicted - b.observed)
        < 3 * math.sqrt(b.observed * (1 - b.observed) / b.count) + 0.01
        for b in report.calibration
    )
    assert "stable" in report.summary()
    stressed, stressed_defaults = loan_book(4000, 3, stressed=True)
    drift = validate(scorecard, stressed, stressed_defaults, reference_scores=reference)
    assert drift.psi is not None and drift.psi > 0.10  # the population has moved
    assert validate(scorecard, holdout, defaults).psi is None


def test_explanations_add_up_and_name_the_right_reasons(
    fitted: tuple[Scorecard, dict[str, Any], np.ndarray],
) -> None:
    scorecard, _, _ = fitted
    risky = {"income": 18000.0, "debt_ratio": 0.85, "delinquencies": 3.0, "history_years": 2.0}
    explanation = scorecard.explain(risky)
    assert explanation.score == pytest.approx(
        scorecard.base_points + sum(explanation.points.values())
    )
    assert {"debt_ratio", "delinquencies"} <= set(explanation.reasons)
    safe = {"income": 90000.0, "debt_ratio": 0.05, "delinquencies": 0.0, "history_years": 24.0}
    assert scorecard.explain(safe).score > explanation.score + 80
    assert scorecard.explain(safe).probability_of_default < explanation.probability_of_default


def test_scorecard_errors() -> None:
    with pytest.raises(ValueError):
        Scorecard(pdo=0)
    with pytest.raises(RuntimeError):
        Scorecard().score({"x": [1.0]})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="information value"):  # large n: noise IV ~ 0.001
        Scorecard().fit({"noise": rng.normal(size=20000)}, (rng.random(20000) < 0.2).astype(float))
    features, defaults = loan_book(2000, 4)
    scorecard = Scorecard().fit(features, defaults)
    with pytest.raises(KeyError, match="debt_ratio"):
        scorecard.score({"income": [1.0]})


# ========================================
# 📂 Fairness Tests
# ========================================
def test_group_outcomes_and_adverse_impact_ratio() -> None:
    scores = [700, 650, 500, 690, 480, 470]
    defaults = [0, 0, 1, 0, 1, 0]
    groups = ["A", "A", "A", "B", "B", "B"]
    outcomes, ratio = group_outcomes(scores, defaults, groups, cutoff=600)
    assert outcomes["A"].approval_rate == pytest.approx(2 / 3)
    assert outcomes["B"].approval_rate == pytest.approx(1 / 3)
    assert ratio == pytest.approx(0.5)  # below the four-fifths screening line
    assert outcomes["A"].default_rate_approved == 0.0
    nobody, _ = group_outcomes([1, 2], [0, 1], ["A", "B"], cutoff=10)
    assert math.isnan(nobody["A"].default_rate_approved)
    with pytest.raises(ValueError):
        group_outcomes([1, 2], [0], ["A", "B"], cutoff=1)


def test_information_value_is_biased_upward_in_small_samples() -> None:
    """Documents a real trap: on 500 rows pure noise can pass the usual IV threshold."""
    rng = np.random.default_rng(0)
    small = WoeBinning.fit("noise", rng.normal(size=500), (rng.random(500) < 0.2).astype(float))
    large = WoeBinning.fit("noise", rng.normal(size=20000), (rng.random(20000) < 0.2).astype(float))
    assert small.information_value > 0.02 > large.information_value
