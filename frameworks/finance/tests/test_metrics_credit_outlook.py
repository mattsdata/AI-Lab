# ==================================================================================================
# 📜 ailab-finance :: tests :: metrics, credit and outlook
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/tests/test_metrics_credit_outlook.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the financial calculations against hand-worked values and the economic behaviour
#    a risk reviewer would check (more leverage or volatility means more default risk).
#
#  🧠 TESTING APPROACH
#    Small hand-checkable arrays for formulas; a seeded synthetic price path for the outlook;
#    and an explicit look-ahead test that changes future prices and expects no change.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for metrics, naive distance to default and historical outlook.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import math

import numpy as np
import pytest

from ailab_finance import (
    Instrument,
    PriceSeries,
    annualised_volatility,
    forward_returns,
    historical_outlook,
    historical_var,
    log_returns,
    max_drawdown,
    naive_distance_to_default,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    warning_flag,
)


def random_walk(days: int = 600, seed: int = 7) -> PriceSeries:
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, days)))
    dates = np.arange(
        np.datetime64("2023-01-01"), np.datetime64("2023-01-01") + np.timedelta64(days, "D")
    )
    return PriceSeries(Instrument("SYN"), dates, prices)


# ========================================
# 📂 Metrics Tests
# ========================================
def test_returns() -> None:
    prices = [100.0, 110.0, 99.0]
    assert np.allclose(simple_returns(prices), [0.10, -0.10])
    assert math.isclose(log_returns(prices).sum(), math.log(99 / 100))  # log returns add up
    assert np.allclose(forward_returns([100.0, 105.0, 110.0, 121.0], 2), [0.10, 121 / 105 - 1])


def test_volatility_sharpe_sortino() -> None:
    r = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    assert math.isclose(annualised_volatility(r), np.std(r, ddof=1) * math.sqrt(252))
    assert math.isclose(sharpe_ratio(r), np.mean(r) / np.std(r, ddof=1) * math.sqrt(252))
    downside = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert math.isclose(sortino_ratio(r), np.mean(r) / downside * math.sqrt(252))


def test_ratios_refuse_degenerate_input() -> None:
    with pytest.raises(ValueError):
        sharpe_ratio([0.01, 0.01, 0.01])
    with pytest.raises(ValueError):
        sortino_ratio([0.01, 0.02])
    with pytest.raises(ValueError):
        simple_returns([100.0])
    with pytest.raises(ValueError):
        forward_returns([1.0, 2.0], 0)


def test_drawdown_and_var() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 130.0, 117.0]) == pytest.approx(-0.25)
    returns = np.linspace(-0.05, 0.05, 101)  # 1% steps of 0.001
    assert historical_var(returns, 0.95) == pytest.approx(0.045)
    with pytest.raises(ValueError):
        historical_var(returns, 0.3)


# ========================================
# 📂 Credit Risk Tests
# ========================================
def test_distance_to_default_matches_hand_calculation() -> None:
    est = naive_distance_to_default(
        equity_value=100, debt=100, equity_volatility=0.40, expected_return=0.05
    )
    # sigma_D = 0.15, sigma_V = 0.275, DD = (ln 2 + 0.05 - 0.275^2 / 2) / 0.275
    assert est.asset_volatility == pytest.approx(0.275)
    assert est.distance_to_default == pytest.approx(2.5649, abs=1e-3)
    assert est.default_probability == pytest.approx(0.00516, abs=1e-4)


def test_more_leverage_or_volatility_means_more_risk() -> None:
    base = dict(equity_value=100.0, debt=50.0, equity_volatility=0.3, expected_return=0.05)
    safe = naive_distance_to_default(**base)
    leveraged = naive_distance_to_default(**{**base, "debt": 200.0})
    volatile = naive_distance_to_default(**{**base, "equity_volatility": 0.8})
    assert leveraged.default_probability > safe.default_probability
    assert volatile.default_probability > safe.default_probability


def test_credit_validation_and_flags() -> None:
    with pytest.raises(ValueError):
        naive_distance_to_default(
            equity_value=0, debt=1, equity_volatility=0.3, expected_return=0.0
        )
    with pytest.raises(ValueError):
        naive_distance_to_default(equity_value=1, debt=1, equity_volatility=0, expected_return=0.0)
    est = naive_distance_to_default(
        equity_value=100, debt=100, equity_volatility=0.40, expected_return=0.05
    )
    assert warning_flag(est) == "GREEN"  # PD ~0.5%
    assert warning_flag(est, amber=0.005) == "AMBER"
    assert warning_flag(est, amber=0.001, red=0.005) == "RED"
    with pytest.raises(ValueError):
        warning_flag(est, amber=0.1, red=0.05)


# ========================================
# 📂 Outlook Tests
# ========================================
def test_outlook_probabilities_are_consistent() -> None:
    outlook = historical_outlook(random_walk(), horizon=21)
    p = outlook.threshold_probabilities
    assert 0 <= p[0.50] <= p[0.25] <= p[0.10] <= 1  # higher targets are never more likely
    low, high = outlook.interval
    assert low < outlook.median_return < high
    text = outlook.summary()
    assert "P(return >= +10%)" in text and "Not financial advice" in text


def test_outlook_has_no_look_ahead() -> None:
    original = random_walk()
    as_of = "2024-03-01"
    future_changed = PriceSeries(
        original.instrument,
        original.dates,
        np.where(original.dates > np.datetime64(as_of), original.close * 3, original.close),
    )
    a = historical_outlook(original, as_of=as_of)
    b = historical_outlook(future_changed, as_of=as_of)
    assert a == b  # tripling every later price changes nothing
    assert a.as_of == np.datetime64(as_of)


def test_outlook_requires_enough_history() -> None:
    short = random_walk(days=50)
    with pytest.raises(ValueError, match="windows"):
        historical_outlook(short, horizon=21)
    with pytest.raises(ValueError, match="more than"):
        historical_outlook(short, horizon=60)
    with pytest.raises(ValueError):
        historical_outlook(random_walk(), interval_level=1.0)
