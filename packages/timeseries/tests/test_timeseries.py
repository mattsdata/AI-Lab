# ==================================================================================================
# 📜 ailab-timeseries :: tests
# --------------------------------------------------------------------------------------------------
#  File         : packages/timeseries/tests/test_timeseries.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves windows are causal and match naive loops, aggregation enforces coverage, trend
#    intervals have the coverage they claim (simulation), and seasonal tools behave on a
#    synthetic warming climate.
#
#  🧠 TESTING APPROACH
#    Reference loops for rolling statistics; Monte Carlo coverage checks for confidence
#    intervals, including autocorrelated noise where unadjusted intervals are known to be too
#    narrow; a synthetic daily climate with a seasonal cycle and known warming.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for ailab-timeseries.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import numpy as np
import pytest

from ailab_timeseries import (
    aggregate,
    daily_climatology,
    day_of_year,
    linear_trend,
    rolling_mean,
    rolling_std,
    seasonal_anomalies,
    seasonal_threshold_flags,
    t_quantile,
)


def synthetic_climate(start: int = 1961, end: int = 2020, warming: float = 0.03, seed: int = 0):  # type: ignore[no-untyped-def]
    """Daily temperature: seasonal cycle + `warming` deg C per year + noise."""
    dates = np.arange(np.datetime64(f"{start}-01-01"), np.datetime64(f"{end + 1}-01-01"))
    years = (dates.astype("datetime64[D]") - np.datetime64(f"{start}-01-01")).astype(int) / 365.25
    season = 3 * np.sin(2 * np.pi * day_of_year(dates) / 365)
    noise = np.random.default_rng(seed).normal(0, 1.0, len(dates))
    return dates, 27 + season + warming * years + noise


# ========================================
# 📂 Rolling Window Tests
# ========================================
def test_rolling_matches_naive_loops_and_is_causal() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    x[[5, 17]] = np.nan
    mean = rolling_mean(x, 7, min_periods=5)
    std = rolling_std(x, 7, min_periods=5)
    for i in range(50):
        window = x[max(0, i - 6) : i + 1]
        present = window[~np.isnan(window)]
        if len(present) >= 5:
            assert mean[i] == pytest.approx(present.mean())
            assert std[i] == pytest.approx(present.std(ddof=1))
        else:
            assert np.isnan(mean[i]) and np.isnan(std[i])
    changed = x.copy()
    changed[30:] = 100.0
    assert np.array_equal(rolling_mean(changed, 7)[:30], rolling_mean(x, 7)[:30], equal_nan=True)


def test_rolling_validation() -> None:
    with pytest.raises(ValueError):
        rolling_mean([1.0, 2.0], 0)
    with pytest.raises(ValueError):
        rolling_mean([1.0, 2.0], 2, min_periods=3)
    with pytest.raises(ValueError):
        rolling_std(np.zeros((2, 2)), 2)
    assert np.isnan(rolling_std([1.0, 2.0, 3.0], 3, min_periods=1)[0])  # one value: no std


# ========================================
# 📂 Aggregation Tests
# ========================================
def test_aggregate_enforces_coverage() -> None:
    dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-03-01"))  # Jan + Feb (leap)
    values = np.ones(len(dates))
    values[31 + 10 :] = np.nan  # February: only 10 of 29 days
    starts, monthly = aggregate(dates, values, freq="month", how="sum")
    assert starts.tolist() == [np.datetime64("2024-01-01"), np.datetime64("2024-02-01")]
    assert monthly[0] == 31 and np.isnan(monthly[1])
    _, lenient = aggregate(dates, values, freq="month", how="sum", min_coverage=0.3)
    assert lenient[1] == 10


def test_aggregate_year_and_validation() -> None:
    dates = np.arange(np.datetime64("2023-01-01"), np.datetime64("2025-01-01"))
    values = np.arange(len(dates), dtype=float)
    _, maxima = aggregate(dates, values, freq="year", how="max")
    assert maxima.tolist() == [364.0, 365.0 + 365.0]  # 2024 has 366 days
    with pytest.raises(ValueError):
        aggregate(dates, values, min_coverage=0)
    with pytest.raises(ValueError):
        aggregate(dates, values[:-1])


# ========================================
# 📂 Trend Tests
# ========================================
def test_t_quantiles_match_tables() -> None:
    for dof, table in [(5, 2.571), (10, 2.228), (30, 2.042)]:
        assert t_quantile(0.975, dof) == pytest.approx(table, abs=0.01)


def test_trend_recovers_slope_per_decade() -> None:
    years = np.arange(1961, 2021, dtype=float)
    rng = np.random.default_rng(3)
    trend = linear_trend(years, 0.02 * years + rng.normal(0, 0.1, len(years)), per=10)
    assert trend.low < 0.2 < trend.high and trend.is_significant
    flat = linear_trend(years, rng.normal(0, 0.1, len(years)), per=10)
    assert flat.low < 0 < flat.high and not flat.is_significant


def coverage(phi: float, adjust: bool, sims: int = 400) -> float:
    rng = np.random.default_rng(11)
    x = np.arange(60, dtype=float)
    hits = 0
    for _ in range(sims):
        e = np.zeros(60)
        shocks = rng.normal(size=60)
        for i in range(60):
            e[i] = (phi * e[i - 1] if i else 0.0) + shocks[i]
        t = linear_trend(x, 0.05 * x + e, adjust_autocorrelation=adjust)
        hits += t.low <= 0.05 <= t.high
    return hits / sims


def test_intervals_have_their_claimed_coverage() -> None:
    assert 0.92 <= coverage(0.0, adjust=True) <= 0.98  # independent noise: ~95%
    naive, adjusted = coverage(0.6, adjust=False), coverage(0.6, adjust=True)
    assert naive < 0.85  # plain OLS is overconfident on autocorrelated data
    assert adjusted > naive + 0.07  # the effective-sample-size correction restores most of it


def test_trend_validation_and_nan_handling() -> None:
    with pytest.raises(ValueError):
        linear_trend([1, 1, 1, 1], [1, 2, 3, 4])
    with pytest.raises(ValueError):
        linear_trend([1, 2, 3, 4], [1, 2, 3, 4], level=1.0)
    trend = linear_trend([1, 2, 3, 4, 5, np.nan], [2, 4, 6, 8, 10, 99])
    assert trend.slope == pytest.approx(2.0) and trend.n == 5


# ========================================
# 📂 Seasonal Tests
# ========================================
def test_day_of_year_folds_leap_day() -> None:
    days = np.array(
        ["2024-01-01", "2024-02-28", "2024-02-29", "2024-03-01", "2024-12-31"],
        dtype="datetime64[D]",
    )
    assert day_of_year(days).tolist() == [0, 58, 58, 59, 364]


def test_climatology_and_anomalies_on_synthetic_climate() -> None:
    dates, temperature = synthetic_climate()
    normals = daily_climatology(dates, temperature, baseline=(1961, 1990))
    assert len(normals) == 365 and np.ptp(normals) == pytest.approx(6, abs=0.3)  # 2 x amplitude 3
    anomalies = seasonal_anomalies(dates, temperature, baseline=(1961, 1990))
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    assert abs(np.mean(anomalies[years <= 1990])) < 0.05  # baseline anomalies centre on zero
    assert np.mean(anomalies[years >= 2011]) == pytest.approx(0.03 * 35, abs=0.15)  # ~1.05 deg C


def test_threshold_flags_detect_more_hot_days_after_warming() -> None:
    dates, temperature = synthetic_climate()
    flags = seasonal_threshold_flags(dates, temperature, baseline=(1961, 1990), quantile=0.9)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    assert np.mean(flags[years <= 1990]) == pytest.approx(0.10, abs=0.01)  # by construction
    assert np.mean(flags[years >= 2011]) > 0.3  # warming makes "hot" days common


def test_seasonal_validation() -> None:
    dates, temperature = synthetic_climate(2000, 2001)
    with pytest.raises(ValueError, match="baseline"):
        daily_climatology(dates, temperature, baseline=(1961, 1990))
    with pytest.raises(ValueError, match="baseline"):
        seasonal_threshold_flags(dates, temperature, baseline=(1961, 1990))
    with pytest.raises(ValueError):
        seasonal_threshold_flags(dates, temperature, baseline=(2000, 2001), quantile=1.0)
