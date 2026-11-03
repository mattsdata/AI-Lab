# ==================================================================================================
# 📜 ailab-finance :: tests :: forecast
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/tests/test_forecast.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the outlook model is honest: no look-ahead anywhere, no skill claimed where none
#    exists, real skill found where information exists, and sensible handling of rare events.
#
#  🧠 TESTING APPROACH
#    Two synthetic worlds with a fixed seed:
#      1. a pure random walk, where nothing can be predicted, so skill must not be positive;
#      2. the same walk with "news" that genuinely carries information about the next month,
#         so a correct model must clearly beat the baseline.
#    Look-ahead is tested by changing the future and checking nothing before it moves.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for features, OutlookModel, walk_forward and model_outlook.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import types
from typing import Any

import numpy as np
import pytest

from ailab_finance import Instrument, PriceSeries
from ailab_finance.forecast import (
    MARKET_FEATURES,
    SENTIMENT_FEATURES,
    WARMUP,
    OutlookModel,
    build_features,
    model_outlook,
    walk_forward,
)


def random_walk(days: int = 1500, seed: int = 7) -> PriceSeries:
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, days)))
    start = np.datetime64("2020-01-01")
    return PriceSeries(
        Instrument("SYN"), np.arange(start, start + np.timedelta64(days, "D")), prices
    )


def informative_news(series: PriceSeries) -> Any:
    """Synthetic world where news carries real information about the next month."""
    forward = build_features(series).forward
    rng = np.random.default_rng(1)
    scores = {
        d: float(np.clip(np.tanh(5 * (f if np.isfinite(f) else 0.0)) + rng.normal(0, 0.1), -1, 1))
        for d, f in zip(series.dates, forward, strict=True)
    }
    return lambda day: types.SimpleNamespace(score=scores[day], count=5)


def no_news(day: Any) -> Any:
    return types.SimpleNamespace(score=0.0, count=0)


# ========================================
# 📂 Feature Tests
# ========================================
def test_features_shape_warmup_and_labels() -> None:
    s = random_walk(400)
    data = build_features(s, horizon=21, sentiment=no_news)
    assert data.names == MARKET_FEATURES + SENTIMENT_FEATURES
    assert np.isnan(data.features[WARMUP - 1]).all() and not np.isnan(data.features[WARMUP]).any()
    assert data.forward[0] == pytest.approx(s.close[21] / s.close[0] - 1)
    assert np.isnan(data.forward[-21:]).all()  # the future of the last month is unknown
    assert data.usable.sum() == 400 - WARMUP - 21


def test_features_are_causal() -> None:
    s = random_walk(600)
    cut = 450
    changed = PriceSeries(
        s.instrument, s.dates, np.where(np.arange(600) > cut, s.close * 2, s.close)
    )
    a, b = build_features(s).features, build_features(changed).features
    assert np.array_equal(a[: cut + 1], b[: cut + 1], equal_nan=True)


def test_feature_validation() -> None:
    with pytest.raises(ValueError):
        build_features(random_walk(300), horizon=0)


# ========================================
# 📂 Model Tests
# ========================================
def test_model_learns_a_real_signal_and_keeps_probabilities_monotone() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 3))
    y = 0.08 * x[:, 0] + rng.normal(0, 0.03, 800)  # feature 0 drives returns
    model = OutlookModel(thresholds=(0.25, 0.10)).fit(x, y)  # unsorted input on purpose
    high = model.predict([2.0, 0.0, 0.0])
    low = model.predict([-2.0, 0.0, 0.0])
    assert high.median_return > 0.1 > -0.1 > low.median_return
    assert high.probabilities[0.10] > 0.8 > 0.2 > low.probabilities[0.10]
    assert high.probabilities[0.25] <= high.probabilities[0.10]  # monotone across targets
    assert high.interval[0] < high.median_return < high.interval[1]
    assert model.thresholds == (0.10, 0.25)


def test_rare_events_fall_back_to_base_rate() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    y = rng.normal(0, 0.02, 300)  # a +50% month never happens here
    prediction = OutlookModel(thresholds=(0.5,)).fit(x, y).predict([0.0, 0.0])
    assert prediction.modelled == {0.5: False}
    assert prediction.probabilities[0.5] == pytest.approx(0.5 / 301)  # smoothed, never 0


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        OutlookModel(interval_level=1.0)
    with pytest.raises(ValueError):
        OutlookModel(min_events=0)
    with pytest.raises(RuntimeError):
        OutlookModel().predict([0.0])
    with pytest.raises(ValueError):
        OutlookModel().fit(np.zeros((10, 2)), np.zeros(10))  # too few rows
    with pytest.raises(ValueError):
        OutlookModel().fit(np.zeros(40), np.zeros(40))  # not 2-D


# ========================================
# 📂 Walk-Forward Tests
# ========================================
def test_walk_forward_claims_no_skill_on_a_random_walk() -> None:
    report = walk_forward(build_features(random_walk()))
    assert report.predictions > 30
    assert report.skill(0.10) < 0.02  # nothing to predict, and the report must not pretend
    assert report.min_purge_gap >= 0  # every training label ended before its prediction
    assert 0.6 <= report.interval_coverage <= 0.95
    assert "no better than history" in report.summary() or "worse than" in report.summary()


def test_walk_forward_finds_skill_when_news_is_informative() -> None:
    s = random_walk()
    market_only = walk_forward(build_features(s))
    with_news = walk_forward(build_features(s, sentiment=informative_news(s)))
    assert with_news.skill(0.10) > 0.3
    assert with_news.skill(0.10) > market_only.skill(0.10) + 0.3
    assert "better than history" in with_news.summary()


def test_walk_forward_validation() -> None:
    with pytest.raises(ValueError):
        walk_forward(build_features(random_walk(300)), step=0)
    with pytest.raises(ValueError, match="no usable rows"):
        walk_forward(build_features(random_walk(200)))
    with pytest.raises(ValueError, match="not enough history"):
        walk_forward(build_features(random_walk(520)))


def test_walk_forward_skips_days_with_missing_news() -> None:
    s = random_walk()
    outage = (np.datetime64("2021-02-01"), np.datetime64("2022-03-01"))  # news feed down

    def patchy(day: Any) -> Any:
        if outage[0] <= day <= outage[1]:
            return types.SimpleNamespace(score=float("nan"), count=0)
        return no_news(day)

    data = build_features(s, sentiment=patchy)
    assert not data.usable[(s.dates >= outage[0]) & (s.dates <= outage[1])].any()
    report = walk_forward(data)  # rows with missing data are skipped, not filled with guesses
    assert report.predictions > 10


# ========================================
# 📂 Today's Outlook Tests
# ========================================
def test_model_outlook_has_no_look_ahead_and_never_asks_about_the_future() -> None:
    s = random_walk()
    as_of = "2023-06-30"
    asked: list[Any] = []

    def spy(day: Any) -> Any:
        asked.append(day)
        return no_news(day)

    later_doubled = PriceSeries(
        s.instrument, s.dates, np.where(s.dates > np.datetime64(as_of), s.close * 2, s.close)
    )
    a = model_outlook(s, as_of=as_of, sentiment=spy)
    b = model_outlook(later_doubled, as_of=as_of)
    assert max(asked) <= np.datetime64(as_of)
    assert a.threshold_probabilities.keys() == b.threshold_probabilities.keys()
    c = model_outlook(s, as_of=as_of)
    assert c == b  # identical inputs up to as_of give identical outlooks
    assert "news sentiment" in a.method and "news" not in c.method
    assert "Not financial advice" in a.summary()


def test_model_outlook_requires_history() -> None:
    with pytest.raises(ValueError, match="more than"):
        model_outlook(random_walk(200))
    with pytest.raises(ValueError, match="training rows"):
        model_outlook(random_walk(400))
