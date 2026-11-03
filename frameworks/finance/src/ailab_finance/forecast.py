# ==================================================================================================
# 📜 ailab-finance :: forecast
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/forecast.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.2
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    The model behind the Stock Outlook: combine market signals (momentum, volatility,
#    drawdown) with news sentiment to estimate next month's return range and the probability
#    of +10% / +25% / +50%, and prove, out of sample, whether it beats the historical baseline.
#
#  🔧 WHAT IT DOES
#    - build_features  : causal features for every day (market + optional news sentiment) and
#                        the forward-return label for training.
#    - OutlookModel    : ridge regression for the central return, residual quantiles for the
#                        range, one logistic model per threshold for the probabilities.
#    - walk_forward    : purged, expanding-window back-test scored with Brier scores against the
#                        historical base rate, plus interval coverage.
#    - model_outlook   : today's outlook from the model, in the same ReturnOutlook format as
#                        the baseline, so the two can be shown side by side.
#
#  🧠 DESIGN NOTES
#    - No look-ahead, three times over: features use data up to day t only; training rows are
#      purged so every label ended before the prediction date (i + horizon <= t); and the
#      sentiment provider is only ever asked about dates up to the decision date.
#    - Rare events are not forced. With fewer than `min_events` examples (a +50% month is very
#      rare), a logistic model would be noise, so that threshold falls back to the smoothed
#      base rate and the prediction says so (`modelled[t] is False`).
#    - Probabilities are made monotone: P(>= +25%) can never exceed P(>= +10%).
#    - Simple models on purpose. With roughly 1,000 overlapping monthly samples per stock,
#      regularised linear models are hard to beat and easy to explain; gradient boosting can be
#      plugged in later through `model_factory` and must then win the same walk-forward test.
#    - Brier skill = 1 - Brier(model) / Brier(baseline). Positive means better than history,
#      zero means no better, negative means worse. The report states this in words.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: features, OutlookModel, purged walk-forward, model_outlook.
#
#    V1.1 (2026-11-03) - Mathew Varghese
#      - Logistic solver moved to ailab-ml (shared with the credit scorecard, ADR-0003).
#
#    V1.2 (2026-11-03) - Mathew Varghese
#      - Rolling volatility from ailab-timeseries (shared with the climate framework, ADR-0003).
# ==================================================================================================
"""Outlook model (market + news sentiment) with purged walk-forward evaluation."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from ailab_ml import logistic_irls, sigmoid
from ailab_timeseries import rolling_std

from ailab_finance.outlook import ReturnOutlook
from ailab_finance.types import PriceSeries

FloatArray = npt.NDArray[np.float64]
WARMUP = 252  # trading days of history needed before the first feature row (one year)
MARKET_FEATURES = (
    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "volatility_21d",
    "volatility_63d",
    "drawdown_252d",
)
SENTIMENT_FEATURES = ("news_sentiment", "news_volume_log")


# ========================================
# 📂 Sentiment Input
# ========================================
class SentimentLike(Protocol):
    """Any object with a sentiment score and an article count (e.g. ailab_nlp.SentimentSummary)."""

    @property
    def score(self) -> float: ...

    @property
    def count(self) -> int: ...


# Called with a decision date; must return the news sentiment known on that date.
SentimentProvider = Callable[[np.datetime64], SentimentLike]


# ========================================
# 📂 Features
# ========================================
def market_features(close: npt.ArrayLike) -> FloatArray:
    """Causal market features per day; rows before WARMUP are NaN."""
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    out = np.full((n, len(MARKET_FEATURES)), np.nan)
    log_ret = np.concatenate([[np.nan], np.diff(np.log(c))])
    annualise = math.sqrt(252)
    volatility_21 = rolling_std(log_ret, 21) * annualise  # causal windows from ailab-timeseries
    volatility_63 = rolling_std(log_ret, 63) * annualise
    for i in range(WARMUP, n):
        out[i, 0] = c[i] / c[i - 21] - 1
        out[i, 1] = c[i] / c[i - 63] - 1
        out[i, 2] = c[i] / c[i - 126] - 1
        out[i, 3] = volatility_21[i]
        out[i, 4] = volatility_63[i]
        out[i, 5] = c[i] / np.max(c[i - 251 : i + 1]) - 1
    return out


@dataclass(frozen=True)
class FeatureSet:
    dates: npt.NDArray[np.datetime64]
    features: FloatArray  # (days, features); NaN where history is too short
    forward: FloatArray  # label: return over the next `horizon` days; NaN where unknown
    names: tuple[str, ...]
    horizon: int

    @property
    def usable(self) -> npt.NDArray[np.bool_]:
        """Rows with complete features and a known label."""
        return ~np.isnan(self.features).any(axis=1) & np.isfinite(self.forward)


def build_features(
    series: PriceSeries, *, horizon: int = 21, sentiment: SentimentProvider | None = None
) -> FeatureSet:
    """Features for every day of ``series`` and the forward-return label."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    c = series.close
    n = len(c)
    features = market_features(c)
    names: tuple[str, ...] = MARKET_FEATURES
    if sentiment is not None:
        news = np.full((n, len(SENTIMENT_FEATURES)), np.nan)
        for i in range(WARMUP, n):
            summary = sentiment(series.dates[i])  # asked only about day i: no look-ahead
            news[i] = (summary.score, math.log1p(summary.count))
        features = np.hstack([features, news])
        names = names + SENTIMENT_FEATURES
    forward = np.full(n, np.nan)
    if n > horizon:
        forward[: n - horizon] = c[horizon:] / c[:-horizon] - 1
    return FeatureSet(series.dates, features, forward, names, horizon)


# ========================================
# 📂 Model
# ========================================
@dataclass(frozen=True)
class Prediction:
    median_return: float
    interval: tuple[float, float]
    probabilities: dict[float, float]
    modelled: dict[float, bool]  # False: too few past events, smoothed base rate used instead


class OutlookModel:
    """Ridge for the central return, residual quantiles for the range, logistic per threshold."""

    def __init__(
        self,
        *,
        thresholds: Sequence[float] = (0.10, 0.25, 0.50),
        interval_level: float = 0.80,
        l2: float = 1.0,
        min_events: int = 10,
    ) -> None:
        if not 0 < interval_level < 1:
            raise ValueError("interval_level must be in (0, 1)")
        if l2 < 0 or min_events < 1:
            raise ValueError("l2 must be >= 0 and min_events >= 1")
        self.thresholds = tuple(sorted(thresholds))
        self.interval_level = interval_level
        self._l2 = l2
        self._min_events = min_events
        self._fitted = False

    def _design(self, x: FloatArray) -> FloatArray:
        scaled = (x - self._mean) / self._scale
        return np.hstack([np.ones((len(scaled), 1)), scaled])

    def fit(self, features: npt.ArrayLike, returns: npt.ArrayLike) -> OutlookModel:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(returns, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be 2-D with one row per return")
        if len(y) < 30 or np.isnan(x).any() or np.isnan(y).any():
            raise ValueError("need at least 30 complete rows to fit")
        # Standardise with training statistics only (test data must not influence scaling).
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        design = self._design(x)

        penalty = np.full(design.shape[1], self._l2)
        penalty[0] = 0.0
        self._beta = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ y)
        tail = (1 - self.interval_level) / 2
        self._residual_q = np.quantile(y - design @ self._beta, [tail, 0.5, 1 - tail])

        n = len(y)
        self._classifiers: dict[float, FloatArray] = {}
        self._base_rates: dict[float, float] = {}
        for t in self.thresholds:
            events = (y >= t).astype(np.float64)
            k = float(events.sum())
            self._base_rates[t] = (k + 0.5) / (n + 1)  # smoothed: never exactly 0 or 1
            if k >= self._min_events and n - k >= self._min_events:
                self._classifiers[t] = logistic_irls(design, events, l2=self._l2)
        self._fitted = True
        return self

    def predict(self, features: npt.ArrayLike) -> Prediction:
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        design = self._design(row)
        centre = float((design @ self._beta)[0])
        low, mid, high = (centre + q for q in self._residual_q)

        probabilities: dict[float, float] = {}
        modelled: dict[float, bool] = {}
        ceiling = 1.0
        for t in self.thresholds:  # ascending, so each target is capped by the easier one
            if t in self._classifiers:
                p = float(sigmoid(design @ self._classifiers[t])[0])
            else:
                p = self._base_rates[t]
            probabilities[t] = min(p, ceiling)
            modelled[t] = t in self._classifiers
            ceiling = probabilities[t]
        return Prediction(float(mid), (float(low), float(high)), probabilities, modelled)


# ========================================
# 📂 Walk-Forward Evaluation
# ========================================
@dataclass(frozen=True)
class WalkForwardReport:
    horizon: int
    predictions: int
    thresholds: tuple[float, ...]
    brier_model: dict[float, float]
    brier_baseline: dict[float, float]
    interval_level: float
    interval_coverage: float  # share of realised returns inside the predicted range
    min_purge_gap: int  # smallest gap (days) between a training label's end and a prediction

    def skill(self, threshold: float) -> float:
        """Brier skill vs the historical base rate: > 0 better, 0 equal, < 0 worse."""
        baseline = self.brier_baseline[threshold]
        return 1.0 - self.brier_model[threshold] / baseline if baseline > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Walk-forward test: {self.predictions} out-of-sample predictions, "
            f"{self.horizon}-day horizon, purge gap >= {self.min_purge_gap} days",
        ]
        for t in self.thresholds:
            s = self.skill(t)
            verdict = "better than" if s > 0.02 else "worse than" if s < -0.02 else "no better than"
            lines.append(
                f"P(return >= {t:+.0%}): Brier {self.brier_model[t]:.4f} vs baseline "
                f"{self.brier_baseline[t]:.4f}, skill {s:+.1%} ({verdict} history)"
            )
        lines.append(
            f"{self.interval_level:.0%} range contained the outcome {self.interval_coverage:.0%} "
            f"of the time (well calibrated if close to {self.interval_level:.0%})."
        )
        return "\n".join(lines)


def walk_forward(
    data: FeatureSet,
    *,
    step: int = 21,
    min_train: int = 252,
    model_factory: Callable[[], OutlookModel] = OutlookModel,
) -> WalkForwardReport:
    """Expanding-window back-test. At each test day t the model is trained only on rows whose
    label had ended by t (purging), then scored on the realised outcome at t."""
    if step < 1 or min_train < 30:
        raise ValueError("step must be >= 1 and min_train >= 30")
    h = data.horizon
    usable = data.usable
    rows = np.flatnonzero(usable)
    if len(rows) == 0:
        raise ValueError("no usable rows: the series is shorter than the feature warm-up")

    squared_model: dict[float, list[float]] = {}
    squared_base: dict[float, list[float]] = {}
    inside: list[bool] = []
    gaps: list[int] = []
    level = 0.0
    thresholds: tuple[float, ...] = ()
    for t in range(int(rows[0]) + min_train + h, len(data.dates), step):
        if not usable[t]:
            continue
        train = rows[rows + h <= t]
        if len(train) < min_train:
            continue
        model = model_factory().fit(data.features[train], data.forward[train])
        prediction = model.predict(data.features[t])
        actual = float(data.forward[t])
        thresholds, level = model.thresholds, model.interval_level
        for thr in thresholds:
            outcome = float(actual >= thr)
            base = float(np.mean(data.forward[train] >= thr))
            squared_model.setdefault(thr, []).append((prediction.probabilities[thr] - outcome) ** 2)
            squared_base.setdefault(thr, []).append((base - outcome) ** 2)
        low, high = prediction.interval
        inside.append(low <= actual <= high)
        gaps.append(int(t - (train[-1] + h)))

    if not inside:
        raise ValueError("not enough history for a walk-forward test; use a longer series")
    return WalkForwardReport(
        horizon=h,
        predictions=len(inside),
        thresholds=thresholds,
        brier_model={t: float(np.mean(v)) for t, v in squared_model.items()},
        brier_baseline={t: float(np.mean(v)) for t, v in squared_base.items()},
        interval_level=level,
        interval_coverage=float(np.mean(inside)),
        min_purge_gap=min(gaps),
    )


# ========================================
# 📂 Today's Outlook
# ========================================
def model_outlook(
    series: PriceSeries,
    *,
    as_of: str | None = None,
    horizon: int = 21,
    sentiment: SentimentProvider | None = None,
    model: OutlookModel | None = None,
    min_train: int = 252,
) -> ReturnOutlook:
    """Model-based outlook as of ``as_of``, in the same format as historical_outlook()."""
    history = series if as_of is None else series.until(as_of)
    data = build_features(history, horizon=horizon, sentiment=sentiment)
    today = len(history) - 1
    if np.isnan(data.features[today]).any():
        raise ValueError(f"need more than {WARMUP} prices up to the decision date")
    train = np.flatnonzero(data.usable)  # labels known within `history`, so all ended by today
    if len(train) < min_train:
        raise ValueError(f"only {len(train)} training rows; at least {min_train} required")

    fitted = (model or OutlookModel()).fit(data.features[train], data.forward[train])
    prediction = fitted.predict(data.features[today])
    method = "model: ridge + logistic, market features"
    if sentiment is not None:
        method += " + news sentiment"
    return ReturnOutlook(
        symbol=series.instrument.symbol,
        as_of=history.end,
        horizon_days=horizon,
        samples=len(train),
        median_return=prediction.median_return,
        interval=prediction.interval,
        interval_level=fitted.interval_level,
        threshold_probabilities=prediction.probabilities,
        method=method,
    )
