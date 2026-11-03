# ==================================================================================================
# 📜 ailab-timeseries :: core
# --------------------------------------------------------------------------------------------------
#  File         : packages/timeseries/src/ailab_timeseries/core.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Time-series building blocks that every domain needs and that are easy to get subtly wrong:
#    trailing windows that never peek ahead, calendar aggregation that refuses half-empty
#    periods, and trends whose uncertainty is not overstated.
#
#  🔧 WHAT IT DOES
#    - rolling_mean, rolling_std : trailing (causal) windows, NaN-aware, with min_periods.
#    - aggregate                 : daily values to month or year totals / means / maxima, with a
#                                  minimum data-coverage rule per period.
#    - linear_trend              : OLS slope per chosen unit (e.g. per decade) with a confidence
#                                  interval widened for lag-1 autocorrelation.
#    - Trend                     : the result, with a plain-language `is_significant`.
#
#  🧠 DESIGN NOTES
#    - Windows end at the current point: value i uses data up to i only, so the same function is
#      safe in back-tests (finance) and in real-time monitoring (climate, operations).
#    - Coverage rule: a month with 10 of 31 days recorded is NaN, not a mean of 10 days; partial
#      periods silently bias totals (a rain total from 10 days looks like a drought).
#    - Autocorrelated residuals (common in climate and economic series) make plain OLS intervals
#      too narrow. The effective sample size n_eff = n(1 - r1)/(1 + r1) widens them (Santer et
#      al., 2000). Student-t quantiles use the Cornish-Fisher expansion, avoiding SciPy.
#    - Extracted on second use: rolling statistics began in the finance outlook features and
#      moved here when the climate framework needed them (ADR-0003).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: rolling statistics, coverage-aware aggregation, honest trends.
# ==================================================================================================
"""Causal rolling statistics, coverage-aware aggregation and honest linear trends.

Example:
    >>> import numpy as np
    >>> rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2).tolist()
    [nan, 1.5, 2.5, 3.5]
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

FloatArray = npt.NDArray[np.float64]
DateArray = npt.NDArray[np.datetime64]


# ========================================
# 📂 Rolling Windows
# ========================================
def _windows(values: npt.ArrayLike, window: int, min_periods: int | None) -> tuple[FloatArray, int]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or window < 1:
        raise ValueError("values must be 1-D and window >= 1")
    need = window if min_periods is None else min_periods
    if not 1 <= need <= window:
        raise ValueError("min_periods must be between 1 and window")
    padded = np.concatenate([np.full(window - 1, np.nan), x])
    return sliding_window_view(padded, window), need


def rolling_mean(
    values: npt.ArrayLike, window: int, *, min_periods: int | None = None
) -> FloatArray:
    """Mean of the last ``window`` values (inclusive); NaN until ``min_periods`` values exist."""
    windows, need = _windows(values, window, min_periods)
    counts = np.sum(~np.isnan(windows), axis=1)
    sums = np.nansum(windows, axis=1)
    out: FloatArray = np.where(counts >= need, sums / np.maximum(counts, 1), np.nan)
    return out


def rolling_std(
    values: npt.ArrayLike, window: int, *, ddof: int = 1, min_periods: int | None = None
) -> FloatArray:
    """Standard deviation of the last ``window`` values (inclusive), NaN-aware."""
    windows, need = _windows(values, window, min_periods)
    counts = np.sum(~np.isnan(windows), axis=1)
    ok = counts >= max(need, ddof + 1)
    out = np.full(len(windows), np.nan)
    if ok.any():
        mean = np.nansum(windows[ok], axis=1) / counts[ok]
        squares = np.nansum((windows[ok] - mean[:, None]) ** 2, axis=1)
        out[ok] = np.sqrt(squares / (counts[ok] - ddof))
    return out


# ========================================
# 📂 Calendar Aggregation
# ========================================
Freq = Literal["month", "year"]
How = Literal["mean", "sum", "max", "min"]


def aggregate(
    dates: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    freq: Freq = "year",
    how: How = "mean",
    min_coverage: float = 0.8,
) -> tuple[DateArray, FloatArray]:
    """Aggregate daily values per calendar month or year.

    A period whose share of days with data is below ``min_coverage`` becomes NaN. Returns the
    period start dates and the aggregated values.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    d = np.asarray(dates, dtype="datetime64[D]")
    x = np.asarray(values, dtype=np.float64)
    if d.shape != x.shape or d.ndim != 1:
        raise ValueError("dates and values must be 1-D arrays of equal length")
    unit = "M" if freq == "month" else "Y"
    step = np.timedelta64(1, "M") if freq == "month" else np.timedelta64(1, "Y")
    periods = d.astype(f"datetime64[{unit}]")
    starts = np.unique(periods)
    out = np.full(len(starts), np.nan)
    reducers: dict[str, Callable[[FloatArray], float]] = {
        "mean": lambda a: float(np.mean(a)),
        "sum": lambda a: float(np.sum(a)),
        "max": lambda a: float(np.max(a)),
        "min": lambda a: float(np.min(a)),
    }
    reducer = reducers[how]
    for k, start in enumerate(starts):
        first = start.astype("datetime64[D]")
        following = (start + step).astype("datetime64[D]")
        length = int((following - first).astype(int))
        member = x[periods == start]
        present = member[~np.isnan(member)]
        if len(present) >= min_coverage * length:
            out[k] = reducer(present)
    return starts.astype("datetime64[D]"), out


# ========================================
# 📂 Trend with Honest Uncertainty
# ========================================
def t_quantile(p: float, dof: float) -> float:
    """Student-t quantile via the Cornish-Fisher expansion (accurate to ~1e-3 for dof >= 3)."""
    z = NormalDist().inv_cdf(p)
    v = dof
    return (
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


@dataclass(frozen=True, slots=True)
class Trend:
    slope: float  # change per `per` units of x (e.g. per decade)
    low: float  # confidence interval
    high: float
    level: float
    n: int
    n_effective: float  # sample size after allowing for autocorrelation
    lag1_autocorrelation: float

    @property
    def is_significant(self) -> bool:
        """True if the interval excludes zero (the direction of change is clear)."""
        return self.low > 0 or self.high < 0


def linear_trend(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    per: float = 1.0,
    level: float = 0.95,
    adjust_autocorrelation: bool = True,
) -> Trend:
    """OLS trend of y on x, reported per ``per`` units of x, NaN pairs ignored."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    keep = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[keep], ys[keep]
    n = len(xs)
    if n < 4 or np.ptp(xs) == 0:
        raise ValueError("need at least 4 points with varying x")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    centred = xs - xs.mean()
    slope = float(np.sum(centred * (ys - ys.mean())) / np.sum(centred**2))
    residuals = ys - ys.mean() - slope * centred
    r1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if np.std(residuals) > 0 else 0.0
    n_eff = n * (1 - r1) / (1 + r1) if adjust_autocorrelation and r1 > 0 else float(n)
    n_eff = max(n_eff, 3.0)
    variance = np.sum(residuals**2) / (n - 2)
    se = math.sqrt(variance / np.sum(centred**2)) * math.sqrt((n - 2) / max(n_eff - 2, 1.0))
    margin = t_quantile(0.5 + level / 2, max(n_eff - 2, 1.0)) * se
    return Trend(
        slope=slope * per,
        low=(slope - margin) * per,
        high=(slope + margin) * per,
        level=level,
        n=n,
        n_effective=n_eff,
        lag1_autocorrelation=r1,
    )
