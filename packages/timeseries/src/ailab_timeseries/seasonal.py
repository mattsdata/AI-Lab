# ==================================================================================================
# 📜 ailab-timeseries :: seasonal
# --------------------------------------------------------------------------------------------------
#  File         : packages/timeseries/src/ailab_timeseries/seasonal.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Most real series have a calendar rhythm (temperature, energy demand, hospital admissions,
#    retail sales). A 30 degree day means something different in January and in May. This module
#    compares each day with what is normal for that time of year.
#
#  🔧 WHAT IT DOES
#    - day_of_year       : 0..364 with 29 February folded onto 28 February.
#    - daily_climatology : the normal value for each calendar day over a baseline period,
#                          smoothed over +/- `half_window` days (circular across New Year).
#    - seasonal_anomalies: each value minus its calendar-day normal.
#    - seasonal_threshold_flags : True where a value exceeds the baseline quantile for its
#                          calendar day (the basis of indices such as "hot days").
#
#  🧠 DESIGN NOTES
#    - The baseline is explicit (e.g. 1961-1990). Normals computed from the full record would
#      absorb the very change being measured and hide it.
#    - Smoothing across neighbouring days gives each calendar day ~30 x 31 samples instead of 30,
#      making thresholds stable; the window wraps around year end.
#    - Leap days are folded, not dropped, so no observation is lost.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: calendar-day climatology, anomalies and threshold flags.
# ==================================================================================================
"""Calendar-day normals, anomalies and seasonal thresholds."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
_CUMULATIVE_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def day_of_year(dates: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Day of year 0..364 on a 365-day calendar (29 February counts as 28 February)."""
    d = np.asarray(dates, dtype="datetime64[D]")
    month_start = d.astype("datetime64[M]")
    month = month_start.astype(np.int64) % 12
    day = (d - month_start.astype("datetime64[D]")).astype(np.int64)
    day = np.where((month == 1) & (day == 28), 27, day)  # fold 29 Feb onto 28 Feb
    out: npt.NDArray[np.int64] = _CUMULATIVE_DAYS[month] + day
    return out


def _baseline_mask(dates: npt.ArrayLike, baseline: tuple[int, int]) -> npt.NDArray[np.bool_]:
    years = np.asarray(dates, dtype="datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    return (years >= baseline[0]) & (years <= baseline[1])


def _circular_distance(doy: npt.NDArray[np.int64], target: int) -> npt.NDArray[np.int64]:
    diff = np.abs(doy - target)
    out: npt.NDArray[np.int64] = np.minimum(diff, 365 - diff)
    return out


def daily_climatology(
    dates: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    baseline: tuple[int, int],
    half_window: int = 7,
) -> FloatArray:
    """Normal value for each of the 365 calendar days over the baseline years."""
    x = np.asarray(values, dtype=np.float64)
    in_base = _baseline_mask(dates, baseline) & ~np.isnan(x)
    if not in_base.any():
        raise ValueError(f"no data inside the baseline {baseline[0]}-{baseline[1]}")
    doy = day_of_year(dates)[in_base]
    sums: FloatArray = np.asarray(
        np.bincount(doy, weights=x[in_base], minlength=365), dtype=np.float64
    )
    counts: FloatArray = np.asarray(np.bincount(doy, minlength=365), dtype=np.float64)
    kernel = np.ones(2 * half_window + 1)

    def circular(series: FloatArray) -> FloatArray:
        wrapped = np.concatenate([series[-half_window:], series, series[:half_window]])
        out: FloatArray = np.convolve(wrapped, kernel, mode="valid")
        return out

    smoothed_counts = circular(counts)
    out: FloatArray = np.where(
        smoothed_counts > 0, circular(sums) / np.maximum(smoothed_counts, 1), np.nan
    )
    return out


def seasonal_anomalies(
    dates: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    baseline: tuple[int, int],
    half_window: int = 7,
) -> FloatArray:
    """Each value minus the baseline normal for its calendar day."""
    normals = daily_climatology(dates, values, baseline=baseline, half_window=half_window)
    out: FloatArray = np.asarray(values, dtype=np.float64) - normals[day_of_year(dates)]
    return out


def seasonal_threshold_flags(
    dates: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    baseline: tuple[int, int],
    quantile: float = 0.9,
    half_window: int = 7,
) -> npt.NDArray[np.bool_]:
    """True where a value exceeds the baseline ``quantile`` for its calendar day."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    x = np.asarray(values, dtype=np.float64)
    doy_all = day_of_year(dates)
    in_base = _baseline_mask(dates, baseline) & ~np.isnan(x)
    if not in_base.any():
        raise ValueError(f"no data inside the baseline {baseline[0]}-{baseline[1]}")
    base_doy, base_x = doy_all[in_base], x[in_base]
    thresholds = np.array(
        [
            np.quantile(base_x[_circular_distance(base_doy, d) <= half_window], quantile)
            for d in range(365)
        ]
    )
    flags: npt.NDArray[np.bool_] = ~np.isnan(x) & (x > thresholds[doy_all])
    return flags
