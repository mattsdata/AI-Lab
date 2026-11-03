# ==================================================================================================
# 📜 ailab-climate :: indices
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/climate/src/ailab_climate/indices.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Turn decades of daily weather into a handful of yearly numbers that describe a changing
#    climate in terms people feel: how warm, how many unusually hot days and nights, how much
#    rain, how often it pours.
#
#  🔧 WHAT IT DOES
#    - annual_indices : per calendar year,
#        mean_temperature  annual mean of (Tmax + Tmin) / 2                        [deg C]
#        hot_days          days with Tmax above the baseline 90th percentile      [days]
#        warm_nights       days with Tmin above the baseline 90th percentile      [days]
#        rainfall_total    total precipitation                                     [mm]
#        heavy_rain_days   days with at least 20 mm                                [days]
#        wettest_day       largest one-day total                                   [mm]
#    - AnnualIndices  : the result, with each index's name and unit.
#
#  🧠 DESIGN NOTES
#    - Definitions follow the internationally used ETCCDI climate change indices in spirit
#      (TX90p / TN90p as day counts, R20mm, Rx1day), so results are comparable with literature.
#    - Percentile thresholds are calendar-day specific and come from the baseline only; a
#      30 deg C day is "hot" in some months and ordinary in others.
#    - A year with less than `min_coverage` of its days observed is NaN for every index.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: six annual indices with coverage rules.
# ==================================================================================================
"""Annual climate indices from daily weather."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ailab_timeseries import aggregate, seasonal_threshold_flags

from ailab_climate.data import DailyWeather

FloatArray = npt.NDArray[np.float64]

INDEX_UNITS: dict[str, tuple[str, str]] = {
    "mean_temperature": ("Annual mean temperature", "deg C"),
    "hot_days": ("Hot days (Tmax above baseline 90th percentile)", "days"),
    "warm_nights": ("Warm nights (Tmin above baseline 90th percentile)", "days"),
    "rainfall_total": ("Annual rainfall", "mm"),
    "heavy_rain_days": ("Heavy rain days (>= 20 mm)", "days"),
    "wettest_day": ("Wettest day of the year", "mm"),
}


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True)
class AnnualIndices:
    years: npt.NDArray[np.int64]
    values: dict[str, FloatArray]  # index name -> one value per year (NaN if not enough data)
    baseline: tuple[int, int]

    def label(self, name: str) -> str:
        return INDEX_UNITS[name][0]

    def unit(self, name: str) -> str:
        return INDEX_UNITS[name][1]


# ========================================
# 📂 Computation
# ========================================
def annual_indices(
    weather: DailyWeather,
    *,
    baseline: tuple[int, int] = (1961, 1990),
    min_coverage: float = 0.9,
    heavy_rain_mm: float = 20.0,
) -> AnnualIndices:
    """Six annual indices; thresholds come from ``baseline`` years only."""
    d = weather.dates

    def yearly(values: FloatArray, how: str) -> tuple[npt.NDArray[np.int64], FloatArray]:
        starts, out = aggregate(d, values, freq="year", how=how, min_coverage=min_coverage)  # type: ignore[arg-type]
        years: npt.NDArray[np.int64] = starts.astype("datetime64[Y]").astype(np.int64) + 1970
        return years, out

    def as_count(flags: npt.NDArray[np.bool_], source: FloatArray) -> FloatArray:
        out: FloatArray = np.where(np.isnan(source), np.nan, flags.astype(np.float64))
        return out

    hot = seasonal_threshold_flags(d, weather.tmax, baseline=baseline, quantile=0.9)
    warm = seasonal_threshold_flags(d, weather.tmin, baseline=baseline, quantile=0.9)
    rain = weather.precipitation
    years, mean_temperature = yearly(weather.tmean, "mean")
    values = {
        "mean_temperature": mean_temperature,
        "hot_days": yearly(as_count(hot, weather.tmax), "sum")[1],
        "warm_nights": yearly(as_count(warm, weather.tmin), "sum")[1],
        "rainfall_total": yearly(rain, "sum")[1],
        "heavy_rain_days": yearly(as_count(rain >= heavy_rain_mm, rain), "sum")[1],
        "wettest_day": yearly(rain, "max")[1],
    }
    return AnnualIndices(years, values, baseline)
