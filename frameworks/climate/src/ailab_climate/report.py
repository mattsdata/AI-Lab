# ==================================================================================================
# 📜 ailab-climate :: report
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/climate/src/ailab_climate/report.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A plain-language climate change summary for a place: what changed, by how much per decade,
#    how sure we can be, and how the place compares with the global picture. Written so that a
#    council officer, a journalist or a student can read it, and a scientist can check it.
#
#  🔧 WHAT IT DOES
#    - place_report     : trends per decade for every annual index, with 95% intervals, and the
#                         last decade compared with the baseline.
#    - global_context   : how much warmer the world's last ten years were than its baseline.
#    - ClimateReport    : the result; summary() prints the report with its caveats.
#
#  🧠 DESIGN NOTES
#    - Every trend comes with an interval, and the verdict is worded from it: "clear increase",
#      "clear decrease" or "no clear trend". Rainfall in the tropics is noisy; saying so is part
#      of the result, not a failure of the method.
#    - Intervals allow for year-to-year autocorrelation (ailab-timeseries.linear_trend).
#    - The report states its own limits: reanalysis grid, not a station; past trends are not
#      projections. Projections need climate model scenarios (e.g. CMIP6), a planned extension.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: place_report, global_context, ClimateReport.
# ==================================================================================================
"""Plain-language climate change report for a place."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ailab_timeseries import Trend, linear_trend

from ailab_climate.data import DailyWeather, MonthlyAnomalies
from ailab_climate.indices import INDEX_UNITS, AnnualIndices, annual_indices

CAVEATS = (
    "Data: ERA5 reanalysis via Open-Meteo, a ~25 km grid cell, not a single weather station.",
    "Trends describe the past. They are not projections; those need climate model scenarios.",
    "Intervals are 95% and allow for year-to-year autocorrelation.",
)


# ========================================
# 📂 Result Types
# ========================================
@dataclass(frozen=True)
class IndexTrend:
    name: str
    label: str
    unit: str
    trend: Trend  # per decade
    baseline_mean: float
    recent_mean: float  # last ten years with data

    @property
    def verdict(self) -> str:
        if self.trend.low > 0:
            return "clear increase"
        if self.trend.high < 0:
            return "clear decrease"
        return "no clear trend (the interval includes zero)"


@dataclass(frozen=True)
class ClimateReport:
    place: str
    latitude: float
    longitude: float
    first_year: int
    last_year: int
    baseline: tuple[int, int]
    trends: list[IndexTrend]
    source: str
    global_context: str | None = None

    def trend(self, name: str) -> IndexTrend:
        return next(t for t in self.trends if t.name == name)

    def summary(self) -> str:
        lines = [
            f"Climate change at {self.place} ({self.latitude:.2f}, {self.longitude:.2f}), "
            f"{self.first_year}-{self.last_year}, baseline {self.baseline[0]}-{self.baseline[1]}",
            "",
        ]
        for t in self.trends:
            change = t.recent_mean - t.baseline_mean
            lines.append(
                f"{t.label}: {t.trend.slope:+.2f} {t.unit} per decade "
                f"(95% CI {t.trend.low:+.2f} to {t.trend.high:+.2f}), {t.verdict}. "
                f"Last 10 years vs baseline: {change:+.2f} {t.unit}."
            )
        if self.global_context:
            lines += ["", self.global_context]
        lines += ["", *CAVEATS, f"Source: {self.source}"]
        return "\n".join(lines)


# ========================================
# 📂 Report Builders
# ========================================
def _index_trend(indices: AnnualIndices, name: str) -> IndexTrend:
    years = indices.years.astype(np.float64)
    values = indices.values[name]
    in_base = (indices.years >= indices.baseline[0]) & (indices.years <= indices.baseline[1])
    present = np.flatnonzero(~np.isnan(values))
    if not np.any(in_base & ~np.isnan(values)) or len(present) < 10:
        raise ValueError(f"{name}: need data in the baseline and at least 10 years overall")
    recent = present[-10:]
    label, unit = INDEX_UNITS[name]
    return IndexTrend(
        name=name,
        label=label,
        unit=unit,
        trend=linear_trend(years, values, per=10),
        baseline_mean=float(np.nanmean(values[in_base])),
        recent_mean=float(np.mean(values[recent])),
    )


def global_context(global_anomalies: MonthlyAnomalies) -> str:
    """'The world's last ten complete years averaged +X deg C above its baseline.'"""
    # Monthly input: a year counts only if all 12 months are present. (ailab_timeseries.aggregate
    # measures coverage in days, so it is for daily data, not this monthly record.)
    years = global_anomalies.dates.astype("datetime64[Y]").astype(np.int64) + 1970
    annual = [
        (int(y), float(values.mean()))
        for y in np.unique(years)
        if len(values := global_anomalies.anomalies[years == y]) == 12
        and not np.isnan(values).any()
    ]
    if len(annual) < 10:
        raise ValueError("need at least ten complete years of global data")
    recent = annual[-10:]
    mean = sum(v for _, v in recent) / len(recent)
    b0, b1 = global_anomalies.baseline
    return (
        f"Global context: {recent[0][0]}-{recent[-1][0]} averaged {mean:+.2f} deg C above the "
        f"{b0}-{b1} average ({global_anomalies.source})."
    )


def place_report(
    weather: DailyWeather,
    *,
    baseline: tuple[int, int] = (1961, 1990),
    global_anomalies: MonthlyAnomalies | None = None,
) -> ClimateReport:
    """Climate change summary for one location from its daily weather history."""
    indices = annual_indices(weather, baseline=baseline)
    present = indices.years[~np.isnan(indices.values["mean_temperature"])]
    if len(present) == 0:
        raise ValueError("no year has enough data")
    return ClimateReport(
        place=weather.place,
        latitude=weather.latitude,
        longitude=weather.longitude,
        first_year=int(present[0]),
        last_year=int(present[-1]),
        baseline=baseline,
        trends=[_index_trend(indices, name) for name in INDEX_UNITS],
        source=weather.source,
        global_context=global_context(global_anomalies) if global_anomalies else None,
    )
