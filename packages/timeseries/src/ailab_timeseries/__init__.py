# ==================================================================================================
# 📜 ailab-timeseries
# --------------------------------------------------------------------------------------------------
#  File         : packages/timeseries/src/ailab_timeseries/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab time-series package, shared by finance (market features) and
#    climate (trends, anomalies, extremes), and ready for any other domain.
#
#  🔧 WHAT IT DOES
#    Re-exports rolling statistics, calendar aggregation, trends and seasonal tools.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release, extracted from finance on second use (climate), per ADR-0003.
# ==================================================================================================
"""ailab-timeseries: causal, coverage-aware and uncertainty-honest time-series tools."""

# ========================================
# 📂 Imports
# ========================================
from ailab_timeseries.core import (
    Trend,
    aggregate,
    linear_trend,
    rolling_mean,
    rolling_std,
    t_quantile,
)
from ailab_timeseries.seasonal import (
    daily_climatology,
    day_of_year,
    seasonal_anomalies,
    seasonal_threshold_flags,
)

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "Trend",
    "aggregate",
    "daily_climatology",
    "day_of_year",
    "linear_trend",
    "rolling_mean",
    "rolling_std",
    "seasonal_anomalies",
    "seasonal_threshold_flags",
    "t_quantile",
]
