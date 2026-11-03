# ==================================================================================================
# 📜 ailab-climate
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/climate/src/ailab_climate/__init__.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab climate framework: how has the climate of a place changed, how
#    sure are we, and how does it compare with the world?
#
#  🔧 WHAT IT DOES
#    Re-exports data sources, annual climate indices and the place report.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: Open-Meteo and GISTEMP data, six indices, place report.
# ==================================================================================================
"""ailab-climate: local climate change analysis with honest uncertainty."""

# ========================================
# 📂 Imports
# ========================================
from ailab_climate.data import (
    DailyWeather,
    GistempClient,
    MonthlyAnomalies,
    OpenMeteoClient,
    parse_gistemp,
)
from ailab_climate.indices import INDEX_UNITS, AnnualIndices, annual_indices
from ailab_climate.report import ClimateReport, IndexTrend, global_context, place_report

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "INDEX_UNITS",
    "AnnualIndices",
    "ClimateReport",
    "DailyWeather",
    "GistempClient",
    "IndexTrend",
    "MonthlyAnomalies",
    "OpenMeteoClient",
    "annual_indices",
    "global_context",
    "parse_gistemp",
    "place_report",
]
