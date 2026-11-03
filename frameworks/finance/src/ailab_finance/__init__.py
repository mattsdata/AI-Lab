# ==================================================================================================
# 📜 ailab-finance
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab finance framework: market analytics and a corporate early
#    warning system combining equity outlook and credit risk for listed companies.
#
#  🔧 WHAT IT DOES
#    Re-exports the public API: types, data sources, metrics, credit risk and outlook.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: foundations of the Corporate Early Warning System.
# ==================================================================================================
"""ailab-finance: market analytics and corporate early warning (research use only)."""

# ========================================
# 📂 Imports
# ========================================
from ailab_finance.credit import DefaultEstimate, naive_distance_to_default, warning_flag
from ailab_finance.data import CsvSource, DataSource, YahooSource
from ailab_finance.metrics import (
    annualised_volatility,
    forward_returns,
    historical_var,
    log_returns,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
)
from ailab_finance.outlook import DISCLAIMER, ReturnOutlook, historical_outlook
from ailab_finance.types import Instrument, Money, PriceSeries

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "DISCLAIMER",
    "CsvSource",
    "DataSource",
    "DefaultEstimate",
    "Instrument",
    "Money",
    "PriceSeries",
    "ReturnOutlook",
    "YahooSource",
    "annualised_volatility",
    "forward_returns",
    "historical_outlook",
    "historical_var",
    "log_returns",
    "max_drawdown",
    "naive_distance_to_default",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "warning_flag",
]
