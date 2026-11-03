# ==================================================================================================
# 📜 ailab-finance
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.3
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
#
#    V1.1 (2026-11-03) - Mathew Varghese
#      - Added the outlook model: market + news features, purged walk-forward evaluation.
#
#    V1.2 (2026-11-03) - Mathew Varghese
#      - Added SEC EDGAR fundamentals: point-in-time credit inputs with provenance.
#
#    V1.3 (2026-11-03) - Mathew Varghese
#      - Added credit scorecard: WoE, points, reason codes, validation pack, fairness check.
# ==================================================================================================
"""ailab-finance: market analytics and corporate early warning (research use only)."""

# ========================================
# 📂 Imports
# ========================================
from ailab_finance.credit import DefaultEstimate, naive_distance_to_default, warning_flag
from ailab_finance.data import CsvSource, DataSource, YahooSource
from ailab_finance.forecast import (
    FeatureSet,
    OutlookModel,
    Prediction,
    SentimentLike,
    WalkForwardReport,
    build_features,
    model_outlook,
    walk_forward,
)
from ailab_finance.fundamentals import CreditInputs, EdgarClient, Fact, latest_fact
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
from ailab_finance.scorecard import (
    Explanation,
    GroupOutcome,
    Scorecard,
    ValidationReport,
    WoeBinning,
    group_outcomes,
    validate,
)
from ailab_finance.types import Instrument, Money, PriceSeries

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.3.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "DISCLAIMER",
    "CreditInputs",
    "CsvSource",
    "DataSource",
    "DefaultEstimate",
    "EdgarClient",
    "Explanation",
    "Fact",
    "FeatureSet",
    "GroupOutcome",
    "Instrument",
    "Money",
    "OutlookModel",
    "Prediction",
    "PriceSeries",
    "ReturnOutlook",
    "Scorecard",
    "SentimentLike",
    "ValidationReport",
    "WalkForwardReport",
    "WoeBinning",
    "YahooSource",
    "annualised_volatility",
    "build_features",
    "forward_returns",
    "group_outcomes",
    "historical_outlook",
    "historical_var",
    "latest_fact",
    "log_returns",
    "max_drawdown",
    "model_outlook",
    "naive_distance_to_default",
    "sharpe_ratio",
    "simple_returns",
    "sortino_ratio",
    "validate",
    "walk_forward",
    "warning_flag",
]
