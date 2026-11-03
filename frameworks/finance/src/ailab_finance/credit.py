# ==================================================================================================
# 📜 ailab-finance :: credit
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/credit.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Corporate credit early warning from market data. A listed borrower's share price and its
#    volatility react to trouble long before accounts are filed, so they give a bank an early,
#    daily signal of rising default risk across a loan book.
#
#  🔧 WHAT IT DOES
#    - naive_distance_to_default : the "naive" Merton distance to default and the implied
#                                  default probability (Bharath & Shumway, 2008).
#    - warning_flag              : GREEN / AMBER / RED from the default probability.
#
#  🧠 DESIGN NOTES
#    - Merton's model treats equity as a call option on the firm's assets: default happens if
#      asset value falls below debt. Distance to default (DD) is how many standard deviations
#      the firm is from that point. The full model solves two non-linear equations; the naive
#      version uses a closed form that, in the published comparison, forecasts default about
#      as well. It needs no default history, which is scarce and usually paid-for data.
#    - Debt is the default point. The common KMV convention is short-term liabilities plus half
#      of long-term debt; the caller supplies the figure so the convention is explicit.
#    - The probability is model-implied, not a calibrated real-world PD. Banks map DD to PD
#      using their own default history; this module reports DD alongside it for that reason.
#    - Flag thresholds are illustrative defaults. In a bank they come from the credit risk
#      appetite statement, so they are parameters, not constants.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: naive Merton distance to default, warning flags.
# ==================================================================================================
"""Market-based corporate credit risk (naive Merton distance to default).

Example:
    >>> est = naive_distance_to_default(
    ...     equity_value=100, debt=100, equity_volatility=0.40, expected_return=0.05
    ... )
    >>> round(est.distance_to_default, 2)
    2.56
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True, slots=True)
class DefaultEstimate:
    distance_to_default: float  # standard deviations between asset value and the default point
    default_probability: float  # model-implied probability of default over the horizon
    asset_volatility: float  # estimated annual volatility of the firm's total assets


# ========================================
# 📂 Distance to Default
# ========================================
def naive_distance_to_default(
    *,
    equity_value: float,
    debt: float,
    equity_volatility: float,
    expected_return: float,
    horizon_years: float = 1.0,
) -> DefaultEstimate:
    """Naive Merton DD (Bharath & Shumway, 2008).

    Args:
        equity_value: market capitalisation.
        debt: face value of debt used as the default point (same currency as equity_value).
        equity_volatility: annualised volatility of the share price, e.g. 0.35.
        expected_return: proxy for asset drift, usually the stock's return over the past year.
        horizon_years: forecast horizon, normally 1.
    """
    if equity_value <= 0 or debt <= 0:
        raise ValueError("equity_value and debt must be positive")
    if equity_volatility <= 0 or horizon_years <= 0:
        raise ValueError("equity_volatility and horizon_years must be positive")

    # Naive debt volatility: 5% base plus a quarter of equity volatility
    # (a riskier firm has riskier debt).
    debt_volatility = 0.05 + 0.25 * equity_volatility
    firm_value = equity_value + debt
    asset_volatility = (equity_value / firm_value) * equity_volatility + (
        debt / firm_value
    ) * debt_volatility

    numerator = (
        math.log(firm_value / debt) + (expected_return - 0.5 * asset_volatility**2) * horizon_years
    )
    distance = numerator / (asset_volatility * math.sqrt(horizon_years))
    return DefaultEstimate(distance, NormalDist().cdf(-distance), asset_volatility)


# ========================================
# 📂 Early Warning Flag
# ========================================
def warning_flag(estimate: DefaultEstimate, *, amber: float = 0.01, red: float = 0.05) -> str:
    """GREEN below ``amber``, AMBER from ``amber``, RED from ``red`` (default probability)."""
    if not 0 < amber < red < 1:
        raise ValueError("thresholds must satisfy 0 < amber < red < 1")
    if estimate.default_probability >= red:
        return "RED"
    if estimate.default_probability >= amber:
        return "AMBER"
    return "GREEN"
