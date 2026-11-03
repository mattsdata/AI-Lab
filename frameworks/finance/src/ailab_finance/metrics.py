# ==================================================================================================
# 📜 ailab-finance :: metrics
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/metrics.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    The standard return and risk measures used across investment and risk teams, implemented
#    once, tested against hand-calculated values, and shared by every finance scenario.
#
#  🔧 WHAT IT DOES
#    - simple_returns, log_returns, forward_returns
#    - annualised_volatility, sharpe_ratio, sortino_ratio
#    - max_drawdown, historical_var
#
#  🧠 DESIGN NOTES
#    - Pure functions on arrays: no hidden state, trivially testable, usable on any series.
#    - Sample standard deviation (ddof=1), the convention in finance reports.
#    - Ratios raise on zero volatility instead of returning inf or 0: a "Sharpe of infinity"
#      in a report is a bug that should surface, not a number to rank on.
#    - VaR is historical (empirical quantile), so it makes no normality assumption and keeps
#      the fat tails real markets have. It is reported as a positive loss fraction.
#    - 252 trading days per year is the default; pass 12 for monthly data, 52 for weekly.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: returns, volatility, Sharpe, Sortino, drawdown, VaR.
# ==================================================================================================
"""Return and risk metrics on price or return arrays.

Example:
    >>> import numpy as np
    >>> round(max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])), 2)
    -0.25
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
TRADING_DAYS = 252


def _as_array(values: npt.ArrayLike, minimum: int, what: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < minimum:
        raise ValueError(f"{what} needs a 1-D array with at least {minimum} values")
    return array


# ========================================
# 📂 Returns
# ========================================
def simple_returns(close: npt.ArrayLike) -> FloatArray:
    """Period-over-period returns: p[t] / p[t-1] - 1."""
    prices = _as_array(close, 2, "simple_returns")
    return prices[1:] / prices[:-1] - 1.0


def log_returns(close: npt.ArrayLike) -> FloatArray:
    """Continuously compounded returns: ln(p[t] / p[t-1]). These add up across time."""
    prices = _as_array(close, 2, "log_returns")
    return np.diff(np.log(prices))


def forward_returns(close: npt.ArrayLike, horizon: int) -> FloatArray:
    """Return over the next ``horizon`` periods from each start point: p[t+h] / p[t] - 1."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    prices = _as_array(close, horizon + 1, "forward_returns")
    return prices[horizon:] / prices[:-horizon] - 1.0


# ========================================
# 📂 Risk-Adjusted Performance
# ========================================
def annualised_volatility(returns: npt.ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of returns scaled to one year (square-root-of-time rule)."""
    r = _as_array(returns, 2, "annualised_volatility")
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: npt.ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised excess return per unit of total volatility."""
    excess = _as_array(returns, 2, "sharpe_ratio") - risk_free_rate / periods_per_year
    sd = float(np.std(excess, ddof=1))
    if sd == 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")
    return float(np.mean(excess) / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: npt.ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Like Sharpe, but penalises only downside volatility (losses), not upside surprises."""
    excess = _as_array(returns, 2, "sortino_ratio") - risk_free_rate / periods_per_year
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    if downside == 0:
        raise ValueError("Sortino ratio is undefined with no downside returns")
    return float(np.mean(excess) / downside * math.sqrt(periods_per_year))


# ========================================
# 📂 Loss Measures
# ========================================
def max_drawdown(close: npt.ArrayLike) -> float:
    """Largest peak-to-trough fall, as a negative fraction (-0.25 means a 25% fall)."""
    prices = _as_array(close, 2, "max_drawdown")
    running_peak = np.maximum.accumulate(prices)
    return float(np.min(prices / running_peak - 1.0))


def historical_var(returns: npt.ArrayLike, level: float = 0.95) -> float:
    """Historical Value-at-Risk: the loss not exceeded on ``level`` of periods (positive number)."""
    if not 0.5 <= level < 1.0:
        raise ValueError("level must be in [0.5, 1)")
    r = _as_array(returns, 2, "historical_var")
    return float(-np.quantile(r, 1.0 - level))
