# ==================================================================================================
# 📜 ailab-finance :: outlook
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/outlook.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    "What return could this stock make over the next month, and how likely is +10%, +25% or
#    +50%?" This module answers from the stock's own history. It is the honest baseline that
#    any smarter model (news sentiment, machine learning) must beat to be worth using.
#
#  🔧 WHAT IT DOES
#    - historical_outlook : distribution of past `horizon`-day returns as of a decision date:
#                           median, an interval (e.g. 10th-90th percentile) and the
#                           probability of reaching each return threshold.
#    - ReturnOutlook      : the result, with a plain-text summary().
#
#  🧠 DESIGN NOTES
#    - Uses only data up to `as_of` (PriceSeries.until). A test proves that changing any later
#      price leaves the outlook unchanged, i.e. there is no look-ahead.
#    - Overlapping windows: each day starts a new 21-day window, so neighbouring samples share
#      most of their days. The effective sample size is far smaller than `samples`; the
#      summary says so instead of implying false precision.
#    - The past is not a forecast. Regimes change, and survivors dominate history. This is a
#      reference point for comparison, not a prediction, and never investment advice.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: historical baseline outlook with threshold probabilities.
# ==================================================================================================
"""Historical baseline for forward-return outlooks (no look-ahead)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ailab_finance.metrics import forward_returns
from ailab_finance.types import PriceSeries

DISCLAIMER = "Research and education only. Not financial advice."


# ========================================
# 📂 Result Type
# ========================================
@dataclass(frozen=True)
class ReturnOutlook:
    symbol: str
    as_of: np.datetime64
    horizon_days: int
    samples: int
    median_return: float
    interval: tuple[float, float]
    interval_level: float
    threshold_probabilities: dict[float, float]
    method: str = "historical baseline"

    def summary(self) -> str:
        low, high = self.interval
        lines = [
            f"{self.symbol} · as of {self.as_of} · {self.horizon_days}-day horizon · {self.method}",
            f"Median return: {self.median_return:+.1%}   "
            f"{self.interval_level:.0%} range: {low:+.1%} to {high:+.1%}",
        ]
        lines += [
            f"P(return >= {t:+.0%}): {p:.1%}"
            for t, p in sorted(self.threshold_probabilities.items())
        ]
        lines.append(
            f"Based on {self.samples} overlapping windows (effective sample size is smaller)."
        )
        lines.append(DISCLAIMER)
        return "\n".join(lines)


# ========================================
# 📂 Historical Baseline
# ========================================
def historical_outlook(
    series: PriceSeries,
    *,
    as_of: str | None = None,
    horizon: int = 21,
    thresholds: Sequence[float] = (0.10, 0.25, 0.50),
    interval_level: float = 0.80,
    min_samples: int = 60,
) -> ReturnOutlook:
    """Outlook from the empirical distribution of past ``horizon``-day returns.

    21 trading days is roughly one calendar month.
    """
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be in (0, 1)")
    history = series if as_of is None else series.until(as_of)
    if len(history) <= horizon:
        raise ValueError(f"need more than {horizon} prices up to the decision date")
    returns = forward_returns(history.close, horizon)
    if len(returns) < min_samples:
        raise ValueError(f"only {len(returns)} windows; at least {min_samples} required")

    tail = (1 - interval_level) / 2
    low, high = np.quantile(returns, [tail, 1 - tail])
    return ReturnOutlook(
        symbol=series.instrument.symbol,
        as_of=history.end,
        horizon_days=horizon,
        samples=len(returns),
        median_return=float(np.median(returns)),
        interval=(float(low), float(high)),
        interval_level=interval_level,
        threshold_probabilities={t: float(np.mean(returns >= t)) for t in thresholds},
    )
