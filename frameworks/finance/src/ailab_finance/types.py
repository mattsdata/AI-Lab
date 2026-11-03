# ==================================================================================================
# 📜 ailab-finance :: types
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/types.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    The core vocabulary of the finance framework: what is traded, its price history, and money.
#    Getting these types right prevents the two most expensive classes of financial-software
#    bugs: rounding errors in cash amounts and accidental use of future data in analysis.
#
#  🔧 WHAT IT DOES
#    - Instrument  : a tradable security (symbol, currency, exchange).
#    - PriceSeries : validated daily closing prices; `until(date)` gives an as-of view.
#    - Money       : an exact Decimal amount with a currency.
#
#  🧠 DESIGN NOTES
#    - Money uses Decimal and refuses floats: Decimal(0.1) is 0.1000000000000000055..., so a
#      float would carry its binary error into every ledger total. Analytics (returns,
#      volatility) use float64 for speed, where tiny errors are irrelevant. See ADR-0002.
#    - PriceSeries validates on construction (sorted dates, positive prices) and stores
#      read-only copies, so downstream code can trust it and nobody can edit history in place.
#    - `until(as_of)` is the guard against look-ahead bias: every model asks for the series as
#      it was known on the decision date, never the full history.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: Instrument, PriceSeries, Money.
# ==================================================================================================
"""Core finance types: instruments, price series and exact money.

Example:
    >>> from decimal import Decimal
    >>> fee = Money(Decimal("1.10")) + Money(Decimal("2.20"))
    >>> str(fee)
    '3.30 USD'
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import numpy.typing as npt

# ========================================
# 📂 Types
# ========================================
DateArray = npt.NDArray[np.datetime64]
FloatArray = npt.NDArray[np.float64]


# ========================================
# 📂 Instrument
# ========================================
@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable security, e.g. Instrument("AAPL") or Instrument("HSBA.L", "GBP", "LSE")."""

    symbol: str
    currency: str = "USD"
    exchange: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")


# ========================================
# 📂 Price Series
# ========================================
@dataclass(frozen=True)
class PriceSeries:
    """Daily closing prices for one instrument, sorted, positive and read-only."""

    instrument: Instrument
    dates: DateArray
    close: FloatArray

    def __post_init__(self) -> None:
        # Copy, then lock: the caller's arrays stay writable, ours cannot be changed.
        dates = np.array(self.dates, dtype="datetime64[D]", copy=True)
        close = np.array(self.close, dtype=np.float64, copy=True)
        if dates.ndim != 1 or dates.shape != close.shape:
            raise ValueError("dates and close must be 1-D arrays of equal length")
        if len(close) < 2:
            raise ValueError("a price series needs at least two observations")
        if np.any(np.diff(dates) <= np.timedelta64(0, "D")):
            raise ValueError("dates must be strictly increasing (sorted, no duplicates)")
        if not np.all(np.isfinite(close)) or np.any(close <= 0):
            raise ValueError("prices must be positive and finite")
        dates.setflags(write=False)
        close.setflags(write=False)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "close", close)

    def __len__(self) -> int:
        return len(self.close)

    @property
    def start(self) -> np.datetime64:
        first: np.datetime64 = self.dates[0]
        return first

    @property
    def end(self) -> np.datetime64:
        last: np.datetime64 = self.dates[-1]
        return last

    def until(self, as_of: str | np.datetime64) -> PriceSeries:
        """The series as known on ``as_of`` (inclusive): the look-ahead guard."""
        mask = self.dates <= np.datetime64(as_of, "D")
        return PriceSeries(self.instrument, self.dates[mask], self.close[mask])

    def between(self, start: str | None = None, end: str | None = None) -> PriceSeries:
        """Sub-series from ``start`` to ``end`` (both inclusive, either optional)."""
        mask = np.ones(len(self), dtype=bool)
        if start is not None:
            mask &= self.dates >= np.datetime64(start, "D")
        if end is not None:
            mask &= self.dates <= np.datetime64(end, "D")
        return PriceSeries(self.instrument, self.dates[mask], self.close[mask])


# ========================================
# 📂 Money
# ========================================
@dataclass(frozen=True, slots=True)
class Money:
    """An exact amount of money. Build it from Decimal, int or str, never float."""

    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        if isinstance(self.amount, float):
            raise TypeError("use Decimal('0.10') or a string, not a float, for money")
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(self.amount))

    def _same_currency(self, other: Money) -> None:
        if other.currency != self.currency:
            raise ValueError(f"cannot combine {self.currency} with {other.currency}")

    def __add__(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        if isinstance(factor, float):
            raise TypeError("multiply money by Decimal or int, not float")
        return Money(self.amount * factor, self.currency)

    def rounded(self, places: int = 2) -> Money:
        """Round half-up, the convention used on statements and invoices."""
        return Money(self.amount.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP), self.currency)

    def __str__(self) -> str:
        return f"{self.rounded().amount:,.2f} {self.currency}"
