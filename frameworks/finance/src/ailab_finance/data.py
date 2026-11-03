# ==================================================================================================
# 📜 ailab-finance :: data
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/data.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Analysis code should not care where prices come from: a vendor API today, a CSV export or
#    an internal database tomorrow. This module defines one DataSource interface and two
#    implementations, so swapping the source is a one-line change.
#
#  🔧 WHAT IT DOES
#    - DataSource  : interface with load(instrument, start, end) -> PriceSeries.
#    - CsvSource   : reads <directory>/<SYMBOL>.csv (Date, Close or Adj Close columns).
#    - YahooSource : downloads daily history via the optional `yfinance` library.
#
#  🧠 DESIGN NOTES
#    - CSV is the reference source: tests, demos and audits run offline and reproducibly.
#    - "Adj Close" is preferred over "Close" when present. Adjusted prices include splits and
#      dividends; unadjusted ones show false crashes on split dates and distort every return.
#    - YahooSource retries network errors only (OSError and subclasses, which covers requests'
#      connection errors), via mattslib.retry. A bad symbol is not retried; it fails at once.
#    - yfinance is unofficial and Yahoo's terms limit use to personal and research purposes.
#      Production systems should use a licensed data vendor behind the same interface.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: DataSource protocol, CsvSource, YahooSource.
# ==================================================================================================
"""Price data sources behind one interface."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from mattslib import retry

from ailab_finance.types import Instrument, PriceSeries


# ========================================
# 📂 Interface
# ========================================
@runtime_checkable
class DataSource(Protocol):
    """Anything that can return daily prices for an instrument."""

    def load(
        self, instrument: Instrument, *, start: str | None = None, end: str | None = None
    ) -> PriceSeries: ...


# ========================================
# 📂 CSV Source
# ========================================
class CsvSource:
    """Reads ``<directory>/<SYMBOL>.csv`` files with a Date column and a price column."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def load(
        self, instrument: Instrument, *, start: str | None = None, end: str | None = None
    ) -> PriceSeries:
        path = self._directory / f"{instrument.symbol}.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames or []
            price_column = "Adj Close" if "Adj Close" in columns else "Close"
            if "Date" not in columns or price_column not in columns:
                raise ValueError(f"{path.name}: needs 'Date' and 'Close' or 'Adj Close' columns")
            # Vendors write gaps as "", "null" or "NaN"; skip them rather than crash or invent data.
            rows = sorted(
                (row["Date"][:10], float(row[price_column]))
                for row in reader
                if row[price_column] not in ("", "null", "NaN")
            )
        series = PriceSeries(
            instrument,
            np.array([d for d, _ in rows], dtype="datetime64[D]"),
            np.array([p for _, p in rows], dtype=np.float64),
        )
        return series.between(start, end)


# ========================================
# 📂 Yahoo Source (optional)
# ========================================
class YahooSource:
    """Daily adjusted closes from Yahoo Finance via ``yfinance`` (personal/research use)."""

    def __init__(self, *, attempts: int = 3, base_delay: float = 1.0) -> None:
        try:
            import yfinance
        except ImportError as exc:
            raise ImportError(
                "YahooSource needs yfinance: pip install 'ailab-finance[yahoo]'"
            ) from exc
        self._yf: Any = yfinance
        self._fetch = retry(attempts=attempts, exceptions=(OSError,), base_delay=base_delay)(
            self._history
        )

    def _history(self, symbol: str, start: str | None, end: str | None) -> Any:
        return self._yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)

    def load(
        self, instrument: Instrument, *, start: str | None = None, end: str | None = None
    ) -> PriceSeries:
        frame = self._fetch(instrument.symbol, start, end)
        if len(frame) == 0:
            raise LookupError(f"no prices returned for {instrument.symbol!r}")
        dates = np.array([ts.strftime("%Y-%m-%d") for ts in frame.index], dtype="datetime64[D]")
        close = np.asarray(frame["Close"].to_numpy(), dtype=np.float64)
        return PriceSeries(instrument, dates, close)
