# ==================================================================================================
# 📜 ailab-finance :: tests :: types and data
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/tests/test_types_and_data.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the validation rules of the core types and both data sources.
#
#  🧠 TESTING APPROACH
#    CSV files are written to pytest's tmp_path; Yahoo is replaced by a fake `yfinance` module
#    with the same shape, so tests run offline and never depend on a vendor being up.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for Instrument, PriceSeries, Money, CsvSource and YahooSource.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import sys
import types
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ailab_finance import CsvSource, DataSource, Instrument, Money, PriceSeries, YahooSource

AAPL = Instrument("AAPL")


def series(prices: list[float], start: str = "2026-01-01") -> PriceSeries:
    dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(len(prices), "D"))
    return PriceSeries(AAPL, dates, np.array(prices))


# ========================================
# 📂 Type Tests
# ========================================
def test_instrument_requires_symbol() -> None:
    with pytest.raises(ValueError):
        Instrument("  ")


@pytest.mark.parametrize(
    ("dates", "prices"),
    [
        (["2026-01-02", "2026-01-01"], [1.0, 2.0]),  # not sorted
        (["2026-01-01", "2026-01-01"], [1.0, 2.0]),  # duplicate date
        (["2026-01-01", "2026-01-02"], [1.0, -2.0]),  # negative price
        (["2026-01-01", "2026-01-02"], [1.0, float("nan")]),  # missing price
        (["2026-01-01"], [1.0]),  # too short
        (["2026-01-01", "2026-01-02"], [1.0]),  # length mismatch
    ],
)
def test_price_series_rejects_bad_data(dates: list[str], prices: list[float]) -> None:
    with pytest.raises(ValueError):
        PriceSeries(AAPL, np.array(dates, dtype="datetime64[D]"), np.array(prices))


def test_price_series_is_read_only_and_slices() -> None:
    s = series([10, 11, 12, 13, 14])
    with pytest.raises(ValueError):
        s.close[0] = 99.0  # history cannot be edited in place
    assert len(s.until("2026-01-03")) == 3
    assert s.until("2026-01-03").end == np.datetime64("2026-01-03")
    assert len(s.between("2026-01-02", "2026-01-04")) == 3
    assert s.start == np.datetime64("2026-01-01")


def test_money_is_exact_and_currency_safe() -> None:
    total = Money(Decimal("0.10")) + Money(Decimal("0.20"))
    assert total.amount == Decimal("0.30")  # 0.1 + 0.2 == 0.30000000000000004 with floats
    assert (Money("10.00") - Money("2.50")).amount == Decimal("7.50")
    assert str(Money(Decimal("1234.565")) * 2) == "2,469.13 USD"
    assert Money("2.345").rounded().amount == Decimal("2.35")  # half-up, as on statements
    with pytest.raises(TypeError):
        Money(0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Money("1") * 1.5  # type: ignore[operator]
    with pytest.raises(ValueError):
        Money("1", "USD") + Money("1", "GBP")


# ========================================
# 📂 CSV Source Tests
# ========================================
def test_csv_prefers_adjusted_close_and_skips_gaps(tmp_path: Path) -> None:
    (tmp_path / "AAPL.csv").write_text(
        "Date,Close,Adj Close\n"
        "2026-01-03,30,15\n"  # unsorted on purpose
        "2026-01-01,10,5\n"
        "2026-01-02,20,null\n"
        "2026-01-04,40,20\n"
    )
    source = CsvSource(tmp_path)
    assert isinstance(source, DataSource)
    s = source.load(AAPL)
    assert list(s.close) == [5.0, 15.0, 20.0]
    assert list(source.load(AAPL, start="2026-01-03").close) == [15.0, 20.0]


def test_csv_requires_columns(tmp_path: Path) -> None:
    (tmp_path / "AAPL.csv").write_text("Day,Price\n2026-01-01,1\n")
    with pytest.raises(ValueError, match="needs 'Date'"):
        CsvSource(tmp_path).load(AAPL)


# ========================================
# 📂 Yahoo Source Tests (fake yfinance)
# ========================================
class _FakeColumn:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_numpy(self) -> Any:
        return np.array(self._values)


class _FakeFrame:
    def __init__(self, rows: dict[str, float]) -> None:
        self.index = [datetime.fromisoformat(d) for d in rows]
        self._close = list(rows.values())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key: str) -> _FakeColumn:
        assert key == "Close"
        return _FakeColumn(self._close)


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch, frames: list[Any]) -> list[str]:
    calls: list[str] = []

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: Any) -> Any:
            calls.append(self.symbol)
            item = frames.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    module = types.ModuleType("yfinance")
    module.Ticker = Ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", module)
    return calls


def test_yahoo_retries_network_errors_then_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _FakeFrame({"2026-01-01": 100.0, "2026-01-02": 101.5})
    calls = _install_fake_yfinance(monkeypatch, [ConnectionError("reset"), frame])
    s = YahooSource(attempts=2, base_delay=0).load(AAPL)
    assert list(s.close) == [100.0, 101.5] and calls == ["AAPL", "AAPL"]


def test_yahoo_empty_result_and_missing_library(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(monkeypatch, [_FakeFrame({})])
    with pytest.raises(LookupError):
        YahooSource().load(Instrument("NOPE"))
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"ailab-finance\[yahoo\]"):
        YahooSource()
