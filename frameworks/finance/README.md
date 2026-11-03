# ailab-finance

*Author: **Mathew Varghese** · Version 1.0 · Research and education only. Not financial advice.*

Market analytics and the foundations of a **Corporate Early Warning System**: for a listed
company, one view of its **equity outlook** (how its share price could move) and its
**credit risk** (how close it is to default), the way a bank monitors corporate borrowers.

```python
from ailab_finance import (
    CsvSource,
    Instrument,
    annualised_volatility,
    historical_outlook,
    naive_distance_to_default,
    simple_returns,
    warning_flag,
)

prices = CsvSource("data/").load(Instrument("XYZ"))

# Equity outlook: how often did the stock gain 10% / 25% / 50% in a month, as of today?
print(historical_outlook(prices, horizon=21).summary())

# Credit risk: naive Merton distance to default
vol = annualised_volatility(simple_returns(prices.close[-252:]))
risk = naive_distance_to_default(
    equity_value=2.5e9, debt=1.8e9, equity_volatility=vol, expected_return=0.04
)
print(round(risk.distance_to_default, 2), f"{risk.default_probability:.2%}", warning_flag(risk))
```

## Components

| Module | What it does |
|---|---|
| `types` | `Instrument`, `PriceSeries` (validated, read-only, `until()` as-of view), `Money` (exact `Decimal`) |
| `data` | `DataSource` interface, `CsvSource`, `YahooSource` (optional, with retries via `mattslib`) |
| `metrics` | Returns, annualised volatility, Sharpe, Sortino, max drawdown, historical VaR |
| `credit` | Naive Merton distance to default (Bharath & Shumway, 2008), GREEN / AMBER / RED flags |
| `outlook` | Historical baseline: return range and probability of +10% / +25% / +50% over a horizon |

## Design decisions

- **No look-ahead.** Every analysis uses the series *as known on the decision date*
  (`PriceSeries.until`). A test triples every later price and proves the outlook does not change.
- **Exact money, fast analytics.** Cash amounts use `Decimal` and reject floats; statistics use
  `float64`. See [ADR-0002](../../docs/adr/0002-decimal-money-float-analytics.md).
- **A baseline before a model.** The historical outlook is the benchmark that the planned
  sentiment and machine-learning model must beat, measured by calibration, before it is trusted.
- **Market-based credit risk.** Distance to default needs only share prices and debt, not
  default history, which is scarce and usually paid-for. The probability is model-implied, not a
  calibrated real-world PD; banks map distance to default to PD with their own default data.
- **Pandas-free core.** NumPy only, so the framework is light and fully type-checked.
  Shared time-series code will move to a `timeseries` package when a second framework needs it
  ([ADR-0003](../../docs/adr/0003-extract-shared-code-on-second-use.md)).

## Roadmap

| Step | Scope | Status |
|---|---|---|
| 1 | Types, data sources, metrics, distance to default, historical outlook | ✅ V1.0 |
| 2 | `nlp` package: sentiment interface with a FinBERT backend | Planned |
| 3 | Outlook model: news sentiment + market features, quantile model, walk-forward evaluation, calibration vs baseline | Planned |
| 4 | Company fundamentals from SEC EDGAR (debt, shares outstanding) | Planned |
| 5 | Combined early warning report and demo | Planned |

## Data and licences

| Source / library | Used for | Terms |
|---|---|---|
| CSV files | Tests, demos, audits | Your own data |
| yfinance (optional) | Yahoo Finance prices | Unofficial; Yahoo's terms allow personal and research use only |
| NumPy | Arrays | BSD-3-Clause |
| mattslib | Retries, logging | MIT |

## Example

```bash
pip install -e "frameworks/finance[yahoo]"
python frameworks/finance/examples/early_warning.py MSFT --debt 6e10 --market-cap 3e12
```

## Development

```bash
cd frameworks/finance
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
