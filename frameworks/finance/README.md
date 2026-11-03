# ailab-finance

*Author: **Mathew Varghese** · Version 1.4 · Research and education only. Not financial advice.*

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
| `scorecard` | Credit scorecard: WoE binning with missing-value bins, IV-based selection, points (600 at 50:1, PDO 20), reason codes, validation pack (AUC, Gini, KS, calibration, PSI), group fairness check |
| `annual_report` | Annual report analyser: standard analyst questions answered over a 10-K with RAG; review memo with quoted sources and uncovered topics |
| `fundamentals` | SEC EDGAR client (facts, filing lists, primary documents): point-in-time (by filing date) default point and shares outstanding, with the source filing for every number |
| `outlook` | Historical baseline: return range and probability of +10% / +25% / +50% over a horizon |
| `forecast` | Outlook model: causal market + news-sentiment features, ridge + logistic model, purged walk-forward test with Brier skill vs the baseline |

## Design decisions

- **No look-ahead.** Every analysis uses the series *as known on the decision date*
  (`PriceSeries.until`). A test triples every later price and proves the outlook does not change.
- **Exact money, fast analytics.** Cash amounts use `Decimal` and reject floats; statistics use
  `float64`. See [ADR-0002](../../docs/adr/0002-decimal-money-float-analytics.md).
- **A baseline before a model.** The historical outlook is the benchmark. The model is judged by
  a purged walk-forward test and reported as Brier skill against that baseline, with the verdict
  in words ([ADR-0004](../../docs/adr/0004-purged-walk-forward-against-a-baseline.md)).
- **Honesty is tested.** On a pure random walk the model must not claim skill; with synthetic news
  that carries real information it must find it (+67% Brier skill in the test world).
- **Rare events are not forced.** A +50% month is too rare to model; the smoothed base rate is
  used and flagged instead of a confident-looking guess.
- **Market-based credit risk.** Distance to default needs only share prices and debt, not
  default history, which is scarce and usually paid-for. The probability is model-implied, not a
  calibrated real-world PD; banks map distance to default to PD with their own default data.
- **Fundamentals as they were known.** EDGAR facts are selected by *filing* date, not period end: a
  31 December balance sheet filed on 20 February is invisible on 10 February. Every number carries
  its concept, period, form and accession so it can be traced to the filing.
- **Scorecards a regulator can read.** Points are additive per characteristic and every decision
  comes with reason codes. Protected characteristics are never model inputs; they are used only
  afterwards to check approval rates across groups (adverse impact ratio). Tests document that
  Information Value is biased upward in small samples (pure noise reached 0.09 on 500 rows).
- **Cite or abstain.** Report answers quote the filing's own words by character offset; a topic the
  report does not cover is listed as "not found" rather than guessed
  ([ADR-0005](../../docs/adr/0005-rag-answers-cite-or-abstain.md)).
- **Pandas-free core.** NumPy only, so the framework is light and fully type-checked.
  Shared time-series code will move to a `timeseries` package when a second framework needs it
  ([ADR-0003](../../docs/adr/0003-extract-shared-code-on-second-use.md)).

## Roadmap

| Step | Scope | Status |
|---|---|---|
| 1 | Types, data sources, metrics, distance to default, historical outlook | ✅ V1.0 |
| 2 | `nlp` package: sentiment interface with a FinBERT backend | ✅ (packages/nlp) |
| 3 | Outlook model: market + news features, walk-forward evaluation, skill vs baseline | ✅ V1.1 |
| 3b | Historical news archive (e.g. GDELT) so news features cover the full price history | Planned |
| 4 | Company fundamentals from SEC EDGAR (debt, shares outstanding) | ✅ V1.2 |
| 5 | Combined early warning report and demo | ✅ V1.2 |
| 6 | Credit risk scorecard with validation and fairness check | ✅ V1.3 |
| 7 | Annual report analyser (LLM + RAG over 10-K filings) | ✅ V1.4 |

## Data and licences

| Source / library | Used for | Terms |
|---|---|---|
| CSV files | Tests, demos, audits | Your own data |
| yfinance (optional) | Yahoo Finance prices | Unofficial; Yahoo's terms allow personal and research use only |
| SEC EDGAR | Company filings (XBRL facts) | Public US filings; declare a User-Agent with contact email and stay under 10 requests/second |
| NumPy | Arrays | BSD-3-Clause |
| mattslib | Retries, logging, plugin registry | MIT |
| ailab-ml | Logistic regression, validation metrics | MIT |
| ailab-nlp | Sentiment, news, retrieval, LLM interface, RAG | MIT |

## Examples

```bash
pip install -e "frameworks/finance[yahoo]"

# Scenario 1: Stock Outlook (baseline vs model, with walk-forward evidence)
python frameworks/finance/examples/stock_outlook.py MSFT
python frameworks/finance/examples/stock_outlook.py MSFT --headlines msft.tsv --backend finbert

# Annual report analyser: latest 10-K to a cited review memo (no model needed; or --llm ollama)
export SEC_USER_AGENT="Your Name you@example.com"
python frameworks/finance/examples/annual_report.py MSFT --llm ollama --model llama3.1

# Credit scorecard: build, validate, explain, fairness check (synthetic book, runs offline)
python frameworks/finance/examples/credit_scorecard.py

# Corporate Early Warning: equity outlook + credit risk (debt and shares from SEC EDGAR)
export SEC_USER_AGENT="Your Name you@example.com"
python frameworks/finance/examples/early_warning.py MSFT
python frameworks/finance/examples/early_warning.py MSFT --debt 6e10 --market-cap 3e12
```

## Development

```bash
cd frameworks/finance
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
