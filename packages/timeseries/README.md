# ailab-timeseries

*Author: **Mathew Varghese** · Version 1.0*

Time-series building blocks shared by every AI Lab framework: finance uses the rolling
statistics for market features; climate uses aggregation, trends and seasonal tools.
NumPy only.

```python
from ailab_timeseries import aggregate, linear_trend, seasonal_threshold_flags

years, annual_mean = aggregate(dates, temperature, freq="year", how="mean", min_coverage=0.9)
trend = linear_trend(years.astype("datetime64[Y]").astype(int) + 1970, annual_mean, per=10)
print(f"{trend.slope:+.2f} per decade (95% CI {trend.low:+.2f} to {trend.high:+.2f})")
hot = seasonal_threshold_flags(dates, tmax, baseline=(1961, 1990), quantile=0.9)
```

## Components

| Module | What it does |
|---|---|
| `core` | `rolling_mean`, `rolling_std` (causal, NaN-aware); `aggregate` (month/year with a coverage rule); `linear_trend` with autocorrelation-adjusted confidence interval |
| `seasonal` | `day_of_year` (leap day folded), `daily_climatology`, `seasonal_anomalies`, `seasonal_threshold_flags` against an explicit baseline |

## Design decisions

- **Causal windows.** Value *i* uses data up to *i* only, so the same code is safe in back-tests.
- **Coverage rule.** A period with too few observations is NaN rather than a biased total.
- **Honest uncertainty.** Trend intervals use the effective sample size for lag-1 autocorrelation
  (Santer et al., 2000). A Monte Carlo test shows plain OLS intervals cover the true slope far less
  than 95% of the time on autocorrelated data, and that the correction restores most of it.
- **Explicit baselines.** Normals come from a fixed reference period (e.g. 1961-1990), never the
  full record, which would absorb the change being measured.
- **Extracted on second use** from finance when climate needed the same tools (ADR-0003).

## Development

```bash
cd packages/timeseries
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
