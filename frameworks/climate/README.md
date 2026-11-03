# ailab-climate

*Author: **Mathew Varghese** · Version 1.0*

How has the climate changed where you live, and how sure can we be? This framework answers
for any location on Earth, using free, citable data and stating its uncertainty in words.

```python
from datetime import date
from ailab_climate import GistempClient, OpenMeteoClient, place_report

kollam = OpenMeteoClient().daily(8.89, 76.61, date(1950, 1, 1), date(2025, 12, 31), place="Kollam")
report = place_report(
    kollam, baseline=(1961, 1990), global_anomalies=GistempClient().global_monthly()
)
print(report.summary())
```

Example output format (figures illustrative, not real results):

```
Annual mean temperature: +0.18 deg C per decade (95% CI +0.12 to +0.24), clear increase. Last 10 years vs baseline: +0.95 deg C.
Annual rainfall: +21.4 mm per decade (95% CI -35.0 to +77.8), no clear trend (the interval includes zero). ...
```

## Components

| Module | What it does |
|---|---|
| `data` | `OpenMeteoClient` (daily Tmax, Tmin, rainfall from ERA5, 1940 onwards), `GistempClient` / `parse_gistemp` (NASA global anomalies), `DailyWeather` |
| `indices` | Six annual indices: mean temperature, hot days, warm nights, rainfall total, heavy rain days (>= 20 mm), wettest day |
| `report` | `place_report`: trend per decade with 95% interval and plain verdict for each index, last decade vs baseline, optional global context, stated caveats |

## Design decisions

- **Uncertainty in words.** Each trend is reported as "clear increase", "clear decrease" or "no
  clear trend", derived from its interval. Tropical rainfall is noisy; saying so is a result.
- **Honest intervals.** Trends use `ailab-timeseries.linear_trend`, which widens intervals for
  year-to-year autocorrelation (tested by simulation in that package).
- **Fixed baseline.** 1961-1990 by default, the WMO reference period for monitoring long-term
  change. Normals are never computed from the full record, which would hide the change.
- **Recognised definitions.** Indices follow the ETCCDI climate change indices in spirit
  (TX90p, TN90p as day counts; R20mm; Rx1day) with calendar-day percentile thresholds.
- **No invented data.** Nulls stay missing; a year below 90% coverage is excluded.
- **Limits stated in every report.** ERA5 is a ~25 km reanalysis grid, not a station, and past
  trends are not projections. Projections need climate model scenarios (CMIP6), a planned extension.

## Data and citation

| Source | Used for | Terms |
|---|---|---|
| [Open-Meteo](https://open-meteo.com/) historical weather API | Daily history from ERA5 / Copernicus | Free for non-commercial use; cite Open-Meteo and ERA5 ("Generated using Copernicus Climate Change Service information") |
| NASA GISS GISTEMP v4 | Global monthly temperature anomalies (1951-1980 baseline) | Public; cite GISTEMP Team, NASA GISS |

## Demo

```bash
pip install -e "frameworks/climate"
python frameworks/climate/examples/place_report.py                     # Kollam, Kerala
python frameworks/climate/examples/place_report.py --global --csv kollam_indices.csv
python frameworks/climate/examples/place_report.py --name London --lat 51.51 --lon -0.13
```

## Roadmap

- Seasonal breakdown (monsoon vs dry season) and monthly rainfall extremes
- Projections from CMIP6 scenarios via the Copernicus Climate Data Store
- Station data (GHCN-Daily) alongside reanalysis for validation

## Development

```bash
cd frameworks/climate
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
