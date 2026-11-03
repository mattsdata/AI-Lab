# ==================================================================================================
# 📜 ailab-climate :: tests
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/climate/tests/test_climate.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the data clients build correct requests and parse real formats, the indices follow
#    their definitions, and the report finds warming where it exists and says "no clear trend"
#    where there is none.
#
#  🧠 TESTING APPROACH
#    A synthetic tropical-coast climate (1961-2020): seasonal cycle, known warming of 0.25 deg C
#    per decade, and stationary random rainfall. Fake HTTP for Open-Meteo and GISTEMP.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for ailab-climate.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import json
import urllib.parse
from datetime import date
from typing import Any

import numpy as np
import pytest

from ailab_climate import (
    DailyWeather,
    GistempClient,
    OpenMeteoClient,
    annual_indices,
    global_context,
    parse_gistemp,
    place_report,
)
from ailab_climate import data as data_module


def synthetic_weather(start: int = 1961, end: int = 2020, warming: float = 0.025) -> DailyWeather:
    rng = np.random.default_rng(42)
    dates = np.arange(np.datetime64(f"{start}-01-01"), np.datetime64(f"{end + 1}-01-01"))
    years = (dates - np.datetime64(f"{start}-01-01")).astype(int) / 365.25
    season = np.sin(2 * np.pi * years)
    tmax = 31 + 2 * season + warming * years + rng.normal(0, 1.0, len(dates))
    tmin = 23 + 1 * season + warming * years + rng.normal(0, 0.8, len(dates))
    wet = rng.random(len(dates)) < 0.4
    rain = np.where(wet, rng.gamma(0.8, 12.0, len(dates)), 0.0)  # stationary rainfall
    return DailyWeather("Testville", 8.9, 76.6, dates, tmax, tmin, rain, "synthetic")


# ========================================
# 📂 Data Tests
# ========================================
def test_open_meteo_request_and_parsing() -> None:
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return json.dumps(
            {
                "latitude": 8.875,
                "longitude": 76.625,
                "daily": {
                    "time": ["2024-01-01", "2024-01-02"],
                    "temperature_2m_max": [31.5, None],
                    "temperature_2m_min": [23.5, 24.0],
                    "precipitation_sum": [0.0, 12.4],
                },
            }
        ).encode()

    weather = OpenMeteoClient(fetch=fetch).daily(
        8.89, 76.61, date(2024, 1, 1), date(2024, 1, 2), place="Kollam"
    )
    assert weather.place == "Kollam" and weather.latitude == 8.875  # grid cell reported
    assert weather.tmean[0] == 27.5 and np.isnan(weather.tmax[1])  # nulls stay missing
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(urls[0]).query))
    assert params["daily"] == "temperature_2m_max,temperature_2m_min,precipitation_sum"
    assert (params["start_date"], params["end_date"], params["timezone"]) == (
        "2024-01-01",
        "2024-01-02",
        "auto",
    )


def test_open_meteo_validation_and_errors() -> None:
    client = OpenMeteoClient(fetch=lambda url: b'{"error": true, "reason": "Latitude invalid"}')
    with pytest.raises(ValueError, match="Latitude invalid"):
        client.daily(8.9, 76.6, date(2024, 1, 1), date(2024, 1, 2))
    with pytest.raises(ValueError):
        client.daily(95.0, 0.0, date(2024, 1, 1), date(2024, 1, 2))
    with pytest.raises(ValueError):
        client.daily(8.9, 76.6, date(1930, 1, 1), date(1931, 1, 1))  # before ERA5


def test_daily_weather_validation() -> None:
    d = np.array(["2024-01-02", "2024-01-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="increasing"):
        DailyWeather("x", 0, 0, d, np.zeros(2), np.zeros(2), np.zeros(2), "s")
    with pytest.raises(ValueError, match="same length"):
        DailyWeather("x", 0, 0, d[:1], np.zeros(2), np.zeros(2), np.zeros(2), "s")


GISTEMP_SAMPLE = """Land-Ocean: Global Means
Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N,DJF,MAM,JJA,SON
2014,.74,.51,.76,.77,.85,.66,.57,.82,.89,.79,.67,.79,.74,.73,.64,.79,.68,.78
2015,.82,.87,.90,.74,.75,.79,.71,.79,.81,1.06,1.03,1.10,.86,.84,.83,.80,.76,.97
2016,1.17,1.37,1.36,1.08,.93,.76,.83,1.00,.88,.88,.91,.87,1.01,1.03,1.21,1.12,.86,.89
2017,1.02,1.13,1.17,.93,.91,.71,.82,.87,.76,.89,.87,.93,.92,.92,1.01,1.00,.80,.84
2018,.81,.85,.89,.89,.82,.77,.83,.76,.80,1.01,.82,.92,.85,.85,.86,.86,.79,.88
2019,.93,.95,1.18,1.01,.86,.91,.94,.95,.93,1.01,.99,1.10,.98,.96,.93,1.02,.93,.98
2020,1.18,1.24,1.17,1.13,1.02,.92,.90,.87,.99,.88,1.10,.81,1.02,1.04,1.17,1.11,.90,.99
2021,.81,.64,.88,.76,.79,.84,.92,.82,.92,.99,.93,.86,.85,.85,.75,.81,.86,.95
2022,.91,.89,1.05,.83,.84,.92,.94,.95,.89,.96,.73,.80,.89,.90,.89,.91,.94,.86
2023,.87,.98,1.20,1.00,.94,1.08,1.19,1.19,1.48,1.34,1.42,1.37,1.17,1.12,.88,1.05,1.15,1.41
2024,1.24,1.44,1.39,1.31,1.17,1.23,1.21,1.29,1.24,***,***,***,****,****,1.38,1.29,1.24,****
"""


def test_gistemp_parsing_and_global_context() -> None:
    record = parse_gistemp(GISTEMP_SAMPLE)
    assert record.baseline == (1951, 1980) and len(record.dates) == 11 * 12
    assert record.anomalies[0] == 0.74 and np.isnan(record.anomalies[-1])  # *** is missing
    text = global_context(record)
    assert "2014-2023" in text and "1951-1980" in text  # incomplete 2024 excluded
    assert "+0.93 deg C" in text
    with pytest.raises(ValueError, match="Year"):
        parse_gistemp("not a table")
    short = parse_gistemp("\n".join(GISTEMP_SAMPLE.splitlines()[:5]))
    with pytest.raises(ValueError, match="ten complete years"):
        global_context(short)


def test_gistemp_client_and_http(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GistempClient(fetch=lambda url: GISTEMP_SAMPLE.encode())
    assert len(client.global_monthly().dates) == 132

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> Response:
        seen["agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert data_module._http_get("https://example.org") == b"ok"
    assert seen["agent"] == "ailab-climate/1.0"


# ========================================
# 📂 Index Tests
# ========================================
def test_indices_follow_their_definitions() -> None:
    weather = synthetic_weather()
    idx = annual_indices(weather)
    assert idx.years[0] == 1961 and idx.years[-1] == 2020
    in_base = idx.years <= 1990
    hot_rate = np.mean(idx.values["hot_days"][in_base]) / 365.25
    assert hot_rate == pytest.approx(0.10, abs=0.01)  # 90th percentile -> ~10% of baseline days
    year = np.flatnonzero(idx.years == 2000)[0]
    days = (weather.dates >= np.datetime64("2000-01-01")) & (
        weather.dates < np.datetime64("2001-01-01")
    )
    assert idx.values["rainfall_total"][year] == pytest.approx(weather.precipitation[days].sum())
    assert idx.values["wettest_day"][year] == pytest.approx(weather.precipitation[days].max())
    assert idx.values["heavy_rain_days"][year] == np.sum(weather.precipitation[days] >= 20)
    assert idx.unit("rainfall_total") == "mm" and "Hot days" in idx.label("hot_days")


def test_years_with_missing_data_are_excluded() -> None:
    weather = synthetic_weather()
    gappy = weather.tmax.copy()
    gappy[
        (weather.dates >= np.datetime64("2005-01-01"))
        & (weather.dates < np.datetime64("2005-03-01"))
    ] = np.nan
    w = DailyWeather("x", 0, 0, weather.dates, gappy, weather.tmin, weather.precipitation, "s")
    idx = annual_indices(w)
    year = np.flatnonzero(idx.years == 2005)[0]
    assert np.isnan(idx.values["mean_temperature"][year])  # 59 of 365 days missing
    assert np.isnan(idx.values["hot_days"][year])
    assert not np.isnan(idx.values["rainfall_total"][year])  # rain was fully observed


# ========================================
# 📂 Report Tests
# ========================================
def test_report_finds_warming_and_no_rain_trend() -> None:
    report = place_report(synthetic_weather())
    temperature = report.trend("mean_temperature")
    assert temperature.trend.low < 0.25 < temperature.trend.high  # true value inside the CI
    assert temperature.verdict == "clear increase"
    assert report.trend("hot_days").verdict == "clear increase"
    assert report.trend("warm_nights").verdict == "clear increase"
    assert report.trend("rainfall_total").verdict.startswith("no clear trend")
    assert temperature.recent_mean - temperature.baseline_mean == pytest.approx(1.1, abs=0.2)
    text = report.summary()
    assert "Testville" in text and "1961-2020" in text and "not projections" in text


def test_report_with_global_context_and_validation() -> None:
    report = place_report(synthetic_weather(), global_anomalies=parse_gistemp(GISTEMP_SAMPLE))
    assert report.global_context is not None and "Global context" in report.summary()
    with pytest.raises(ValueError, match="baseline"):
        place_report(synthetic_weather(1995, 2020), baseline=(1961, 1990))


def test_cooling_short_and_empty_records() -> None:
    cooling = place_report(synthetic_weather(warming=-0.03))
    assert cooling.trend("mean_temperature").verdict == "clear decrease"
    with pytest.raises(ValueError, match="at least 10 years"):
        place_report(synthetic_weather(1961, 1966))
    w = synthetic_weather(1961, 1970)
    sparse = w.tmax.copy()
    sparse[::7] = np.nan  # every year ~86% covered, below the 90% rule
    with pytest.raises(ValueError, match="no year has enough data"):
        place_report(DailyWeather("x", 0, 0, w.dates, sparse, w.tmin, w.precipitation, "s"))


def test_gistemp_ignores_non_data_rows() -> None:
    lines = GISTEMP_SAMPLE.splitlines()
    noisy = "\n".join([*lines[:3], "", "Year,Jan,Feb", *lines[3:]])  # blank and repeated header
    assert len(parse_gistemp(noisy).dates) == 132
