# ==================================================================================================
# 📜 ailab-climate :: data
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/climate/src/ailab_climate/data.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Free, citable climate data for anywhere on Earth: daily weather history for a location
#    (Open-Meteo, from the ERA5 reanalysis, 1940 onwards) and the global temperature record
#    (NASA GISTEMP). Enough to describe how a place's climate has changed since the mid-1900s.
#
#  🔧 WHAT IT DOES
#    - DailyWeather      : daily maximum / minimum temperature and rainfall for one location.
#    - OpenMeteoClient   : fetches DailyWeather from the Open-Meteo historical weather API.
#    - MonthlyAnomalies  : global monthly temperature anomalies with their baseline.
#    - GistempClient, parse_gistemp : NASA GISTEMP v4 global land-ocean anomalies (CSV).
#
#  🧠 DESIGN NOTES
#    - Daily mean temperature is (Tmax + Tmin) / 2, the long-standing climatological convention,
#      computed from the two variables the API documents explicitly.
#    - Missing values (null in JSON, *** in GISTEMP) become NaN and are never filled in; the
#      indices apply coverage rules instead of inventing data.
#    - ERA5 is a reanalysis: a physics model constrained by observations on a ~25 km grid. It is
#      consistent over decades but is not a weather station; coastal cells mix land and sea.
#    - HTTP, retries and sleep are injectable, so tests run offline. Cite Open-Meteo, ERA5
#      (Copernicus Climate Change Service) and NASA GISS when publishing results.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: DailyWeather, Open-Meteo client, GISTEMP parser and client.
# ==================================================================================================
"""Climate data sources: Open-Meteo daily history and NASA GISTEMP global anomalies."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
from mattslib import retry

FloatArray = npt.NDArray[np.float64]
DateArray = npt.NDArray[np.datetime64]

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
GISTEMP_GLOBAL_CSV = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
ERA5_START = date(1940, 1, 1)


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ailab-climate/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body: bytes = response.read()
        return body


# ========================================
# 📂 Daily Weather
# ========================================
@dataclass(frozen=True)
class DailyWeather:
    place: str
    latitude: float
    longitude: float
    dates: DateArray
    tmax: FloatArray  # deg C
    tmin: FloatArray  # deg C
    precipitation: FloatArray  # mm per day
    source: str

    def __post_init__(self) -> None:
        n = len(self.dates)
        if not len(self.tmax) == len(self.tmin) == len(self.precipitation) == n:
            raise ValueError("dates, tmax, tmin and precipitation must have the same length")
        if n and np.any(np.diff(self.dates.astype("datetime64[D]")) <= np.timedelta64(0, "D")):
            raise ValueError("dates must be strictly increasing")

    @property
    def tmean(self) -> FloatArray:
        """Daily mean temperature as (Tmax + Tmin) / 2."""
        out: FloatArray = (self.tmax + self.tmin) / 2
        return out


def _floats(values: list[Any]) -> FloatArray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


class OpenMeteoClient:
    """Open-Meteo historical weather API (ERA5). Free for non-commercial use; see their terms."""

    def __init__(
        self,
        *,
        fetch: Callable[[str], bytes] = _http_get,
        attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._fetch = retry(attempts=attempts, exceptions=(OSError,), sleep=sleep)(fetch)

    def daily(
        self, latitude: float, longitude: float, start: date, end: date, *, place: str = ""
    ) -> DailyWeather:
        """Daily Tmax, Tmin and rainfall for a point, from ``start`` to ``end`` inclusive."""
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("latitude must be in [-90, 90] and longitude in [-180, 180]")
        if start < ERA5_START or end < start:
            raise ValueError(f"need {ERA5_START} <= start <= end")
        query = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            }
        )
        payload: dict[str, Any] = json.loads(self._fetch(f"{OPEN_METEO_ARCHIVE}?{query}"))
        if payload.get("error"):
            raise ValueError(f"Open-Meteo: {payload.get('reason', 'request rejected')}")
        daily = payload["daily"]
        return DailyWeather(
            place=place or f"{latitude:.2f}, {longitude:.2f}",
            latitude=float(payload.get("latitude", latitude)),  # the grid cell actually used
            longitude=float(payload.get("longitude", longitude)),
            dates=np.array(daily["time"], dtype="datetime64[D]"),
            tmax=_floats(daily["temperature_2m_max"]),
            tmin=_floats(daily["temperature_2m_min"]),
            precipitation=_floats(daily["precipitation_sum"]),
            source="Open-Meteo historical weather API (ERA5 reanalysis, ~25 km grid)",
        )


# ========================================
# 📂 Global Temperature (NASA GISTEMP)
# ========================================
@dataclass(frozen=True)
class MonthlyAnomalies:
    dates: DateArray  # first day of each month
    anomalies: FloatArray  # deg C relative to `baseline`
    baseline: tuple[int, int]
    source: str


def parse_gistemp(text: str) -> MonthlyAnomalies:
    """Parse GISTEMP v4 'GLB.Ts+dSST.csv' (monthly columns Jan..Dec, '***' for missing)."""
    lines = text.splitlines()
    try:
        header_at = next(i for i, line in enumerate(lines) if line.startswith("Year,"))
    except StopIteration:
        raise ValueError("not a GISTEMP table: no 'Year,' header row") from None
    dates, values = [], []
    for line in lines[header_at + 1 :]:
        cells = [c.strip() for c in line.split(",")]
        if not cells or not cells[0].isdigit():
            continue
        for month, cell in enumerate(cells[1:13], start=1):
            dates.append(f"{cells[0]}-{month:02d}-01")
            values.append(np.nan if cell.startswith("*") or not cell else float(cell))
    return MonthlyAnomalies(
        dates=np.array(dates, dtype="datetime64[D]"),
        anomalies=np.array(values, dtype=np.float64),
        baseline=(1951, 1980),
        source="NASA GISS Surface Temperature Analysis (GISTEMP v4), global land-ocean",
    )


class GistempClient:
    def __init__(self, *, fetch: Callable[[str], bytes] = _http_get, attempts: int = 3) -> None:
        self._fetch = retry(attempts=attempts, exceptions=(OSError,))(fetch)

    def global_monthly(self) -> MonthlyAnomalies:
        return parse_gistemp(self._fetch(GISTEMP_GLOBAL_CSV).decode("utf-8", errors="replace"))
