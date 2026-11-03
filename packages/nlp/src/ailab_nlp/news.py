# ==================================================================================================
# 📜 ailab-nlp :: news
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/news.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A sentiment model is only as good as its history. To test whether news predicts anything,
#    every past decision date needs the news that was known then. This module connects to the
#    GDELT Project, a free global news monitor, and turns it into daily, look-ahead-safe inputs.
#
#  🔧 WHAT IT DOES
#    - GdeltClient.daily_tone : GDELT's daily average tone and article count for a query, over
#                               long periods (timeline modes cover 2017 onwards).
#    - GdeltClient.articles   : recent article titles and links (GDELT keeps article lists for
#                               about the last three months), to score with our own analyzers.
#    - rolling_tone           : the news mood as known on a decision date, decay-weighted over
#                               the preceding days. Excludes the decision day itself by default.
#    - NewsArticle, DailyTone, ToneSummary : plain result types.
#
#  🧠 DESIGN NOTES
#    - Two GDELT modes, two jobs. Timelines give years of history for back-testing; article
#      lists give recent text for richer models such as FinBERT. Neither is presented as the other.
#    - Tone is GDELT's own lexicon-based measure, typically within -10..+10. It is mapped to
#      -1..+1 by dividing by `tone_scale` (10) and clipping; the scale is a parameter, not hidden.
#    - Same-day news is excluded by default: a day's bucket can include stories published after
#      the market closed, which would leak tomorrow's information into today's decision.
#    - Polite client: requests are spaced by `min_interval` seconds and network errors are retried
#      with backoff (mattslib.retry). A non-JSON reply (GDELT returns plain-text errors for bad
#      queries) raises a ValueError that includes the server's message.
#    - HTTP, clock and sleep are injectable, so tests run offline and instantly.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: GDELT client (tone timeline, article list), rolling_tone.
# ==================================================================================================
"""GDELT news connector and look-ahead-safe rolling news tone.

Example:
    >>> from datetime import date
    >>> days = [DailyTone(date(2026, 1, d), score=0.2 * d, count=10) for d in (1, 2, 3)]
    >>> summary = rolling_tone(days, as_of=date(2026, 1, 3), window_days=7, half_life_days=7)
    >>> summary.count  # Jan 3 itself is excluded: same-day news may postdate the decision
    20
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from mattslib import retry

from ailab_nlp.aggregate import time_decay_weights

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250  # GDELT's limit for article lists


# ========================================
# 📂 Result Types
# ========================================
@dataclass(frozen=True, slots=True)
class NewsArticle:
    published: date
    title: str
    url: str
    source: str
    language: str


@dataclass(frozen=True, slots=True)
class DailyTone:
    day: date
    score: float  # GDELT average tone mapped to -1..+1
    count: int  # number of matching articles that day


@dataclass(frozen=True, slots=True)
class ToneSummary:
    """News mood as known on a decision date. Has `score` and `count`, as finance expects."""

    score: float
    count: int
    days: int  # days with coverage inside the window


def _day(stamp: str) -> date:
    """Parse GDELT stamps such as '20240115T000000Z' (or ISO dates) to a date."""
    digits = re.sub(r"\D", "", stamp)[:8]
    if len(digits) != 8:
        raise ValueError(f"unrecognised GDELT date: {stamp!r}")
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


def _stamp(day: date, end_of_day: bool = False) -> str:
    return day.strftime("%Y%m%d") + ("235959" if end_of_day else "000000")


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ailab-nlp/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body: bytes = response.read()
        return body


# ========================================
# 📂 GDELT Client
# ========================================
class GdeltClient:
    """Client for the GDELT DOC 2.0 API (free; please keep request rates low)."""

    def __init__(
        self,
        *,
        fetch: Callable[[str], bytes] = _http_get,
        min_interval: float = 5.0,
        attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._fetch = retry(attempts=attempts, exceptions=(OSError,), sleep=sleep)(fetch)
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def _get(self, **params: str | int) -> dict[str, Any]:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        url = f"{GDELT_DOC_API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
        body = self._fetch(url)
        self._last_request = self._clock()
        text = body.decode("utf-8", errors="replace").strip()
        if not text:
            return {}  # GDELT answers an empty body when nothing matches
        try:
            parsed: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"GDELT did not return JSON: {text[:200]}") from None
        return parsed

    def articles(
        self, query: str, *, start: date, end: date, max_records: int = MAX_RECORDS
    ) -> list[NewsArticle]:
        """Article titles and links for ``query`` (GDELT keeps roughly the last 3 months)."""
        if not 1 <= max_records <= MAX_RECORDS:
            raise ValueError(f"max_records must be between 1 and {MAX_RECORDS}")
        data = self._get(
            query=query,
            mode="artlist",
            sort="DateAsc",
            maxrecords=max_records,
            startdatetime=_stamp(start),
            enddatetime=_stamp(end, end_of_day=True),
        )
        return [
            NewsArticle(
                published=_day(item.get("seendate", "")),
                title=item.get("title", "").strip(),
                url=item.get("url", ""),
                source=item.get("domain", ""),
                language=item.get("language", ""),
            )
            for item in data.get("articles", [])
        ]

    def _timeline(self, mode: str, query: str, start: date, end: date) -> dict[date, list[Any]]:
        data = self._get(
            query=query,
            mode=mode,
            startdatetime=_stamp(start),
            enddatetime=_stamp(end, end_of_day=True),
        )
        by_day: dict[date, list[Any]] = {}
        series = data.get("timeline", [])
        for point in series[0]["data"] if series else []:
            by_day.setdefault(_day(point["date"]), []).append(point)
        return by_day

    def daily_tone(
        self, query: str, *, start: date, end: date, tone_scale: float = 10.0
    ) -> list[DailyTone]:
        """Daily average tone (mapped to -1..+1) and article counts, oldest first."""
        if tone_scale <= 0:
            raise ValueError("tone_scale must be positive")
        tone = self._timeline("timelinetone", query, start, end)
        volume = self._timeline("timelinevolraw", query, start, end)
        out = []
        for day in sorted(tone):
            # Sub-daily points (short ranges) are averaged into one value per day.
            raw = sum(float(p["value"]) for p in tone[day]) / len(tone[day])
            count = int(sum(float(p.get("value", 0)) for p in volume.get(day, [])))
            out.append(DailyTone(day, max(-1.0, min(1.0, raw / tone_scale)), count))
        return out


# ========================================
# 📂 Rolling Tone for Decisions
# ========================================
def rolling_tone(
    daily: Sequence[DailyTone],
    *,
    as_of: date,
    window_days: int = 7,
    half_life_days: float = 3.0,
    include_same_day: bool = False,
) -> ToneSummary:
    """Article-weighted, time-decayed news tone known on ``as_of``."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    last = as_of if include_same_day else as_of - timedelta(days=1)
    first = as_of - timedelta(days=window_days)
    window = [d for d in daily if first <= d.day <= last and d.count > 0]
    if not window:
        return ToneSummary(0.0, 0, 0)  # no coverage is a valid state: "no news"
    decay = time_decay_weights([(as_of - d.day).days for d in window], half_life_days)
    weights = [w * d.count for w, d in zip(decay, window, strict=True)]
    score = sum(w * d.score for w, d in zip(weights, window, strict=True)) / sum(weights)
    return ToneSummary(score, sum(d.count for d in window), len(window))
