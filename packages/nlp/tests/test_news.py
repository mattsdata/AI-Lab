# ==================================================================================================
# 📜 ailab-nlp :: tests :: news
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/tests/test_news.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the GDELT client builds correct requests, parses responses, retries network errors,
#    respects the request interval, and that rolling_tone never uses future or same-day news.
#
#  🧠 TESTING APPROACH
#    A fake HTTP function returns GDELT-shaped JSON per mode and records every URL; a fake clock
#    and sleep make rate limiting testable in microseconds. No network access is needed.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for GdeltClient and rolling_tone.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import json
import urllib.parse
from datetime import date
from typing import Any

import pytest

from ailab_nlp.news import DailyTone, GdeltClient, _day, rolling_tone

ARTICLES = {
    "articles": [
        {
            "url": "https://example.com/a",
            "title": " Profits surged ",
            "seendate": "20261101T093000Z",
            "domain": "example.com",
            "language": "English",
        }
    ]
}
TONE = {
    "timeline": [
        {
            "series": "Average Tone",
            "data": [
                {"date": "20261101T000000Z", "value": -2.0},
                {"date": "20261101T120000Z", "value": -4.0},  # sub-daily point, same day
                {"date": "20261102T000000Z", "value": 15.0},  # beyond the scale: clipped
            ],
        }
    ]
}
VOLUME = {
    "timeline": [
        {
            "series": "Article Count",
            "data": [
                {"date": "20261101T000000Z", "value": 30, "norm": 1000},
                {"date": "20261102T000000Z", "value": 5, "norm": 900},
            ],
        }
    ]
}


class FakeGdelt:
    def __init__(self, failures: int = 0, body: bytes | None = None) -> None:
        self.urls: list[str] = []
        self.failures = failures
        self.body = body
        self.now = 0.0
        self.slept: list[float] = []

    def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        if self.failures:
            self.failures -= 1
            raise ConnectionError("reset by peer")
        if self.body is not None:
            return self.body
        mode = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["mode"][0]
        payload = {"artlist": ARTICLES, "timelinetone": TONE, "timelinevolraw": VOLUME}[mode]
        return json.dumps(payload).encode()

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def client(self, **kwargs: Any) -> GdeltClient:
        return GdeltClient(fetch=self.fetch, clock=lambda: self.now, sleep=self.sleep, **kwargs)


def params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


# ========================================
# 📂 GDELT Client Tests
# ========================================
def test_articles_request_and_parsing() -> None:
    fake = FakeGdelt()
    (article,) = fake.client().articles(
        '"Acme Corp" sourcelang:english', start=date(2026, 10, 1), end=date(2026, 11, 2)
    )
    assert article.published == date(2026, 11, 1) and article.title == "Profits surged"
    assert article.source == "example.com" and article.language == "English"
    p = params(fake.urls[0])
    assert p["query"] == '"Acme Corp" sourcelang:english'
    assert (p["mode"], p["format"], p["maxrecords"]) == ("artlist", "json", "250")
    assert (p["startdatetime"], p["enddatetime"]) == ("20261001000000", "20261102235959")
    with pytest.raises(ValueError):
        fake.client().articles("x", start=date(2026, 1, 1), end=date(2026, 1, 2), max_records=500)


def test_daily_tone_merges_tone_and_volume() -> None:
    fake = FakeGdelt()
    first, second = fake.client(min_interval=0).daily_tone(
        "acme", start=date(2026, 11, 1), end=date(2026, 11, 2)
    )
    assert first == DailyTone(date(2026, 11, 1), -0.3, 30)  # mean(-2, -4) / 10
    assert second == DailyTone(date(2026, 11, 2), 1.0, 5)  # 15 / 10 clipped to 1
    assert [params(u)["mode"] for u in fake.urls] == ["timelinetone", "timelinevolraw"]
    with pytest.raises(ValueError):
        fake.client().daily_tone("acme", start=date(2026, 1, 1), end=date(2026, 1, 2), tone_scale=0)


def test_requests_are_spaced_and_network_errors_retried() -> None:
    fake = FakeGdelt(failures=1)
    client = fake.client(min_interval=5.0)
    client.articles("a", start=date(2026, 1, 1), end=date(2026, 1, 2))  # one retry inside
    client.articles("b", start=date(2026, 1, 1), end=date(2026, 1, 2))
    assert len(fake.urls) == 3  # failed attempt + 2 successes
    assert fake.slept[-1] == pytest.approx(5.0)  # waited before the second query


def test_empty_and_error_responses() -> None:
    assert (
        FakeGdelt(body=b"").client().articles("x", start=date(2026, 1, 1), end=date(2026, 1, 2))
        == []
    )
    assert (
        FakeGdelt(body=b"{}").client().daily_tone("x", start=date(2026, 1, 1), end=date(2026, 1, 2))
        == []
    )
    with pytest.raises(ValueError, match="did not return JSON: Invalid query"):
        FakeGdelt(body=b"Invalid query syntax").client().articles(
            "(", start=date(2026, 1, 1), end=date(2026, 1, 2)
        )


def test_date_parsing() -> None:
    assert _day("20240115T093000Z") == _day("2024-01-15") == date(2024, 1, 15)
    with pytest.raises(ValueError):
        _day("yesterday")


# ========================================
# 📂 Rolling Tone Tests
# ========================================
def history() -> list[DailyTone]:
    return [
        DailyTone(date(2026, 1, 1), -0.8, 10),
        DailyTone(date(2026, 1, 8), 0.5, 10),
        DailyTone(date(2026, 1, 9), 0.4, 0),  # no articles: ignored
        DailyTone(date(2026, 1, 10), 0.9, 100),  # the decision day itself
        DailyTone(date(2026, 1, 11), -1.0, 100),  # the future
    ]


def test_rolling_tone_excludes_future_and_same_day() -> None:
    summary = rolling_tone(history(), as_of=date(2026, 1, 10), window_days=10)
    assert summary.count == 20 and summary.days == 2  # Jan 1 and Jan 8 only
    assert -0.8 < summary.score < 0.5
    with_today = rolling_tone(history(), as_of=date(2026, 1, 10), include_same_day=True)
    assert with_today.score > summary.score and with_today.count == 110


def test_rolling_tone_weights_recent_and_busy_days() -> None:
    summary = rolling_tone(history(), as_of=date(2026, 1, 9), window_days=10, half_life_days=2)
    assert summary.score > 0.3  # Jan 8 (recent) dominates the older negative day


def test_rolling_tone_no_coverage_and_validation() -> None:
    assert rolling_tone([], as_of=date(2026, 1, 1)) == rolling_tone(
        history(), as_of=date(2025, 1, 1)
    )
    with pytest.raises(ValueError):
        rolling_tone(history(), as_of=date(2026, 1, 1), window_days=0)


def test_default_http_fetch_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"articles": []}'

    def fake_urlopen(request: Any, timeout: float) -> Response:
        seen["agent"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert GdeltClient().articles("x", start=date(2026, 1, 1), end=date(2026, 1, 2)) == []
    assert seen == {"agent": "ailab-nlp/1.0", "timeout": 30}
