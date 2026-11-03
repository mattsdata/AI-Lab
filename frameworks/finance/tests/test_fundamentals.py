# ==================================================================================================
# 📜 ailab-finance :: tests :: fundamentals
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/tests/test_fundamentals.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves point-in-time selection by filing date, concept fallbacks, the default-point
#    arithmetic, SEC fair-access behaviour (User-Agent, spacing, caching) and error messages.
#
#  🧠 TESTING APPROACH
#    A fake EDGAR serves SEC-shaped JSON and records URLs and headers. The key case: a balance
#    sheet for period end 31 Dec that was FILED on 20 Feb must be invisible on 10 Feb.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for EdgarClient, latest_fact and CreditInputs.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import json
from datetime import date
from typing import Any

import pytest

from ailab_finance.fundamentals import EdgarClient, latest_fact


def entry(
    end: str, val: float, filed: str, form: str = "10-Q", accn: str = "0001"
) -> dict[str, Any]:
    return {"end": end, "val": val, "filed": filed, "form": form, "accn": accn, "fy": 2025}


FACTS = {
    "facts": {
        "us-gaap": {
            "LiabilitiesCurrent": {
                "units": {
                    "USD": [
                        entry("2025-09-30", 400.0, "2025-11-01", accn="Q3"),
                        entry("2025-12-31", 500.0, "2026-02-20", "10-K", accn="FY"),
                    ]
                }
            },
            "LongTermDebt": {"units": {"USD": [entry("2025-09-30", 300.0, "2025-11-01")]}},
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [entry("2025-10-20", 1_000.0, "2025-11-01")]}
            }
        },
    }
}
TICKERS = {"0": {"cik_str": 320193, "ticker": "ACME", "title": "Acme Corp"}}
SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K", "8-K", "10-K"],
            "filingDate": ["2026-05-01", "2026-02-20", "2026-01-10", "2025-02-18"],
            "accessionNumber": [
                "0001-26-000003",
                "0001-26-000002",
                "0001-26-000001",
                "0001-25-000009",
            ],
            "primaryDocument": ["q1.htm", "acme-2025.htm", "8k.htm", "acme-2024.htm"],
        }
    }
}


class FakeEdgar:
    def __init__(self, failures: int = 0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.failures = failures
        self.now = 0.0
        self.slept: list[float] = []

    def fetch(self, url: str, user_agent: str) -> bytes:
        self.calls.append((url, user_agent))
        if self.failures:
            self.failures -= 1
            raise TimeoutError("timed out")
        if "Archives" in url:
            return b"<html><p>Annual report</p></html>"
        if "submissions" in url:
            return json.dumps(SUBMISSIONS).encode()
        return json.dumps(TICKERS if "company_tickers" in url else FACTS).encode()

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def client(self) -> EdgarClient:
        return EdgarClient(
            "AI Lab research contact@example.com",
            fetch=self.fetch,
            clock=lambda: self.now,
            sleep=self.sleep,
        )


# ========================================
# 📂 Point-in-Time Tests
# ========================================
def test_filing_date_not_period_end_decides_what_was_known() -> None:
    facts = FACTS["facts"]
    before = latest_fact(facts, ["LiabilitiesCurrent"], "USD", as_of=date(2026, 2, 10))
    after = latest_fact(facts, ["LiabilitiesCurrent"], "USD", as_of=date(2026, 2, 20))
    assert before is not None and before.value == 400.0 and before.accession == "Q3"
    assert after is not None and after.value == 500.0 and after.form == "10-K"
    assert latest_fact(facts, ["LiabilitiesCurrent"], "USD", as_of=date(2025, 1, 1)) is None


def test_concept_fallback_and_unit() -> None:
    facts = FACTS["facts"]
    ltd = latest_fact(facts, ["LongTermDebtNoncurrent", "LongTermDebt"], "USD")
    assert ltd is not None and ltd.concept == "LongTermDebt"  # first concept absent: fell back
    assert latest_fact(facts, ["LiabilitiesCurrent"], "EUR") is None


# ========================================
# 📂 Client Tests
# ========================================
def test_credit_inputs_with_provenance() -> None:
    fake = FakeEdgar()
    inputs = fake.client().credit_inputs("acme", as_of=date(2026, 1, 15))
    assert inputs.default_point == 400.0 + 0.5 * 300.0  # KMV: CL + LTD / 2
    assert inputs.shares_outstanding.value == 1_000.0
    text = inputs.provenance()
    assert "LiabilitiesCurrent: 400" in text and "accession Q3" in text and "10-Q filed" in text
    assert fake.calls[1][0].endswith("CIK0000320193.json")  # zero-padded to 10 digits
    assert all(agent == "AI Lab research contact@example.com" for _, agent in fake.calls)


def test_responses_cached_and_requests_spaced() -> None:
    fake = FakeEdgar(failures=1)
    client = fake.client()
    client.credit_inputs("ACME")
    client.credit_inputs("ACME", as_of=date(2026, 3, 1))
    assert len(fake.calls) == 3  # 1 failed + tickers + facts; second call fully cached
    assert any(s == pytest.approx(0.15) for s in fake.slept)  # spacing before the facts request


def test_missing_long_term_debt_is_explicit() -> None:
    fake = FakeEdgar()
    client = fake.client()
    facts = client.company_facts(client.cik("ACME"))
    del facts["us-gaap"]["LongTermDebt"]
    inputs = client.credit_inputs("ACME")
    assert inputs.long_term_debt is None and inputs.default_point == 500.0


def test_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient("my-bot")
    client = FakeEdgar().client()
    with pytest.raises(LookupError, match="not found"):
        client.cik("NOPE")
    with pytest.raises(LookupError, match="no current liabilities filed by 2020-01-01"):
        client.credit_inputs("ACME", as_of=date(2020, 1, 1))


def test_default_http_fetch_sends_declared_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(TICKERS).encode()

    def fake_urlopen(request: Any, timeout: float) -> Response:
        seen["agent"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert EdgarClient("Research team team@example.com").cik("acme") == 320193
    assert seen["agent"] == "Research team team@example.com"


def test_filings_are_point_in_time_and_documents_load() -> None:
    fake = FakeEdgar()
    client = fake.client()
    latest, previous = client.filings("acme", "10-K")
    assert (latest.filed, latest.document) == (date(2026, 2, 20), "acme-2025.htm")
    assert previous.filed == date(2025, 2, 18)
    assert latest.url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000126000002/acme-2025.htm"
    )
    (known,) = client.filings("ACME", "10-K", as_of=date(2026, 1, 31))  # before Feb filing
    assert known.document == "acme-2024.htm"
    assert client.filings("ACME", "10-K", limit=1) == [latest]
    assert "Annual report" in client.filing_document(latest)
