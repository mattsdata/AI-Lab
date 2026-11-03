# ==================================================================================================
# 📜 ailab-finance :: fundamentals
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/fundamentals.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.1
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Credit risk needs balance-sheet facts: how much a company owes and how many shares exist.
#    This module reads them from SEC EDGAR, the official, free source of US company filings,
#    as they were known on a given date, with the exact filing behind every number.
#
#  🔧 WHAT IT DOES
#    - EdgarClient     : ticker -> CIK lookup and XBRL company facts from data.sec.gov.
#    - latest_fact     : the most recent value of a concept that had been FILED by a date.
#    - credit_inputs   : KMV-style default point (current liabilities + half of long-term debt)
#                        and shares outstanding, each with its source filing.
#    - filings         : a company's filings of one form (e.g. 10-K), newest first, as of a date.
#    - filing_document : the primary document of a filing (HTML), for text analysis and RAG.
#    - Fact, CreditInputs : results that carry their provenance (concept, period, filing, form).
#
#  🧠 DESIGN NOTES
#    - Point in time by FILING date, not period end. A 31 December balance sheet is usually filed
#      in February; using it in January would be look-ahead. `as_of` filters on `filed`.
#    - Companies tag the same idea differently (LongTermDebtNoncurrent vs LongTermDebt), so each
#      input tries an ordered list of concepts and records which one it used.
#    - Missing long-term debt is treated as zero only explicitly (`long_term_debt is None`),
#      so a reviewer can see the assumption instead of a silently smaller default point.
#    - SEC fair access: a declared User-Agent ("Name email@domain") is required and requests
#      are spaced to stay under 10 per second. The caller supplies the User-Agent; nothing
#      personal is hard-coded. Responses are cached per client to avoid repeat downloads.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: EDGAR client, point-in-time facts, credit inputs with provenance.
#
#    V1.1 (2026-11-03) - Mathew Varghese
#      - Added filing lists and primary documents (submissions API, EDGAR archives).
# ==================================================================================================
"""SEC EDGAR company facts, point in time, with provenance."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from mattslib import retry

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

CURRENT_LIABILITIES = ("LiabilitiesCurrent",)
LONG_TERM_DEBT = (
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
)
SHARES_OUTSTANDING = ("EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding")


# ========================================
# 📂 Result Types
# ========================================
@dataclass(frozen=True, slots=True)
class Fact:
    concept: str
    value: float
    unit: str
    period_end: date
    filed: date
    form: str  # e.g. 10-K, 10-Q
    accession: str  # EDGAR filing id, to find the source document


@dataclass(frozen=True, slots=True)
class CreditInputs:
    ticker: str
    as_of: date | None
    current_liabilities: Fact
    long_term_debt: Fact | None  # None: no tagged long-term debt found, treated as zero
    shares_outstanding: Fact

    @property
    def default_point(self) -> float:
        """KMV convention: short-term liabilities plus half of long-term debt."""
        ltd = self.long_term_debt.value if self.long_term_debt else 0.0
        return self.current_liabilities.value + 0.5 * ltd

    def provenance(self) -> str:
        facts = [self.current_liabilities, self.long_term_debt, self.shares_outstanding]
        return "\n".join(
            f"{f.concept}: {f.value:,.0f} {f.unit} (period {f.period_end}, "
            f"{f.form} filed {f.filed}, accession {f.accession})"
            for f in facts
            if f is not None
        )


@dataclass(frozen=True, slots=True)
class Filing:
    ticker: str
    cik: int
    form: str
    filed: date
    accession: str
    document: str  # primary document file name

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.document
        )


# ========================================
# 📂 Point-in-Time Selection
# ========================================
def latest_fact(
    facts: dict[str, Any], concepts: Sequence[str], unit: str, *, as_of: date | None = None
) -> Fact | None:
    """Most recent value of the first available concept, among filings made on or before as_of."""
    for concept in concepts:
        for taxonomy in ("us-gaap", "dei"):
            entries = facts.get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
            known = [e for e in entries if as_of is None or date.fromisoformat(e["filed"]) <= as_of]
            if known:
                best = max(known, key=lambda e: (e["end"], e["filed"]))
                return Fact(
                    concept=concept,
                    value=float(best["val"]),
                    unit=unit,
                    period_end=date.fromisoformat(best["end"]),
                    filed=date.fromisoformat(best["filed"]),
                    form=str(best.get("form", "")),
                    accession=str(best.get("accn", "")),
                )
    return None


# ========================================
# 📂 EDGAR Client
# ========================================
class EdgarClient:
    """Minimal SEC EDGAR client. ``user_agent`` must identify you, e.g. "Jane Doe jane@x.com"."""

    def __init__(
        self,
        user_agent: str,
        *,
        fetch: Callable[[str, str], bytes] | None = None,
        min_interval: float = 0.15,  # stays below SEC's 10 requests per second
        attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with contact email, e.g. 'Name a@b.com'")
        self._user_agent = user_agent
        self._fetch = retry(attempts=attempts, exceptions=(OSError,), sleep=sleep)(
            fetch or self._http_get
        )
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._tickers: dict[str, int] | None = None
        self._facts: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _http_get(url: str, user_agent: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=30) as response:
            body: bytes = response.read()
            return body

    def _get_bytes(self, url: str) -> bytes:
        if self._last is not None:
            wait = self._min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        body = self._fetch(url, self._user_agent)
        self._last = self._clock()
        return body

    def _get_json(self, url: str) -> Any:
        return json.loads(self._get_bytes(url))

    def cik(self, ticker: str) -> int:
        """Central Index Key for a ticker (SEC's company_tickers.json, cached)."""
        if self._tickers is None:
            rows = self._get_json(TICKERS_URL).values()
            self._tickers = {str(r["ticker"]).upper(): int(r["cik_str"]) for r in rows}
        try:
            return self._tickers[ticker.upper()]
        except KeyError:
            raise LookupError(f"ticker {ticker!r} not found in SEC's ticker list") from None

    def company_facts(self, cik: int) -> dict[str, Any]:
        """All XBRL facts filed by a company, grouped by taxonomy (cached per CIK)."""
        if cik not in self._facts:
            self._facts[cik] = self._get_json(FACTS_URL.format(cik=cik)).get("facts", {})
        return self._facts[cik]

    def credit_inputs(self, ticker: str, *, as_of: date | None = None) -> CreditInputs:
        """Default point and shares outstanding as known on ``as_of`` (by filing date)."""
        facts = self.company_facts(self.cik(ticker))
        liabilities = latest_fact(facts, CURRENT_LIABILITIES, "USD", as_of=as_of)
        shares = latest_fact(facts, SHARES_OUTSTANDING, "shares", as_of=as_of)
        if liabilities is None or shares is None:
            missing = "current liabilities" if liabilities is None else "shares outstanding"
            raise LookupError(f"{ticker}: no {missing} filed by {as_of or 'today'}")
        return CreditInputs(
            ticker=ticker.upper(),
            as_of=as_of,
            current_liabilities=liabilities,
            long_term_debt=latest_fact(facts, LONG_TERM_DEBT, "USD", as_of=as_of),
            shares_outstanding=shares,
        )

    def filings(
        self, ticker: str, form: str = "10-K", *, as_of: date | None = None, limit: int = 5
    ) -> list[Filing]:
        """Recent filings of ``form`` filed on or before ``as_of``, newest first."""
        cik = self.cik(ticker)
        recent = self._get_json(SUBMISSIONS_URL.format(cik=cik))["filings"]["recent"]
        rows = zip(
            recent["form"],
            recent["filingDate"],
            recent["accessionNumber"],
            recent["primaryDocument"],
            strict=True,
        )
        found = [
            Filing(ticker.upper(), cik, f, date.fromisoformat(d), accession, document)
            for f, d, accession, document in rows
            if f == form and (as_of is None or date.fromisoformat(d) <= as_of)
        ]
        return sorted(found, key=lambda f: f.filed, reverse=True)[:limit]

    def filing_document(self, filing: Filing) -> str:
        """The filing's primary document (usually HTML) as text."""
        return self._get_bytes(filing.url).decode("utf-8", errors="replace")
