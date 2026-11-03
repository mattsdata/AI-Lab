# ==================================================================================================
# 📜 ailab-finance :: examples :: stock_outlook
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/examples/stock_outlook.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.1
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Scenario 1, end to end: for one stock, show the historical baseline outlook, the model
#    outlook (market features, optionally news sentiment), and the walk-forward evidence of
#    whether the model is actually better than history.
#
#  🔧 WHAT IT DOES
#    prices (Yahoo) + optional news: GDELT daily tone history, or dated headlines (TSV)
#    scored by ailab-nlp
#    -> baseline outlook -> model outlook -> purged walk-forward report
#
#  🧠 DESIGN NOTES
#    - Headlines are scored once, then each decision date sees only headlines published on or
#      before it, weighted by age (half-life). This is where finance meets the nlp package.
#    - For a meaningful news effect, news must cover the whole price history. --gdelt fetches
#      GDELT's daily tone from 2017 onwards; a headline file must span the period itself.
#      Days without coverage count as "no news".
#    - Research and education only. Not financial advice.
#
#  USAGE
#    pip install -e "frameworks/finance[yahoo]"
#    python frameworks/finance/examples/stock_outlook.py MSFT
#    python frameworks/finance/examples/stock_outlook.py MSFT --gdelt '"Microsoft"'
#    python frameworks/finance/examples/stock_outlook.py MSFT --headlines msft.tsv --backend finbert
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: baseline vs model outlook with walk-forward evidence.
#
#    V1.1 (2026-11-03) - Mathew Varghese
#      - Added --gdelt: years of daily news tone from GDELT as the sentiment input.
# ==================================================================================================
"""Stock Outlook: baseline vs model, with walk-forward evidence."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from ailab_nlp import (
    GdeltClient,
    SentimentSummary,
    ToneSummary,
    aggregate_sentiment,
    create_analyzer,
    rolling_tone,
    time_decay_weights,
)

from ailab_finance import (
    Instrument,
    YahooSource,
    build_features,
    historical_outlook,
    model_outlook,
    walk_forward,
)


# ========================================
# 📂 News Sentiment Provider
# ========================================
def news_provider(path: Path, backend: str, half_life: float, window_days: int = 30) -> Any:
    """Score dated headlines once; return a function giving sentiment as known on a date."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            day, text = line.split("\t", 1)
            rows.append((np.datetime64(day.strip(), "D"), text.strip()))
    results = create_analyzer(backend).analyze([text for _, text in rows])
    published = np.array([d for d, _ in rows], dtype="datetime64[D]")

    def provider(day: np.datetime64) -> SentimentSummary:
        age = (np.datetime64(day, "D") - published).astype(int)
        known = np.flatnonzero((age >= 0) & (age <= window_days))  # past only: no look-ahead
        weights = time_decay_weights([float(age[i]) for i in known], half_life)
        return aggregate_sentiment([results[i] for i in known], weights)

    return provider


def gdelt_provider(query: str, start: date) -> Any:
    """Fetch GDELT daily tone once; return a function giving the 7-day mood before a date."""
    daily = GdeltClient().daily_tone(query, start=start, end=date.today())
    print(f"GDELT: {len(daily)} days of news tone for {query!r}")

    def provider(day: np.datetime64) -> ToneSummary:
        return rolling_tone(daily, as_of=np.datetime64(day, "D").astype(date), window_days=7)

    return provider


# ========================================
# 📂 Report
# ========================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol")
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--gdelt", metavar="QUERY", help="use GDELT news tone for this query")
    parser.add_argument("--headlines", type=Path, help="TSV: YYYY-MM-DD<TAB>headline")
    parser.add_argument("--backend", default="lexicon", help="lexicon or finbert")
    parser.add_argument("--half-life", type=float, default=7.0)
    args = parser.parse_args()

    start = (date.today() - timedelta(days=365 * args.years)).isoformat()
    prices = YahooSource().load(Instrument(args.symbol), start=start)
    sentiment = None
    if args.gdelt:
        sentiment = gdelt_provider(args.gdelt, date.fromisoformat(start))
    elif args.headlines:
        sentiment = news_provider(args.headlines, args.backend, args.half_life)

    print("\n=== 1. Historical baseline ===")
    print(historical_outlook(prices).summary())
    print("\n=== 2. Model outlook ===")
    print(model_outlook(prices, sentiment=sentiment).summary())
    print("\n=== 3. Is the model better than history? (out of sample) ===")
    print(walk_forward(build_features(prices, sentiment=sentiment)).summary())
    print("\nTrust the model only where its skill is positive; otherwise use the baseline.\n")


if __name__ == "__main__":
    main()
