# ==================================================================================================
# 📜 ailab-finance :: examples :: early_warning
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/examples/early_warning.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    One-page Corporate Early Warning report for a listed company: equity outlook (historical
#    baseline) and market-based credit risk (naive Merton distance to default).
#
#  🔧 WHAT IT DOES
#    download prices -> outlook (+10/+25/+50% probabilities) -> volatility and 1-year return
#    -> distance to default and default probability -> GREEN / AMBER / RED flag
#
#  🧠 DESIGN NOTES
#    - Market cap and debt can be passed explicitly (--market-cap, --debt) so an analyst can
#      apply their bank's default-point convention; otherwise Yahoo's figures are used and the
#      report says so.
#    - Research and education only. Not financial advice.
#
#  USAGE
#    pip install -e "frameworks/finance[yahoo]"
#    python frameworks/finance/examples/early_warning.py MSFT
#    python frameworks/finance/examples/early_warning.py MSFT --debt 6e10 --market-cap 3e12
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: combined equity outlook and credit risk report.
# ==================================================================================================
"""Corporate Early Warning report for one listed company."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any

import yfinance
from mattslib import setup_logging

from ailab_finance import (
    DISCLAIMER,
    Instrument,
    YahooSource,
    annualised_volatility,
    historical_outlook,
    max_drawdown,
    naive_distance_to_default,
    simple_returns,
    warning_flag,
)


# ========================================
# 📂 Command Line
# ========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="ticker, e.g. MSFT, or HSBA.L for London")
    parser.add_argument("--years", type=int, default=5, help="years of price history to use")
    parser.add_argument("--market-cap", type=float, help="equity value (default: Yahoo)")
    parser.add_argument("--debt", type=float, help="default point / debt (default: Yahoo)")
    return parser.parse_args()


def company_figures(symbol: str) -> dict[str, Any]:
    info: dict[str, Any] = yfinance.Ticker(symbol).info or {}
    return {"market_cap": info.get("marketCap"), "debt": info.get("totalDebt")}


# ========================================
# 📂 Report
# ========================================
def main() -> None:
    args = parse_args()
    setup_logging("WARNING", json_format=False)
    instrument = Instrument(args.symbol)
    start = (date.today() - timedelta(days=365 * args.years)).isoformat()
    prices = YahooSource().load(instrument, start=start)

    print(f"\n=== Corporate Early Warning · {args.symbol} · data to {prices.end} ===\n")
    print("-- Equity outlook (1 month) --")
    print(historical_outlook(prices, horizon=21).summary())

    last_year = prices.close[-253:]
    volatility = annualised_volatility(simple_returns(last_year))
    one_year_return = float(last_year[-1] / last_year[0] - 1)
    print("\n-- Market risk (last 12 months) --")
    print(
        f"Volatility: {volatility:.1%}   Return: {one_year_return:+.1%}   "
        f"Max drawdown: {max_drawdown(last_year):.1%}"
    )

    figures = company_figures(args.symbol)
    market_cap = args.market_cap or figures["market_cap"]
    debt = args.debt or figures["debt"]
    print("\n-- Credit risk (naive Merton, 1 year) --")
    if not market_cap or not debt:
        print("Market cap or debt unavailable; pass --market-cap and --debt.")
    else:
        source = "user input" if args.debt else "Yahoo totalDebt"
        estimate = naive_distance_to_default(
            equity_value=market_cap,
            debt=debt,
            equity_volatility=volatility,
            expected_return=one_year_return,
        )
        print(f"Debt used: {debt:,.0f} ({source})   Leverage: {debt / (market_cap + debt):.0%}")
        print(
            f"Distance to default: {estimate.distance_to_default:.2f}   "
            f"Model-implied PD: {estimate.default_probability:.2%}   "
            f"Flag: {warning_flag(estimate)}"
        )
    print(f"\n{DISCLAIMER}\n")


if __name__ == "__main__":
    main()
