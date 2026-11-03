# ADR-0002: Decimal for money, float64 for analytics

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

Binary floating point cannot represent most decimal fractions exactly: `0.1 + 0.2` is
`0.30000000000000004`. In a ledger, fee schedule or statement, those errors accumulate and
reconciliations fail by pennies. Statistical analysis, on the other hand, runs over millions
of values where speed matters and errors of 1e-16 are irrelevant.

## Decision

- **Cash amounts** (balances, fees, prices on a contract note) use `Money`, backed by
  `decimal.Decimal`. `Money` refuses `float` input and float multipliers, so imprecision
  cannot enter by accident. Rounding is explicit and half-up, as on statements.
- **Analytics** (returns, volatility, risk ratios, model inputs) use NumPy `float64` arrays.

## Consequences

**Positive**
- Ledger arithmetic is exact and reconciles.
- Analytics stay fast and vectorised.

**Trade-offs**
- Converting between the two worlds must be explicit (`Decimal(str(x))` when an analytic
  result becomes a booked amount).
- Two numeric types to understand; the boundary is documented in `types.py`.
