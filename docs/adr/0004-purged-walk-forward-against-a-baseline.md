# ADR-0004: Judge forecasting models by purged walk-forward tests against a baseline

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

Financial forecasting models are easy to make look good and hard to make actually good.
The usual ways they fool their authors are:

1. **Look-ahead:** a feature, a scaling statistic or a news item from after the decision date.
2. **Label leakage:** a training label whose window overlaps the test date (a 21-day return
   that started 5 days before the prediction already "knows" part of the answer).
3. **No baseline:** an accuracy of 87% means nothing if "it never goes up 10%" scores 88%.
4. **Rare events:** fitting a classifier to a handful of +50% months produces confident noise.

## Decision

Every forecasting model in AI Lab is evaluated with a **purged, expanding-window walk-forward
test** and reported **against a baseline**:

- Features are causal (data up to day *t* only); scaling statistics come from training rows only.
- Training rows are purged: a row *i* is used for a prediction at *t* only if `i + horizon <= t`.
  The report states the smallest purge gap so it can be audited.
- Probabilities are scored with the Brier score and compared with the historical base rate
  as **Brier skill**: positive is better than history, zero is no better, negative is worse.
- Ranges are checked for **coverage**: an 80% range should contain about 80% of outcomes.
- Events with fewer than `min_events` examples are not modelled; the smoothed base rate is
  used and flagged.
- The report states its verdict in words ("no better than history") as well as numbers.

## Consequences

**Positive**
- The model is trusted only where it has earned it, and the evidence is reproducible.
- The same test applies to any future model (gradient boosting, LLM-derived features).
- Tests prove the honesty: on a pure random walk the model must not claim skill; when
  synthetic news carries real information it must find it.

**Trade-offs**
- Walk-forward is slower than a single split (one fit per test date).
- Overlapping monthly windows mean the effective number of independent tests is smaller than
  the count shown, which the reports acknowledge.
