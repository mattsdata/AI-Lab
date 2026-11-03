# Changelog

## [Unreleased]

### Added
- `nlp` V1.1: GDELT news connector (daily tone history from 2017, recent article lists) with
  polite rate limiting and retries, and `rolling_tone` that excludes future and same-day news.
  Stock Outlook demo gains `--gdelt QUERY`.
- `finance` V1.1: Stock Outlook model. Causal market and news-sentiment features, ridge +
  logistic model with base-rate fallback for rare events, purged walk-forward test with Brier
  skill vs the historical baseline, `model_outlook`, demo. ADR-0004.
- `nlp` package V1.0: SentimentResult and analyzer interface, lexicon baseline with negation,
  FinBERT backend, time-decay aggregation with a look-ahead guard, demo, CI.
- `finance` framework V1.0 foundations: Instrument, PriceSeries (look-ahead guard), exact
  Money, CSV and Yahoo data sources, risk metrics, naive Merton distance to default with
  early warning flags, historical return outlook, demo report, CI.
- Architecture CI: import-linter contracts for ADR-0001. ADR-0002 and ADR-0003.
- `vision` package V1.0: model-independent detection types, Detector protocol and
  backend registry, YOLO backend, IoU tracker, video helpers, demo, CI on Python 3.10-3.13.
- MIT licence file.
- Repository structure, layered-architecture ADR and licence.

### Changed
- `vision` V1.1: backend registry now provided by `mattslib.Registry` (ADR-0003 applied: the
  registry was extracted when `nlp` became its second user). Backend options may now be named `name`.
