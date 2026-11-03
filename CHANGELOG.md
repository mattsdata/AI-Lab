# Changelog

## [Unreleased]

### Added
- `climate` framework V1.0: Open-Meteo (ERA5) daily history and NASA GISTEMP clients, six
  annual indices (mean temperature, hot days, warm nights, rainfall, heavy rain days, wettest
  day), place report with trends per decade, 95% intervals, plain verdicts and global context;
  Kollam demo. Frameworks-independence rule now checks finance and climate.
- `timeseries` package V1.0: causal rolling mean/std, calendar aggregation with a coverage
  rule, linear trend with autocorrelation-adjusted confidence interval (Monte Carlo tested),
  calendar-day climatology, anomalies and threshold flags. Extracted on second use (ADR-0003).
- `nlp` V1.2: RAG stack. Traceable chunking, HTML to text, BM25 / embedding / hybrid (RRF)
  retrieval, `LanguageModel` interface with Ollama and OpenAI-compatible backends, `RagPipeline`
  with verified citations and abstention, extractive baseline, `evaluate_rag`. ADR-0005.
- `finance` V1.4: annual report analyser over 10-K filings (EDGAR filing lists and documents),
  review memo with quoted sources; demo. `ailab-nlp` is now a finance dependency.
- `ml` package V1.0: logistic regression (IRLS) and validation metrics (AUC, Gini, KS, Brier,
  calibration table, PSI), extracted from finance on second use (ADR-0003).
- `finance` V1.3: credit scorecard with WoE binning, IV selection, points scaling, reason codes,
  validation pack and group fairness check; offline demo.
- `finance` V1.2: SEC EDGAR fundamentals. Ticker to CIK lookup, company facts, point-in-time
  selection by filing date, KMV default point and shares outstanding with source filings; Early
  Warning demo uses them via `SEC_USER_AGENT`.
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
- `finance` 1.4.1: rolling volatility features use `ailab-timeseries` (output verified identical).
- `finance` forecast now uses `ailab-ml` for its logistic models (behaviour unchanged).
- `vision` V1.1: backend registry now provided by `mattslib.Registry` (ADR-0003 applied: the
  registry was extracted when `nlp` became its second user). Backend options may now be named `name`.
