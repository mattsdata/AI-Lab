# AI Lab

**Author:** Mathew Varghese · **Version:** V1.0

Reusable applied-AI components and domain frameworks, built in two layers so that
the same building blocks can serve any industry.

- **Packages** are horizontal: they solve one technical problem and know nothing about any domain.
- **Frameworks** are vertical: they combine packages with domain concepts for one field.

Packages never import frameworks, and frameworks never import each other. These rules are
enforced in CI with import-linter and explained in [ADR-0001](docs/adr/0001-layered-architecture.md).

## Architecture decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-layered-architecture.md) | Two layers: packages and frameworks, one-way dependencies |
| [0002](docs/adr/0002-decimal-money-float-analytics.md) | Decimal for money, float64 for analytics |
| [0003](docs/adr/0003-extract-shared-code-on-second-use.md) | Extract shared code into a package on its second use |
| [0004](docs/adr/0004-purged-walk-forward-against-a-baseline.md) | Judge forecasting models by purged walk-forward tests against a baseline |
| [0005](docs/adr/0005-rag-answers-cite-or-abstain.md) | RAG answers must cite verifiable sources or abstain |
| [0006](docs/adr/0006-wellbeing-tools-safety-first-non-diagnostic.md) | Wellbeing tools are safety-first, private and non-diagnostic |
| [0007](docs/adr/0007-heritage-ethics-and-location-protection.md) | Heritage tools propose candidates, demand evidence and protect locations |

## Packages

| Package | Scope | Status |
|---|---|---|
| [`vision`](packages/vision) | Object detection, tracking, video I/O | ✅ V1.1 |
| [`nlp`](packages/nlp) | Sentiment (lexicon, FinBERT), GDELT news, RAG (BM25 / embeddings / hybrid, LLM backends, verified citations, evaluation) | ✅ V1.2 |
| [`timeseries`](packages/timeseries) | Causal rolling statistics, coverage-aware aggregation, autocorrelation-aware trends, seasonal anomalies and thresholds | ✅ V1.0 |
| `speech` | Speech-to-text, text-to-speech | Planned |
| [`geo`](packages/geo) | Georeferenced rasters, terrain analysis (slope, hillshade, local relief), landform change by diffusion, region shape measures | ✅ V1.0 |
| [`ml`](packages/ml) | Explainable models (logistic regression) and validation metrics: AUC, Gini, KS, Brier, calibration, PSI | ✅ V1.0 |

The core utilities live in [`mattslib`](https://github.com/mattsdata/mattslib), including the
plugin registry that every package uses to select backends by name.

## Frameworks

| Framework | Domain | Status |
|---|---|---|
| [`finance`](frameworks/finance) | Stock Outlook model (market + news, walk-forward tested), Corporate Early Warning (Merton + SEC EDGAR), Credit Scorecard (WoE, points, reason codes, validation, fairness), Annual Report Analyser (RAG over 10-Ks) | ✅ V1.4 |
| [`climate`](frameworks/climate) | Local climate change for any place: Open-Meteo (ERA5) and NASA GISTEMP data, six ETCCDI-style indices, trends with honest uncertainty; Kollam demo | ✅ V1.0 |
| [`mental-health`](frameworks/mental-health) | Private, safety-first, non-diagnostic: PHQ-9 / GAD-7 scoring with safety alerts, crisis contacts, encrypted journal, gentle summaries | ✅ V1.0 |
| [`heritage`](frameworks/heritage) | Landscape archaeology from elevation data: candidate earthworks by shape, evidence-based status, location protection, landscape change over 10 / 500 / 1000 years | ✅ V1.0 |

## Principles

- Every result records its provenance: data source, version and parameters.
- Every estimate reports its uncertainty.
- Personal data is never logged or sent outside the user's machine by default.
- Model weights and datasets are never committed. Download scripts are provided instead.
- Libraries are wrapped, not copied. Each dependency's licence is listed in its package README.

## Licence

MIT © Mathew Varghese
