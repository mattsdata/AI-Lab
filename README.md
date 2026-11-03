# AI Lab

**Author:** Mathew Varghese · **Version:** V1.0

Reusable applied-AI components and domain frameworks, built in two layers so that
the same building blocks can serve any industry.

- **Packages** are horizontal: they solve one technical problem and know nothing about any domain.
- **Frameworks** are vertical: they combine packages with domain concepts for one field.

Packages never import frameworks. This rule is enforced in CI and explained in
[ADR-0001](docs/adr/0001-layered-architecture.md).

## Packages

| Package | Scope | Status |
|---|---|---|
| [`vision`](packages/vision) | Object detection, tracking, video I/O | ✅ V1.0 |
| `nlp` | LLM backend interface, RAG, evaluation | Planned |
| `timeseries` | Forecasting, anomaly detection | Planned |
| `speech` | Speech-to-text, text-to-speech | Planned |
| `geo` | Raster, terrain and map utilities | Planned |
| `ml-core` | Training loop, metrics, seed control | Planned |

The core utilities live in [`mattslib`](https://github.com/mattsdata/mattslib).

## Frameworks (planned)

| Framework | Domain | Status |
|---|---|---|
| `finance` | Instruments, portfolios, backtesting, risk | Planned |
| `climate` | Station data, anomalies, scenarios | Planned |
| `mental-health` | Self-tracking and research tools, privacy first | Planned |
| `heritage` | Sites, sources, evidence with confidence | Planned |

## Principles

- Every result records its provenance: data source, version and parameters.
- Every estimate reports its uncertainty.
- Personal data is never logged or sent outside the user's machine by default.
- Model weights and datasets are never committed. Download scripts are provided instead.
- Libraries are wrapped, not copied. Each dependency's licence is listed in its package README.

## Licence

MIT © Mathew Varghese
