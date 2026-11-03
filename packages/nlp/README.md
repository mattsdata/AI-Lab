# ailab-nlp

*Author: **Mathew Varghese** · Version 1.1*

Model-independent language components for any AI Lab framework. It provides
**sentiment analysis** (one interface, a transparent baseline, a financial transformer model,
time-weighted aggregation) and **news data** from the GDELT Project with look-ahead-safe rolling tone.

```python
from ailab_nlp import aggregate_sentiment, create_analyzer, time_decay_weights

analyzer = create_analyzer("finbert")  # or "lexicon": no dependencies, instant
results = analyzer.analyze(
    [
        "Profits surged to a record as growth beat forecasts",
        "Regulator opens investigation into accounting practices",
    ]
)
summary = aggregate_sentiment(results, time_decay_weights([0, 10], half_life_days=7))
print(summary.label, round(summary.score, 2))
```

## Components

| Module | What it does |
|---|---|
| `sentiment` | `SentimentResult` (label, score in [-1, 1], confidence), `SentimentAnalyzer` interface, backend registry |
| `backends/lexicon` | Word-list baseline with negation handling and smoothing. Dependency-free and explainable |
| `backends/finbert` | FinBERT via Hugging Face `transformers`; understands financial phrasing |
| `aggregate` | Exponential time-decay weights; weighted summary with label shares |
| `news` | `GdeltClient`: daily news tone history (2017 onwards) and recent article lists (about 3 months); `rolling_tone` for decision dates |

## Design decisions

- **Score = P(positive) - P(negative).** One signed number that averages and charts naturally.
- **A baseline that a model must beat.** The lexicon backend is the reference point. FinBERT
  earns its cost only if it is clearly better on labelled data for the task at hand.
- **No look-ahead.** Texts dated after the decision time raise an error in `time_decay_weights`.
- **Lazy heavy imports.** `transformers` and PyTorch load only when the `finbert` backend is created.
- **Shared registry.** Backends are registered through `mattslib.Registry`, the same mechanism the
  vision package uses (extracted on its second use, per ADR-0003).

## News history from GDELT

```python
from datetime import date
from ailab_nlp import GdeltClient, rolling_tone

gdelt = GdeltClient()  # polite: one request per 5 s by default
daily = gdelt.daily_tone('"Microsoft" sourcelang:english', start=date(2019, 1, 1), end=date.today())
mood = rolling_tone(daily, as_of=date(2024, 3, 1), window_days=7)  # excludes Mar 1 itself
recent = gdelt.articles('"Microsoft"', start=date.today().replace(day=1), end=date.today())
```

- **Two modes, two jobs.** GDELT timelines give years of daily tone for back-testing; article
  lists cover roughly the last three months and provide text for FinBERT.
- **Tone scale.** GDELT tone is typically within -10..+10 and is mapped to -1..+1 (`tone_scale=10`).
- **Same-day news excluded** by default, because a day's bucket can include stories published after
  the decision time.

## Install

```bash
pip install -e "packages/nlp"            # core + lexicon backend
pip install -e "packages/nlp[finbert]"   # + FinBERT (transformers, PyTorch)
```

## Demo

```bash
printf "2026-10-30\tProfits surged to a record\n2026-11-01\tShares plunged on a profit warning\n" > headlines.tsv
python packages/nlp/examples/news_sentiment.py headlines.tsv --as-of 2026-11-02
```

## Third-party licences

| Component | Used for | Licence |
|---|---|---|
| Hugging Face `transformers` | FinBERT backend (optional) | Apache-2.0 |
| PyTorch | FinBERT backend (optional) | BSD-style |
| `ProsusAI/finbert` model weights | Default FinBERT model | See the model card on Hugging Face before commercial use |
| GDELT Project data | News tone and article lists | Free and open; see GDELT's terms and cite GDELT when publishing results |

## Roadmap

- LLM backend interface (local via Ollama, or hosted APIs) for richer, explainable sentiment
- RAG pipeline with citation and evaluation (annual report analyser)
- Evaluation helpers: accuracy, macro-F1 and calibration against labelled samples

## Development

```bash
cd packages/nlp
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
