# ailab-mental-health

*Author: **Mathew Varghese** · Version 1.0 · Not a medical device. Not a diagnosis.*

Private, safety-first wellbeing tools: score validated screening questionnaires exactly as
published, keep an encrypted daily journal on your own computer, and read gentle summaries
that describe patterns and point to support, without ever labelling anyone.

> If you are struggling right now, please contact **Samaritans on 116 123** (UK, free, 24/7),
> **NHS 111**, or **999** in an emergency.

```python
from ailab_mental_health import score_phq9, wellbeing_summary

result = score_phq9([1, 0, 1, 2, 0, 1, 0, 0, 0])
print(result.total, result.band, result.safety_alert)  # 5 mild False
print(result.note)  # "This is a screening score, not a diagnosis..."
```

## Components

| Module | What it does |
|---|---|
| `instruments` | PHQ-9 and GAD-7 scoring with published cut-offs; PHQ-9 item 9 always raises a safety alert |
| `safety` | Regional crisis contacts (UK, IN, US), safety message, conservative crisis-language check |
| `journal` | `CheckIn` (mood, sleep, private note), `JournalStore` encrypted with Fernet, atomic writes |
| `summary` | Last 7 days vs the 4 weeks before, sleep, sustained-low-mood nudge, safety first |

## Design decisions ([ADR-0006](../../docs/adr/0006-wellbeing-tools-safety-first-non-diagnostic.md))

- **Non-diagnostic.** Describes the person's own entries; never names a condition. A test scans
  summaries for diagnostic terms.
- **Safety overrides everything.** Item 9 > 0 or crisis language puts crisis contacts first.
  The check errs towards showing help.
- **Private by design.** Local only, encrypted at rest, no plain-text mode, notes hidden from
  reprs and logs, nothing sent over the network.
- **Validated instruments only.** Scoring follows the published PHQ-9 / GAD-7 cut-offs; question
  wording is not reproduced (use the official forms).
- **Clinical review before real use.** Thresholds are parameters; deployment to patients would
  need clinical governance and regulatory review (e.g. MHRA in the UK).

## Command-line journal

```bash
pip install -e "frameworks/mental-health[crypto]"
python frameworks/mental-health/examples/wellbeing.py key          # once; keep the key safe
export AILAB_JOURNAL_KEY="<the key>"
python frameworks/mental-health/examples/wellbeing.py checkin --mood 6 --sleep 7.5 --note "ok day"
python frameworks/mental-health/examples/wellbeing.py summary
python frameworks/mental-health/examples/wellbeing.py phq9 1 0 1 2 0 1 0 0 0
```

## Roadmap

- Optional journal-note sentiment trend using `ailab-nlp` (still local, still non-diagnostic)
- Further validated instruments after a licensing check (e.g. adult ADHD self-report screeners)
- Research tools on public, consented neuroimaging datasets (e.g. via MNE / nilearn)

## Development

```bash
cd frameworks/mental-health
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
