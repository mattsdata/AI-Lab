# ailab-ml

*Author: **Mathew Varghese** · Version 1.0*

Explainable models and the validation metrics used to approve them, shared by every AI Lab
framework. NumPy only.

```python
from ailab_ml import LogisticRegression, calibration_table, gini, ks_statistic, roc_auc

model = LogisticRegression(l2=1.0).fit(x_train, y_train)
pd_test = model.predict_proba(x_test)
print(roc_auc(y_test, pd_test), gini(y_test, pd_test), ks_statistic(y_test, pd_test))
for band in calibration_table(y_test, pd_test, bins=10):
    print(f"predicted {band.predicted:.1%}  observed {band.observed:.1%}  n={band.count}")
```

## Components

| Module | What it does |
|---|---|
| `linear` | `LogisticRegression` (L2, optional standardisation), `logistic_irls` solver, safe `sigmoid` |
| `metrics` | `roc_auc`, `gini`, `ks_statistic`, `brier_score`, `calibration_table`, `population_stability_index` |

## Design decisions

- **Exact solver, no tuning.** Newton / IRLS reaches the penalised maximum-likelihood solution in a
  few iterations; there is no learning rate to tune. The intercept is never penalised.
- **Verified against definitions.** Tests compare AUC with brute-force pair counting (including
  ties) and KS with an explicit CDF comparison on random data.
- **Banking conventions.** Gini = 2 x AUC - 1; PSI with quantile bins from the reference sample
  (< 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift).
- **Extracted on second use.** Logistic regression began in the finance forecast model and moved
  here when the credit scorecard needed it (ADR-0003).

## Development

```bash
cd packages/ml
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
