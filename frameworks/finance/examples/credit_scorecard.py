# ==================================================================================================
# 📜 ailab-finance :: examples :: credit_scorecard
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/examples/credit_scorecard.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Build, validate, explain and fairness-check a credit scorecard end to end, either on a
#    synthetic loan book (default, runs offline) or on your own CSV.
#
#  🔧 WHAT IT DOES
#    split build / hold-out -> fit scorecard -> points table -> validation pack (AUC, Gini, KS,
#    calibration, PSI) -> explanation for one applicant -> outcomes by group at a cut-off
#
#  🧠 DESIGN NOTES
#    - The group column (e.g. an age band) is used ONLY for the fairness check, never as a
#      model input.
#    - The cut-off approves the best-scoring 80% by default; banks set it from risk appetite.
#
#  USAGE
#    python frameworks/finance/examples/credit_scorecard.py
#    python frameworks/finance/examples/credit_scorecard.py --csv loans.csv --target default \
#        --features income debt_ratio delinquencies --group age_band
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: scorecard build, validation, explanation and fairness check.
# ==================================================================================================
"""Credit scorecard demo: build, validate, explain, check fairness."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from ailab_ml import sigmoid

from ailab_finance import Scorecard, group_outcomes, validate


# ========================================
# 📂 Data
# ========================================
def synthetic_book(n: int = 12000, seed: int = 42) -> tuple[dict[str, Any], Any, Any]:
    """A loan book with known drivers, a noise column, 5% missing incomes and two groups."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.3, 0.5, n)
    debt_ratio = rng.uniform(0, 0.9, n)
    delinquencies = rng.poisson(0.4, n).astype(float)
    history = rng.uniform(0, 25, n)
    z = (
        -3.2
        - 0.8 * np.log(income / 30000)
        + 2.5 * debt_ratio
        + 0.7 * delinquencies
        - 0.05 * history
    )
    defaults = (rng.random(n) < sigmoid(z)).astype(float)
    income[rng.random(n) < 0.05] = np.nan
    features = {
        "income": income,
        "debt_ratio": debt_ratio,
        "delinquencies": delinquencies,
        "history_years": history,
        "noise": rng.normal(size=n),
    }
    groups = np.where(rng.random(n) < 0.5, "group A", "group B")
    return features, defaults, groups


def read_csv(
    path: Path, target: str, columns: list[str], group: str | None
) -> tuple[Any, Any, Any]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def number(text: str) -> float:
        return float(text) if text.strip() not in ("", "NA", "null") else float("nan")

    features = {c: np.array([number(r[c]) for r in rows]) for c in columns}
    defaults = np.array([number(r[target]) for r in rows])
    groups = np.array([r[group] for r in rows]) if group else np.array(["all"] * len(rows))
    return features, defaults, groups


# ========================================
# 📂 Pipeline
# ========================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--target", default="default", help="column with 1 = default, 0 = repaid")
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--group", help="column used only for the fairness check")
    parser.add_argument("--approve", type=float, default=0.8, help="share approved at cut-off")
    args = parser.parse_args()

    if args.csv:
        features, defaults, groups = read_csv(args.csv, args.target, args.features, args.group)
    else:
        features, defaults, groups = synthetic_book()

    n = len(defaults)
    order = np.random.default_rng(0).permutation(n)
    build, holdout = order[: int(n * 0.7)], order[int(n * 0.7) :]

    def take(rows: Any) -> dict[str, Any]:
        return {name: values[rows] for name, values in features.items()}

    scorecard = Scorecard().fit(take(build), defaults[build])
    print("=== Scorecard ===")
    print(scorecard.table())

    print("\n=== Validation on hold-out ===")
    reference = scorecard.score(take(build))
    print(
        validate(scorecard, take(holdout), defaults[holdout], reference_scores=reference).summary()
    )

    print("\n=== Explanation for the riskiest hold-out applicant ===")
    scores = scorecard.score(take(holdout))
    worst = holdout[int(np.argmin(scores))]
    applicant = {name: float(values[worst]) for name, values in features.items()}
    explanation = scorecard.explain(applicant)
    print(f"Score {explanation.score:.0f} · PD {explanation.probability_of_default:.1%}")
    print("Main reasons:", ", ".join(explanation.reasons))

    print("\n=== Fairness check (group used for monitoring only) ===")
    cutoff = float(np.quantile(scores, 1 - args.approve))
    outcomes, ratio = group_outcomes(scores, defaults[holdout], groups[holdout], cutoff=cutoff)
    for name, o in outcomes.items():
        print(
            f"{name:<10} n={o.count:<5} approved {o.approval_rate:5.1%}  "
            f"default rate if approved {o.default_rate_approved:5.1%}"
        )
    verdict = "passes" if ratio >= 0.8 else "fails"
    print(f"Adverse impact ratio {ratio:.2f} ({verdict} the four-fifths screening heuristic)")


if __name__ == "__main__":
    main()
