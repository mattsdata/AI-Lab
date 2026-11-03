# ==================================================================================================
# 📜 ailab-nlp :: examples :: news_sentiment
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/examples/news_sentiment.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Score a file of dated headlines and produce a time-weighted sentiment summary as of a
#    decision date, the input the finance outlook model will use.
#
#  🔧 WHAT IT DOES
#    read "YYYY-MM-DD<TAB>headline" lines -> drop anything after the decision date
#    -> score with the chosen backend -> weight by age -> print per-headline and overall results
#
#  USAGE
#    python packages/nlp/examples/news_sentiment.py headlines.tsv --as-of 2026-11-02
#    python packages/nlp/examples/news_sentiment.py headlines.tsv --backend finbert
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: dated headlines to a decayed sentiment summary.
# ==================================================================================================
"""Time-weighted sentiment of dated headlines as of a decision date."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ailab_nlp import aggregate_sentiment, create_analyzer, time_decay_weights


# ========================================
# 📂 Pipeline
# ========================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="TSV: YYYY-MM-DD<TAB>headline per line")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="decision date")
    parser.add_argument("--backend", default="lexicon", help="lexicon or finbert")
    parser.add_argument("--half-life", type=float, default=7.0, help="days")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)
    rows = []
    for line in args.file.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        day, text = line.split("\t", 1)
        published = date.fromisoformat(day.strip())
        if published <= as_of:  # never use news from after the decision date
            rows.append((published, text.strip()))

    analyzer = create_analyzer(args.backend)
    results = analyzer.analyze([text for _, text in rows])
    weights = time_decay_weights([(as_of - d).days for d, _ in rows], args.half_life)

    for (published, text), result, weight in zip(rows, results, weights, strict=True):
        print(f"{published}  {result.label:<8} {result.score:+.2f}  w={weight:.2f}  {text[:70]}")
    summary = aggregate_sentiment(results, weights)
    print(
        f"\nAs of {as_of}: {summary.label.upper()} {summary.score:+.2f} "
        f"from {summary.count} headlines ({args.backend}, half-life {args.half_life:g} days)"
    )


if __name__ == "__main__":
    main()
