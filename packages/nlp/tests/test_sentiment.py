# ==================================================================================================
# 📜 ailab-nlp :: tests :: sentiment
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/tests/test_sentiment.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the result type, the lexicon baseline (including negation), aggregation with time
#    decay, the registry, and the FinBERT adapter's conversion logic.
#
#  🧠 TESTING APPROACH
#    Real headlines-style sentences for behaviour; a fake `transformers` module for FinBERT, so
#    the suite runs in milliseconds without downloading a 400 MB model.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for ailab-nlp sentiment.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import sys
import types
from typing import Any

import pytest

from ailab_nlp import (
    SentimentAnalyzer,
    SentimentResult,
    aggregate_sentiment,
    available_analyzers,
    create_analyzer,
    time_decay_weights,
)
from ailab_nlp.sentiment import ANALYZERS


# ========================================
# 📂 Result Type Tests
# ========================================
def test_from_probabilities_normalises_and_resolves_ties() -> None:
    r = SentimentResult.from_probabilities(positive=2, negative=1, neutral=1)
    assert r.label == "positive" and r.score == pytest.approx(0.25)
    tie = SentimentResult.from_probabilities(positive=0.4, negative=0.4, neutral=0.2)
    assert tie.label == "neutral" and tie.score == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "angry", "score": 0.0, "confidence": 0.5},
        {"label": "positive", "score": 1.5, "confidence": 0.5},
        {"label": "positive", "score": 0.5, "confidence": 2.0},
    ],
)
def test_result_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SentimentResult(**kwargs)


def test_probabilities_must_be_valid() -> None:
    with pytest.raises(ValueError):
        SentimentResult.from_probabilities(positive=0, negative=0, neutral=0)


# ========================================
# 📂 Lexicon Backend Tests
# ========================================
@pytest.fixture
def lexicon() -> SentimentAnalyzer:
    return create_analyzer("lexicon")


def test_lexicon_reads_headlines(lexicon: SentimentAnalyzer) -> None:
    good, bad, flat = lexicon.analyze(
        [
            "Profits surged to a record as growth beat forecasts",
            "Shares plunged after a profit warning and fraud investigation",
            "The company will hold its annual meeting on Tuesday",
        ]
    )
    assert good.label == "positive" and good.score > 0.4
    # "profit warning" is bad news, but "profit" counts as a positive word: a known lexicon
    # limit (context), which is why the score is only moderately negative. FinBERT fixes this.
    assert bad.label == "negative" and bad.score < -0.3
    assert flat.label == "neutral" and flat.score == 0.0


def test_lexicon_handles_negation(lexicon: SentimentAnalyzer) -> None:
    (negated,) = lexicon.analyze(["Results were not strong and growth didn't improve"])
    assert negated.label == "negative"


def test_lexicon_more_evidence_means_more_confidence(lexicon: SentimentAnalyzer) -> None:
    one, many = lexicon.analyze(["growth", "growth profit record surge"])
    assert many.score > one.score > 0


def test_lexicon_custom_lists_and_conflicts() -> None:
    custom = create_analyzer("lexicon", positive=["calm"], negative=["anxious"])
    assert custom.analyze(["calm day"])[0].label == "positive"
    with pytest.raises(ValueError, match="both positive and negative"):
        create_analyzer("lexicon", positive=["x"], negative=["x"])


# ========================================
# 📂 Aggregation Tests
# ========================================
def test_time_decay_weights() -> None:
    assert time_decay_weights([0, 7, 14], half_life_days=7) == pytest.approx([1, 0.5, 0.25])
    with pytest.raises(ValueError, match="look-ahead"):
        time_decay_weights([-1])
    with pytest.raises(ValueError):
        time_decay_weights([1], half_life_days=0)


def test_aggregate_weights_recent_news_more() -> None:
    old_bad = SentimentResult("negative", -0.8, 0.9)
    new_good = SentimentResult("positive", 0.6, 0.8)
    equal = aggregate_sentiment([old_bad, new_good])
    recent = aggregate_sentiment([old_bad, new_good], time_decay_weights([21, 0]))
    assert equal.label == "negative"  # -0.1 on a plain average
    assert recent.label == "positive"  # the fresh story dominates
    assert recent.count == 2 and recent.positive_share == 0.5


def test_aggregate_edge_cases() -> None:
    empty = aggregate_sentiment([])
    assert (empty.label, empty.count, empty.score) == ("neutral", 0, 0.0)
    tiny = aggregate_sentiment([SentimentResult("positive", 0.02, 0.4)])
    assert tiny.label == "neutral"  # inside the neutral band
    with pytest.raises(ValueError):
        aggregate_sentiment([tiny.__class__ and SentimentResult("neutral", 0, 1)], [1, 2])
    with pytest.raises(ValueError):
        aggregate_sentiment([SentimentResult("neutral", 0, 1)], [0])


# ========================================
# 📂 Registry and FinBERT Tests (fake transformers)
# ========================================
def test_registry_lists_builtins_and_rejects_unknown() -> None:
    assert {"lexicon", "finbert"} <= set(available_analyzers())
    with pytest.raises(ValueError, match="available: finbert, lexicon"):
        create_analyzer("vader")


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def pipeline(task: str, **kwargs: Any) -> Any:
        calls.append({"task": task, **kwargs})

        def run(texts: list[str], **call_kwargs: Any) -> list[list[dict[str, Any]]]:
            calls.append(call_kwargs)
            return [
                [
                    {"label": "Positive", "score": 0.1},
                    {"label": "Negative", "score": 0.8},
                    {"label": "Neutral", "score": 0.1},
                ]
                for _ in texts
            ]

        return run

    module = types.ModuleType("transformers")
    module.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.delitem(sys.modules, "ailab_nlp.backends.finbert", raising=False)
    ANALYZERS.unregister("finbert")
    return calls


def test_finbert_backend_converts_probabilities(fake_transformers: list[dict[str, Any]]) -> None:
    finbert = create_analyzer("finbert", batch_size=4, device="cpu")
    (result,) = finbert.analyze(["Revenue fell sharply"])
    assert result.label == "negative" and result.score == pytest.approx(-0.7)
    assert fake_transformers[0]["model"] == "ProsusAI/finbert"
    assert fake_transformers[0]["top_k"] is None  # all three probabilities requested
    assert fake_transformers[1] == {"batch_size": 4, "truncation": True}
    assert finbert.analyze([]) == []


def test_finbert_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.delitem(sys.modules, "ailab_nlp.backends.finbert", raising=False)
    ANALYZERS.unregister("finbert")
    with pytest.raises(ImportError, match=r"ailab-nlp\[finbert\]"):
        create_analyzer("finbert")
