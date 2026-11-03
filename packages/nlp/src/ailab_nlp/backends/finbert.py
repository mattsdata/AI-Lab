# ==================================================================================================
# 📜 ailab-nlp :: backends :: finbert
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/backends/finbert.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE). Model weights have their own licence (see package README).
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Financial-language sentiment with FinBERT, a BERT model fine-tuned on financial text.
#    It understands that "costs fell" is good news and "revenue fell" is bad, which a word
#    list cannot. Runs locally, free, on CPU or GPU.
#
#  🔧 WHAT IT DOES
#    - FinBertAnalyzer : wraps a Hugging Face text-classification pipeline and converts its
#                        positive / negative / neutral probabilities to SentimentResult.
#                        Registered as the "finbert" backend.
#
#  🧠 DESIGN NOTES
#    - transformers (and PyTorch) are imported in __init__, only when this backend is created.
#    - All three class probabilities are requested (top_k=None), so the score uses the full
#      distribution rather than only the winning label.
#    - Long texts are truncated to the model's limit (512 tokens). For long documents, score
#      paragraphs separately and combine them with aggregate_sentiment.
#    - The model name is a parameter: any compatible 3-class model can be dropped in.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: FinBERT backend via Hugging Face pipelines.
# ==================================================================================================
"""FinBERT sentiment backend (Hugging Face transformers)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ailab_nlp.sentiment import SentimentResult, register_analyzer


# ========================================
# 📂 FinBERT Backend
# ========================================
@register_analyzer("finbert")
class FinBertAnalyzer:
    """FinBERT (or any 3-class positive/negative/neutral model) via transformers."""

    def __init__(
        self,
        model: str = "ProsusAI/finbert",
        *,
        batch_size: int = 16,
        device: int | str | None = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "the 'finbert' backend needs transformers: pip install 'ailab-nlp[finbert]'"
            ) from exc
        self._pipe: Any = pipeline("text-classification", model=model, top_k=None, device=device)
        self._batch_size = batch_size

    def analyze(self, texts: Sequence[str]) -> list[SentimentResult]:
        if not texts:
            return []
        outputs = self._pipe(list(texts), batch_size=self._batch_size, truncation=True)
        results = []
        for class_scores in outputs:  # e.g. [{"label": "positive", "score": 0.91}, ...]
            p = {item["label"].lower(): float(item["score"]) for item in class_scores}
            results.append(
                SentimentResult.from_probabilities(
                    positive=p.get("positive", 0.0),
                    negative=p.get("negative", 0.0),
                    neutral=p.get("neutral", 0.0),
                )
            )
        return results
