# ==================================================================================================
# 📜 ailab-nlp
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab NLP package: reusable language components for any domain
#    framework. V1.0 provides sentiment analysis; LLM backends, RAG and evaluation follow.
#
#  🔧 WHAT IT DOES
#    Re-exports the public API: sentiment types, analyzer interface and registry, aggregation.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: sentiment interface, lexicon and FinBERT backends, aggregation.
# ==================================================================================================
"""ailab-nlp: model-independent language components (sentiment first)."""

# ========================================
# 📂 Imports
# ========================================
from ailab_nlp.aggregate import SentimentSummary, aggregate_sentiment, time_decay_weights
from ailab_nlp.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    available_analyzers,
    create_analyzer,
    register_analyzer,
)

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "SentimentAnalyzer",
    "SentimentResult",
    "SentimentSummary",
    "aggregate_sentiment",
    "available_analyzers",
    "create_analyzer",
    "register_analyzer",
    "time_decay_weights",
]
