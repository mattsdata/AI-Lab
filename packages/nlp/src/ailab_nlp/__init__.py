# ==================================================================================================
# 📜 ailab-nlp
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/__init__.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.2
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab NLP package: reusable language components for any domain
#    framework: sentiment, news data (GDELT), LLM backends, RAG with citations, evaluation.
#
#  🔧 WHAT IT DOES
#    Re-exports the public API: sentiment types, analyzer interface and registry, aggregation.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: sentiment interface, lexicon and FinBERT backends, aggregation.
#
#    V1.1 (2026-11-03) - Mathew Varghese
#      - Added news module: GDELT client (daily tone history, recent articles), rolling_tone.
#
#    V1.2 (2026-11-03) - Mathew Varghese
#      - Added RAG: chunking, BM25/embedding/hybrid retrieval, LLM backends, cited answers,
#        evaluation.
# ==================================================================================================
"""ailab-nlp: model-independent language components (sentiment first)."""

# ========================================
# 📂 Imports
# ========================================
from ailab_nlp.aggregate import SentimentSummary, aggregate_sentiment, time_decay_weights
from ailab_nlp.documents import Chunk, chunk_text, html_to_text
from ailab_nlp.evaluation import EvalCase, RagReport, evaluate_rag
from ailab_nlp.llm import LanguageModel, create_llm, register_llm
from ailab_nlp.news import DailyTone, GdeltClient, NewsArticle, ToneSummary, rolling_tone
from ailab_nlp.rag import (
    Answer,
    Citation,
    ExtractiveAnswerer,
    LlmAnswerer,
    RagPipeline,
)
from ailab_nlp.retrieval import Bm25Retriever, EmbeddingRetriever, Hit, HybridRetriever, Retriever
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
__version__ = "1.2.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "Answer",
    "Bm25Retriever",
    "Chunk",
    "Citation",
    "DailyTone",
    "EmbeddingRetriever",
    "EvalCase",
    "ExtractiveAnswerer",
    "GdeltClient",
    "Hit",
    "HybridRetriever",
    "LanguageModel",
    "LlmAnswerer",
    "NewsArticle",
    "RagPipeline",
    "RagReport",
    "Retriever",
    "SentimentAnalyzer",
    "SentimentResult",
    "SentimentSummary",
    "ToneSummary",
    "aggregate_sentiment",
    "available_analyzers",
    "chunk_text",
    "create_analyzer",
    "create_llm",
    "evaluate_rag",
    "html_to_text",
    "register_analyzer",
    "register_llm",
    "rolling_tone",
    "time_decay_weights",
]
