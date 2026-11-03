# ==================================================================================================
# 📜 ailab-nlp :: retrieval
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/retrieval.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    The "R" in RAG: find the passages most likely to contain the answer. A language model can
#    only be as accurate as the evidence it is shown, so retrieval quality decides answer quality.
#
#  🔧 WHAT IT DOES
#    - Bm25Retriever      : classic keyword ranking (Okapi BM25); exact terms such as "covenant"
#                           or "EBITDA" matter in financial and legal text.
#    - EmbeddingRetriever : semantic search with any embedding function (cosine similarity).
#    - HybridRetriever    : combines retrievers with Reciprocal Rank Fusion (RRF).
#    - Retriever, Hit     : the shared interface and result type.
#
#  🧠 DESIGN NOTES
#    - BM25 first: no model, no GPU, explainable scores, and a strong baseline that embeddings
#      must beat on your own evaluation set before they are worth their cost.
#    - The embedding function is injected (e.g. a sentence-transformers model's `encode`), so
#      this module never depends on a specific model library.
#    - Tokens are lower-cased, stop words dropped and plurals folded, the same for queries and
#      text, so "risk factors" matches "risks".
#    - RRF fuses by rank, not raw score, so BM25 scores and cosine similarities need no
#      calibration against each other; k=60 is the value from the original RRF paper.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: BM25, embedding and hybrid (RRF) retrievers.
# ==================================================================================================
"""Retrievers for RAG: BM25, embeddings and hybrid rank fusion."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ailab_nlp.documents import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what which who will with how does do did our we".split()
)


def _stem(token: str) -> str:
    """Light plural folding ("risks" -> "risk"), enough to match questions to text."""
    return (
        token[:-1] if len(token) > 4 and token.endswith("s") and not token.endswith("ss") else token
    )


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


# ========================================
# 📂 Interface
# ========================================
@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


# ========================================
# 📂 BM25
# ========================================
class Bm25Retriever:
    """Okapi BM25 over chunks. k1 controls term-frequency saturation, b length normalisation."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self._chunks = list(chunks)
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in self._chunks]
        self._length = [sum(tf.values()) for tf in self._tf]
        self._avg_length = max(sum(self._length) / len(self._length), 1e-9)
        df: Counter[str] = Counter(term for tf in self._tf for term in tf)
        n = len(self._chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = [t for t in tokenize(query) if t in self._idf]
        scores = []
        for tf, length in zip(self._tf, self._length, strict=True):
            norm = self._k1 * (1 - self._b + self._b * length / self._avg_length)
            score = sum(
                self._idf[t] * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [Hit(self._chunks[i], scores[i]) for i in ranked[:k] if scores[i] > 0]


# ========================================
# 📂 Embeddings
# ========================================
Embedder = Callable[[Sequence[str]], npt.ArrayLike]


class EmbeddingRetriever:
    """Cosine-similarity search using any embedding function (texts -> 2-D array)."""

    def __init__(self, chunks: Sequence[Chunk], embed: Embedder) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self._chunks = list(chunks)
        self._embed = embed
        self._matrix = self._normalise(embed([c.text for c in self._chunks]))

    @staticmethod
    def _normalise(vectors: npt.ArrayLike) -> npt.NDArray[np.float64]:
        m = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        out: npt.NDArray[np.float64] = m / np.where(norms > 0, norms, 1.0)
        return out

    def search(self, query: str, k: int = 5) -> list[Hit]:
        similarity = self._matrix @ self._normalise(self._embed([query]))[0]
        ranked = np.argsort(-similarity, kind="mergesort")[:k]
        return [Hit(self._chunks[int(i)], float(similarity[i])) for i in ranked]


# ========================================
# 📂 Hybrid (Reciprocal Rank Fusion)
# ========================================
class HybridRetriever:
    """Fuse several retrievers: score = sum over retrievers of 1 / (rrf_k + rank)."""

    def __init__(self, *retrievers: Retriever, rrf_k: int = 60, depth: int = 20) -> None:
        if len(retrievers) < 2:
            raise ValueError("a hybrid needs at least two retrievers")
        self._retrievers = retrievers
        self._rrf_k, self._depth = rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self._retrievers:
            for rank, hit in enumerate(retriever.search(query, self._depth), start=1):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1 / (self._rrf_k + rank)
                chunks[hit.chunk.id] = hit.chunk
        ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]
        return [Hit(chunks[cid], fused[cid]) for cid in ranked]
