# ==================================================================================================
# 📜 ailab-nlp :: tests :: rag
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/tests/test_rag.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the RAG stack end to end: traceable chunks, correct retrieval, verified citations,
#    honest abstention, LLM backends that send the right requests, and an evaluation report
#    that catches both hallucinated and missed answers.
#
#  🧠 TESTING APPROACH
#    A short synthetic annual report with distinct sections, a toy bag-of-words embedder, and
#    fake LLMs/HTTP transports, so everything runs offline in milliseconds.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for documents, retrieval, llm backends, rag and evaluation.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import io
import json
import urllib.error
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ailab_nlp import (
    Bm25Retriever,
    EmbeddingRetriever,
    EvalCase,
    ExtractiveAnswerer,
    HybridRetriever,
    LlmAnswerer,
    RagPipeline,
    chunk_text,
    create_llm,
    evaluate_rag,
    html_to_text,
)
from ailab_nlp.backends import http_llm
from ailab_nlp.retrieval import tokenize

REPORT = (
    "Revenue. Total revenue grew 12 percent to 4.2 billion dollars, driven by cloud services.\n\n"
    "Debt. The company has 1.5 billion dollars of long-term debt maturing in 2031. "
    "A revolving credit facility of 500 million dollars was undrawn.\n\n"
    "Risk factors. Our operations depend on three key suppliers. A cyber attack could "
    "disrupt service. Interest rate increases would raise borrowing costs.\n\n"
    "Employees. We employed 12,400 people at year end, of whom 38 percent work outside "
    "the United States."
)


@pytest.fixture
def chunks() -> list[Any]:
    return chunk_text(REPORT, "acme-10k", max_chars=170, overlap=0)  # one section per chunk


def bag_of_words(texts: Sequence[str]) -> np.ndarray:
    vocabulary = sorted(set(tokenize(REPORT)))
    return np.array([[t in set(tokenize(x)) for t in vocabulary] for x in texts], dtype=float)


# ========================================
# 📂 Document Tests
# ========================================
def test_chunks_are_traceable_and_bounded() -> None:
    long_text = (REPORT + "\n\n") * 5 + "x" * 900  # includes one unbroken 900-char token
    for chunk in chunk_text(long_text, "doc", max_chars=300, overlap=60):
        assert long_text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text) <= 300
    first, second = chunk_text("word " * 100, "doc", max_chars=100, overlap=30)[:2]
    assert first.end > second.start  # consecutive chunks overlap
    with pytest.raises(ValueError):
        chunk_text("x", "doc", max_chars=10)
    with pytest.raises(ValueError):
        chunk_text("x", "doc", max_chars=100, overlap=60)


def test_html_to_text() -> None:
    markup = (
        "<html><head><style>p{}</style></head><body><p>Risk&nbsp;factors</p>"
        "<script>x=1</script><div>Debt   was <b>low</b>.</div></body></html>"
    )
    assert html_to_text(markup) == "Risk factors\n\nDebt was low."  # blocks become paragraphs


# ========================================
# 📂 Retrieval Tests
# ========================================
def test_bm25_finds_the_right_section(chunks: list[Any]) -> None:
    bm25 = Bm25Retriever(chunks)
    assert bm25.search("long-term debt maturity")[0].chunk.text.startswith("Debt.")
    assert bm25.search("how many employees")[0].chunk.text.startswith("Employees.")
    assert bm25.search("dividend policy xyzzy") == []
    with pytest.raises(ValueError):
        Bm25Retriever([])


def test_embedding_and_hybrid_retrieval(chunks: list[Any]) -> None:
    dense = EmbeddingRetriever(chunks, bag_of_words)
    assert dense.search("suppliers cyber attack", k=1)[0].chunk.text.startswith("Risk factors.")
    hybrid = HybridRetriever(Bm25Retriever(chunks), dense)
    top = hybrid.search("revenue growth cloud", k=2)
    assert top[0].chunk.text.startswith("Revenue.") and top[0].score > top[1].score
    with pytest.raises(ValueError):
        HybridRetriever(dense)
    with pytest.raises(ValueError):
        EmbeddingRetriever([], bag_of_words)


# ========================================
# 📂 RAG Pipeline Tests
# ========================================
class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append((prompt, system))
        return self.reply


def test_extractive_pipeline_cites_and_abstains(chunks: list[Any]) -> None:
    rag = RagPipeline(Bm25Retriever(chunks), ExtractiveAnswerer(), k=3)
    answer = rag.ask("How much long-term debt does the company have?")
    assert "1.5 billion" in answer.text and answer.grounded and not answer.abstained
    cited = answer.citations[0]
    assert REPORT[cited.start : cited.end].startswith("Debt.")
    nothing = rag.ask("What is the dividend policy xyzzy?")
    assert nothing.abstained and nothing.sources == [] and nothing.grounded
    assert ExtractiveAnswerer().answer("zebra", chunks) == "NOT FOUND"


def test_invented_citations_are_caught(chunks: list[Any]) -> None:
    llm = FakeLLM("Debt is 1.5 billion [1], see also [9] and [1].")
    answer = RagPipeline(Bm25Retriever(chunks), LlmAnswerer(llm), k=2).ask("long-term debt")
    assert [c.number for c in answer.citations] == [1]
    assert answer.invalid_citations == [9] and not answer.grounded
    prompt, system = llm.prompts[0]
    assert prompt.startswith("Sources:\n\n[1] ") and "Question: long-term debt" in prompt
    assert system is not None and "ONLY the numbered sources" in system


def test_model_abstention_and_validation(chunks: list[Any]) -> None:
    answer = RagPipeline(Bm25Retriever(chunks), LlmAnswerer(FakeLLM("NOT FOUND"))).ask("revenue")
    assert answer.abstained and answer.grounded
    with pytest.raises(ValueError):
        RagPipeline(Bm25Retriever(chunks), ExtractiveAnswerer(), k=0)


# ========================================
# 📂 Evaluation Tests
# ========================================
def test_evaluation_report(chunks: list[Any]) -> None:
    rag = RagPipeline(Bm25Retriever(chunks), ExtractiveAnswerer(), k=3)
    cases = [
        EvalCase("How much long-term debt is there?", "1.5 billion", expected_terms=("2031",)),
        EvalCase("How many people were employed?", "12,400", expected_terms=("12,400",)),
        EvalCase("What drove revenue growth?", "cloud services", expected_terms=("cloud",)),
        EvalCase("What is the dividend policy xyzzy?", answerable=False),
    ]
    report = evaluate_rag(rag, cases)
    assert report.hit_rate == 1.0 and report.mrr == 1.0
    assert report.abstention_accuracy == 1.0 and report.citation_precision == 1.0
    assert report.term_coverage == 1.0 and report.failures == []
    assert "4 questions" in report.summary()


def test_evaluation_catches_hallucination_and_misses(chunks: list[Any]) -> None:
    always_answers = RagPipeline(
        Bm25Retriever(chunks), LlmAnswerer(FakeLLM("Revenue grew [1].")), k=2
    )
    report = evaluate_rag(
        always_answers,
        [
            EvalCase("What is the revenue growth?", "12 percent", expected_terms=("12 percent",)),
            EvalCase("revenue dividend policy", answerable=False),  # retrieves, must still refuse
        ],
    )
    assert report.abstention_accuracy == 0.5 and report.term_coverage == 0.0
    assert any("answered an unanswerable" in f for f in report.failures)
    assert any("missing expected terms" in f for f in report.failures)
    refuses = RagPipeline(Bm25Retriever(chunks), LlmAnswerer(FakeLLM("NOT FOUND")), k=2)
    missed = evaluate_rag(refuses, [EvalCase("employees", "zzz-not-in-text")])
    assert missed.hit_rate == 0.0 and any("refused" in f for f in missed.failures)
    assert any("evidence not retrieved" in f for f in missed.failures)
    with pytest.raises(ValueError):
        evaluate_rag(refuses, [])


# ========================================
# 📂 LLM Backend Tests
# ========================================
def test_ollama_backend_request() -> None:
    sent: list[Any] = []

    def post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        sent.append((url, payload, headers))
        return {"response": "  hello  "}

    llm = create_llm("ollama", model="llama3.1", host="http://box:11434/", post=post)
    assert llm.complete("hi", system="be brief") == "hello"
    url, payload, headers = sent[0]
    assert url == "http://box:11434/api/generate" and headers == {}
    assert payload == {
        "model": "llama3.1",
        "prompt": "hi",
        "stream": False,
        "options": {"temperature": 0.0},
        "system": "be brief",
    }


def test_openai_compatible_backend_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    def post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        sent.append((url, payload, headers))
        return {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    llm = create_llm(
        "openai-compatible",
        model="m",
        base_url="https://api.x/v1/",
        api_key_env="TEST_LLM_KEY",
        post=post,
    )
    assert llm.complete("q", system="s") == "answer"
    url, payload, headers = sent[0]
    assert url == "https://api.x/v1/chat/completions"
    assert headers == {"Authorization": "Bearer secret"}
    assert payload["messages"][0] == {"role": "system", "content": "s"}
    monkeypatch.delenv("TEST_LLM_KEY")
    keyless = create_llm(
        "openai-compatible",
        model="m",
        base_url="http://local",
        api_key_env="TEST_LLM_KEY",
        post=post,
    )
    keyless.complete("q")
    assert sent[1][2] == {} and len(sent[1][1]["messages"]) == 1


def test_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"response": "ok"}).encode()

    def ok(request: Any, timeout: float) -> Response:
        assert request.get_method() == "POST" and json.loads(request.data) == {"a": 1}
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", ok)
    assert http_llm._post_json("http://x", {"a": 1}, {}) == {"response": "ok"}

    def server_error(request: Any, timeout: float) -> Response:
        raise urllib.error.HTTPError("http://x", 404, "nf", {}, io.BytesIO(b"model not found"))  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", server_error)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        http_llm._post_json("http://x", {}, {})
