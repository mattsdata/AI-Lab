# ==================================================================================================
# 📜 ailab-nlp :: llm
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/llm.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    One interface for large language models, whether a free local model (Ollama) or any hosted
#    or self-hosted service that speaks the OpenAI-compatible chat API (vLLM, LM Studio, many
#    cloud providers). Applications choose a model by configuration, not by rewriting code.
#
#  🔧 WHAT IT DOES
#    - LanguageModel : the interface: complete(prompt, system=None) -> text.
#    - create_llm    : builds a backend by name: "ollama" or "openai-compatible".
#    - register_llm  : decorator to add a backend (same mattslib.Registry as other packages).
#
#  🧠 DESIGN NOTES
#    - The interface is deliberately tiny: a prompt in, text out. Retrieval, citation checking
#      and evaluation live outside the model, so any model can be swapped and re-evaluated.
#    - Backends are imported lazily and use only the standard library for HTTP.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: LanguageModel interface and registry.
# ==================================================================================================
"""Model-independent large language model interface."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from mattslib import Registry


# ========================================
# 📂 Interface and Registry
# ========================================
@runtime_checkable
class LanguageModel(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


F = TypeVar("F", bound=Callable[..., LanguageModel])

LLMS: Registry[LanguageModel] = Registry(
    "llm",
    builtins={
        "ollama": "ailab_nlp.backends.http_llm",
        "openai-compatible": "ailab_nlp.backends.http_llm",
    },
)


def register_llm(name: str) -> Callable[[F], F]:
    return LLMS.register(name)


def create_llm(name: str, /, **options: Any) -> LanguageModel:
    """e.g. create_llm("ollama", model="llama3.1") or create_llm("openai-compatible", ...)."""
    return LLMS.create(name, **options)
