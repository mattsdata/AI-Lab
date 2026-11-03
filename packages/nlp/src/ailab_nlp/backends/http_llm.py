# ==================================================================================================
# 📜 ailab-nlp :: backends :: http_llm
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/backends/http_llm.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Two HTTP backends for the LanguageModel interface: a local Ollama server, and any service
#    exposing the OpenAI-compatible /chat/completions endpoint.
#
#  🔧 WHAT IT DOES
#    - OllamaLLM           : POST {host}/api/generate (non-streaming). Registered as "ollama".
#    - OpenAICompatibleLLM : POST {base_url}/chat/completions. Registered as
#                            "openai-compatible". The API key is read from an environment
#                            variable, never passed around in code or logs.
#
#  🧠 DESIGN NOTES
#    - Temperature 0 by default: analysis should be repeatable, not creative.
#    - Standard-library HTTP only; the transport is injectable so tests run offline.
#    - Network errors are retried (mattslib.retry); HTTP errors from the server are not, because
#      a wrong model name or key will not fix itself on a second try.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: Ollama and OpenAI-compatible backends.
# ==================================================================================================
"""HTTP language-model backends: Ollama and OpenAI-compatible APIs."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from mattslib import retry

from ailab_nlp.llm import register_llm

Post = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body: dict[str, Any] = json.loads(response.read())
            return body
    except urllib.error.HTTPError as exc:  # server answered: do not retry, explain
        raise RuntimeError(f"LLM server returned HTTP {exc.code}: {exc.read()[:200]!r}") from None


# ========================================
# 📂 Ollama
# ========================================
@register_llm("ollama")
class OllamaLLM:
    """A local model served by Ollama, e.g. OllamaLLM(model="llama3.1")."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        post: Post = _post_json,
    ) -> None:
        self.model = model
        self._url = host.rstrip("/") + "/api/generate"
        self._temperature = temperature
        self._post = retry(attempts=3, exceptions=(OSError,))(post)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if system:
            payload["system"] = system
        return str(self._post(self._url, payload, {})["response"]).strip()


# ========================================
# 📂 OpenAI-Compatible
# ========================================
@register_llm("openai-compatible")
class OpenAICompatibleLLM:
    """Any /chat/completions service. The key comes from ``api_key_env`` (default LLM_API_KEY)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str = "LLM_API_KEY",
        temperature: float = 0.0,
        post: Post = _post_json,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = os.environ.get(api_key_env, "")
        self._temperature = temperature
        self._post = retry(attempts=3, exceptions=(OSError,))(post)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        reply = self._post(
            self._url,
            {"model": self.model, "messages": messages, "temperature": self._temperature},
            headers,
        )
        return str(reply["choices"][0]["message"]["content"]).strip()
