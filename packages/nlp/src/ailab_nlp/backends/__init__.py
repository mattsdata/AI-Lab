# ==================================================================================================
# 📜 ailab-nlp :: backends
# --------------------------------------------------------------------------------------------------
#  File         : packages/nlp/src/ailab_nlp/backends/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Home of the concrete sentiment backends. Each module adapts one approach to the
#    SentimentAnalyzer interface and registers itself by name.
#
#  🧠 DESIGN NOTES
#    - Intentionally imports nothing: backends load on demand through create_analyzer(), so the
#      lexicon baseline never pays for transformers or PyTorch.
#    - Adding a backend: create a module here, decorate the class with
#      @register_analyzer("name") and add the module path to `builtins` of ANALYZERS.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release with the lexicon and FinBERT backends.
# ==================================================================================================
"""Sentiment backends, loaded lazily by :func:`ailab_nlp.create_analyzer`."""
