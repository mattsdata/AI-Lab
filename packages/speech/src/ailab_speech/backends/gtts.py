# ==================================================================================================
# 📜 ailab-speech :: backends.gtts
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/src/ailab_speech/backends/gtts.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Simple, free text-to-speech through gTTS (Google Translate's voice). Good for prototypes
#    and demos in many languages, including Malayalam ("ml") and Hindi ("hi").
#
#  🧠 DESIGN NOTES
#    - Needs internet and sends the text to Google: do not use it for private text; register a
#      local engine (for example Piper) for that.
#    - Long text is split into sentence chunks first (split_for_speech); the MP3 parts are
#      concatenated, which MP3 players handle as one stream.
#    - The gTTS class is injectable so the adapter is testable offline.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
"""gTTS text-to-speech backend (online)."""

from __future__ import annotations

import io
from typing import Any

from ailab_speech.core import SpeechAudio, register_tts
from ailab_speech.text import split_for_speech


@register_tts("gtts")
class GttsTTS:
    def __init__(self, *, slow: bool = False, max_chars: int = 200, engine: Any | None = None):
        if engine is None:
            try:
                from gtts import gTTS
            except ImportError as exc:
                raise ImportError(
                    "the 'gtts' engine needs gTTS: pip install 'ailab-speech[gtts]'"
                ) from exc
            engine = gTTS
        self._engine = engine
        self._slow = slow
        self._max_chars = max_chars

    def synthesize(self, text: str, *, language: str = "en") -> SpeechAudio:
        if not text.strip():
            raise ValueError("nothing to say: text is empty")
        out = io.BytesIO()
        for chunk in split_for_speech(text, max_chars=self._max_chars):
            self._engine(chunk, lang=language, slow=self._slow).write_to_fp(out)
        return SpeechAudio(out.getvalue(), "mp3")
