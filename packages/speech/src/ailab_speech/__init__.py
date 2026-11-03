# ==================================================================================================
# 📜 ailab-speech
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/src/ailab_speech/__init__.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab speech package: pluggable speech-to-text and text-to-speech
#    engines, plus engine-independent tools (word error rate, subtitles, TTS chunking).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
"""ailab-speech: pluggable STT / TTS engines, WER, subtitles."""

# ========================================
# 📂 Imports
# ========================================
from ailab_speech.core import (
    STT,
    TTS,
    Segment,
    SpeechAudio,
    SpeechToText,
    TextToSpeech,
    Transcript,
    create_stt,
    create_tts,
    register_stt,
    register_tts,
)
from ailab_speech.text import split_for_speech, to_srt, word_error_rate

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

__all__ = [
    "STT",
    "TTS",
    "Segment",
    "SpeechAudio",
    "SpeechToText",
    "TextToSpeech",
    "Transcript",
    "create_stt",
    "create_tts",
    "register_stt",
    "register_tts",
    "split_for_speech",
    "to_srt",
    "word_error_rate",
]
