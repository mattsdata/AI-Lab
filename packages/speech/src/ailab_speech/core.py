# ==================================================================================================
# 📜 ailab-speech :: core
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/src/ailab_speech/core.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    One interface for speech-to-text (STT) and one for text-to-speech (TTS), so applications
#    (voice bots, meeting notes, accessible reports) can swap engines without code changes:
#    a local Whisper model for privacy, a cloud voice for quality.
#
#  🔧 WHAT IT DOES
#    - Segment, Transcript       : timed text returned by every STT engine.
#    - SpeechToText, TextToSpeech: the two Protocols an engine implements.
#    - create_stt / create_tts   : build an engine by name from its registry.
#    - register_stt / register_tts: add your own engine.
#
#  🧠 DESIGN NOTES
#    - Same pattern as ailab-vision and ailab-nlp: mattslib.Registry with lazily imported
#      built-ins, so installing this package never pulls in PyTorch or a network client.
#    - Audio goes in as a file path or raw bytes; speech comes out as bytes plus a format, which
#      is what browsers, Colab's Audio widget and messaging APIs all accept.
#    - Segment timings are kept because subtitles, search and "jump to the moment" need them.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: interfaces, registries, transcript types.
# ==================================================================================================
"""Speech-to-text and text-to-speech interfaces with pluggable engines."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from mattslib import Registry

Audio = str | Path | bytes
F = TypeVar("F", bound=Callable[..., Any])


# ========================================
# 📂 Types
# ========================================
@dataclass(frozen=True, slots=True)
class Segment:
    start: float  # seconds
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None = None
    segments: list[Segment] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    data: bytes
    format: str  # "mp3", "wav", ...

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_bytes(self.data)
        return target


# ========================================
# 📂 Interfaces
# ========================================
@runtime_checkable
class SpeechToText(Protocol):
    def transcribe(self, audio: Audio, *, language: str | None = None) -> Transcript: ...


@runtime_checkable
class TextToSpeech(Protocol):
    def synthesize(self, text: str, *, language: str = "en") -> SpeechAudio: ...


# ========================================
# 📂 Registries
# ========================================
STT: Registry[SpeechToText] = Registry(
    "speech-to-text engine", builtins={"whisper": "ailab_speech.backends.whisper"}
)
TTS: Registry[TextToSpeech] = Registry(
    "text-to-speech engine", builtins={"gtts": "ailab_speech.backends.gtts"}
)


def register_stt(name: str) -> Callable[[F], F]:
    return STT.register(name)


def register_tts(name: str) -> Callable[[F], F]:
    return TTS.register(name)


def create_stt(name: str, /, **options: Any) -> SpeechToText:
    """Build the speech-to-text engine registered as ``name``."""
    return STT.create(name, **options)


def create_tts(name: str, /, **options: Any) -> TextToSpeech:
    """Build the text-to-speech engine registered as ``name``."""
    return TTS.create(name, **options)
