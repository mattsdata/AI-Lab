# ==================================================================================================
# 📜 ailab-speech :: tests
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/tests/test_speech.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Offline tests: engines run against fakes (no model downloads, no network); text tools are
#    checked against hand-computed values.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ailab_speech import (
    STT,
    Segment,
    SpeechAudio,
    SpeechToText,
    TextToSpeech,
    Transcript,
    create_stt,
    create_tts,
    register_stt,
    split_for_speech,
    to_srt,
    word_error_rate,
)


# ========================================
# 📂 Fakes
# ========================================
class FakeWhisperModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def transcribe(self, path: str, *, language: str | None, beam_size: int) -> Any:
        self.calls.append((Path(path).suffix, language))
        segs = [
            types.SimpleNamespace(start=0.0, end=1.5, text=" Hello there."),
            types.SimpleNamespace(start=1.5, end=3.25, text=" Good morning."),
        ]
        return iter(segs), types.SimpleNamespace(language=language or "en")


class FakeGtts:
    def __init__(self, text: str, *, lang: str, slow: bool) -> None:
        self.text = text

    def write_to_fp(self, fp: Any) -> None:
        fp.write(f"[{self.text}]".encode())


# ========================================
# 📂 Engines
# ========================================
def test_whisper_from_path_and_bytes() -> None:
    model = FakeWhisperModel()
    stt = create_stt("whisper", engine=model)
    assert isinstance(stt, SpeechToText)
    result = stt.transcribe("talk.wav")
    assert result.text == "Hello there. Good morning."
    assert result.language == "en"
    assert result.segments[1] == Segment(1.5, 3.25, "Good morning.")
    assert stt.transcribe(b"RIFF....", language="ml").language == "ml"
    assert model.calls == [(".wav", None), (".audio", "ml")]


def test_whisper_loads_library_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    )
    assert create_stt("whisper", model="tiny").transcribe("a.mp3").text


def test_missing_libraries_explain_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "gtts", None)
    with pytest.raises(ImportError, match=r"ailab-speech\[whisper\]"):
        create_stt("whisper")
    with pytest.raises(ImportError, match=r"ailab-speech\[gtts\]"):
        create_tts("gtts")


def test_gtts_chunks_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gtts", types.SimpleNamespace(gTTS=FakeGtts))
    tts = create_tts("gtts", max_chars=30)
    assert isinstance(tts, TextToSpeech)
    audio = tts.synthesize("First sentence here. Second sentence is here too.", language="ml")
    assert audio.format == "mp3"
    assert audio.data == b"[First sentence here.][Second sentence is here too.]"
    assert audio.save(tmp_path / "x.mp3").read_bytes() == audio.data
    with pytest.raises(ValueError):
        tts.synthesize("   ")


def test_registry_custom_engine_and_unknown() -> None:
    @register_stt("echo")
    class Echo:
        def transcribe(self, audio: Any, *, language: str | None = None) -> Transcript:
            return Transcript(str(audio))

    try:
        assert create_stt("echo").transcribe("hi").text == "hi"
        assert {"echo", "whisper"} <= set(STT.available())
    finally:
        STT.unregister("echo")
    with pytest.raises(ValueError, match="available"):
        create_stt("nope")


# ========================================
# 📂 Text tools
# ========================================
def test_word_error_rate() -> None:
    assert word_error_rate("Hello, world.", "hello world") == 0.0
    assert word_error_rate("a b c d", "a x c") == 0.5  # one substitution, one deletion
    assert word_error_rate("a b", "a b c d") == 1.0  # two insertions
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "noise") == 1.0
    assert word_error_rate("ഞാൻ വരാം", "ഞാൻ വരാം") == 0.0  # Malayalam words are words


def test_to_srt() -> None:
    srt = to_srt(Transcript("x", segments=[Segment(0, 1.5, " Hi "), Segment(3661.25, 3662, "Bye")]))
    assert srt == "1\n00:00:00,000 --> 00:00:01,500\nHi\n\n2\n01:01:01,250 --> 01:01:02,000\nBye\n"


def test_split_for_speech() -> None:
    long_sentence = "word " * 30
    chunks = split_for_speech("Short. Also short.\n" + long_sentence, max_chars=40)
    assert chunks[0] == "Short. Also short."
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks[1:]).split() == long_sentence.split()
    assert split_for_speech("x" * 50, max_chars=20) == ["x" * 20, "x" * 20, "x" * 10]
    with pytest.raises(ValueError):
        split_for_speech("hi", max_chars=5)


def test_speech_audio_type() -> None:
    assert SpeechAudio(b"", "wav").format == "wav"
