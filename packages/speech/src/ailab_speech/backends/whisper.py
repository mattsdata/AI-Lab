# ==================================================================================================
# 📜 ailab-speech :: backends.whisper
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/src/ailab_speech/backends/whisper.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Local speech recognition with OpenAI's Whisper models through faster-whisper (CTranslate2).
#    Runs offline on CPU or GPU, so audio never leaves the machine: the right default for
#    meetings, interviews and anything personal. Supports about 100 languages, Malayalam too.
#
#  🧠 DESIGN NOTES
#    - faster-whisper rather than the reference implementation: same models, several times
#      faster and lighter on memory, which matters on a free Colab GPU or a laptop CPU.
#    - compute_type "auto" picks float16 on GPU and int8 on CPU.
#    - Bytes are written to a temporary file because the decoder reads from paths (via FFmpeg).
#    - The model object is injectable, which keeps the adapter testable without downloads.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
"""Whisper speech-to-text backend (faster-whisper)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ailab_speech.core import Audio, Segment, Transcript, register_stt


@register_stt("whisper")
class WhisperSTT:
    def __init__(
        self,
        model: str = "small",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        engine: Any | None = None,
    ) -> None:
        if engine is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ImportError(
                    "the 'whisper' engine needs faster-whisper: pip install 'ailab-speech[whisper]'"
                ) from exc
            engine = WhisperModel(model, device=device, compute_type=compute_type)
        self._engine = engine
        self._beam_size = beam_size

    def transcribe(self, audio: Audio, *, language: str | None = None) -> Transcript:
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
                tmp.write(audio)
                tmp.flush()
                return self._run(tmp.name, language)
        return self._run(str(Path(audio)), language)

    def _run(self, path: str, language: str | None) -> Transcript:
        segments, info = self._engine.transcribe(path, language=language, beam_size=self._beam_size)
        parts = [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments]
        return Transcript(
            text=" ".join(p.text for p in parts).strip(),
            language=getattr(info, "language", language),
            segments=parts,
        )
