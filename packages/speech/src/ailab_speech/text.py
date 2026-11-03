# ==================================================================================================
# 📜 ailab-speech :: text
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/src/ailab_speech/text.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    What happens around the engines: measuring transcription quality and turning transcripts
#    into subtitles. Pure Python, so it works with any engine and is fully testable offline.
#
#  🔧 WHAT IT DOES
#    - word_error_rate : (substitutions + deletions + insertions) / reference words, the
#                        standard ASR metric; compare engines on your own recordings.
#    - to_srt          : SubRip subtitles from transcript segments.
#    - split_for_speech: break long text into sentence chunks under a length limit, because
#                        most TTS services reject or truncate long inputs.
#
#  🧠 DESIGN NOTES
#    - WER normalises case and punctuation first, so "Hello, world." and "hello world" match;
#      otherwise the metric measures formatting, not recognition.
#    - Edit distance uses two rolling rows: O(n) memory for long transcripts.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release.
# ==================================================================================================
"""Word error rate, SRT subtitles and text chunking for speech engines."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import re

from ailab_speech.core import Transcript

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")  # letters/digits in any script, keeps "don't"


# ========================================
# 📂 Evaluation
# ========================================
def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the number of reference words.

    >>> word_error_rate("the cat sat", "the cat sat")
    0.0
    >>> round(word_error_rate("the cat sat on the mat", "the cat sit on mat"), 3)
    0.333
    """
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


# ========================================
# 📂 Subtitles
# ========================================
def _timestamp(seconds: float) -> str:
    ms = round(max(seconds, 0.0) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    """SubRip (.srt) text, one cue per segment."""
    cues = [
        f"{i}\n{_timestamp(seg.start)} --> {_timestamp(seg.end)}\n{seg.text.strip()}\n"
        for i, seg in enumerate(transcript.segments, start=1)
    ]
    return "\n".join(cues)


# ========================================
# 📂 Chunking for TTS
# ========================================
def split_for_speech(text: str, *, max_chars: int = 200) -> list[str]:
    """Sentence-aligned chunks of at most ``max_chars`` (long sentences split at spaces)."""
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+|\n+", text) if s.strip()]
    pieces: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        pieces.append(sentence)
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    return chunks
