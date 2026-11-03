# ==================================================================================================
# 📜 ailab-speech :: backends
# --------------------------------------------------------------------------------------------------
#  File         : packages/speech/src/ailab_speech/backends/__init__.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Concrete speech engines. Each module adapts one library to SpeechToText or TextToSpeech
#    and registers itself by name.
#
#  🧠 DESIGN NOTES
#    - Intentionally imports nothing; engines load on demand through create_stt / create_tts.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: whisper (faster-whisper) and gtts.
# ==================================================================================================
"""Speech engines, loaded lazily by :func:`create_stt` and :func:`create_tts`."""
