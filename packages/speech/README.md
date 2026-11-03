# ailab-speech

Speech-to-text and text-to-speech behind two small interfaces, so an application can swap
engines by name: local **Whisper** (faster-whisper, offline, private) or **gTTS** (online, many
languages including Malayalam). Engine-independent tools: **word error rate**, **SRT subtitles**
and **sentence chunking** for TTS.

```python
from ailab_speech import create_stt, create_tts, to_srt, word_error_rate

stt = create_stt("whisper", model="small")  # pip install "ailab-speech[whisper]"
result = stt.transcribe("meeting.wav")
print(result.text)
open("meeting.srt", "w").write(to_srt(result))

tts = create_tts("gtts")  # pip install "ailab-speech[gtts]"; sends text to Google
tts.synthesize("നമസ്കാരം", language="ml").save("hello.mp3")

print(word_error_rate("the reference text", result.text))
```

Add your own engine with `@register_stt("name")` or `@register_tts("name")`.

Install: `pip install "ailab-speech[whisper,gtts] @ git+https://github.com/mattsdata/ai-lab.git#subdirectory=packages/speech"`
