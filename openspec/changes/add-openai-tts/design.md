# Design: Add OpenAI TTS Provider Integration

## Technical Approach

The OpenAI TTS provider follows the exact same plugin pattern as Edge TTS and Kokoro TTS:
1. A single file `lue/tts/openai_tts.py` containing the `OpenAITTS` class extending `TTSBase`
2. Dynamic discovery via the `TTSManager._discover_models()` method automatically registers it
3. The `openai` Python SDK's `AsyncOpenAI` client provides native async support, fitting seamlessly into Lue's async audio pipeline
4. Audio is saved as mp3 files using `response.stream_to_file()` for efficient streaming

## Architecture Decisions

### Decision: Use AsyncOpenAI Instead of run_in_executor
- Pros: Native async support, cleaner code, no thread pool overhead
- Cons: Requires `openai>=1.0.0` (which has httpx async support built in)
- Verdict: Use AsyncOpenAI — the SDK is designed for this

### Decision: Default to tts-1 Model
- Pros: Fastest response time, lowest cost ($15/1M chars), sufficient quality for eBook reading
- Cons: Lower quality than tts-1-hd or gpt-4o-mini-tts
- Verdict: Default to tts-1; model selection can be made configurable later

### Decision: No Word-Level Timing (Sentence Highlighting Fallback)
- Pros: No additional API calls, no extra cost, no added latency, accurate highlighting
- Cons: Less granular than word-level highlighting
- Verdict: Sentence-level highlighting is actually better than fake estimated timing

### Decision: Add supports_word_timing Property to TTSBase
- Pros: Clean abstraction, providers self-report capabilities, enables automatic UI adaptation
- Cons: Minor API change to base class (backward compatible via default True)
- Verdict: Add the property — it's a small change with clear benefits

## Data Flow

```
User selects --tts openai
       ↓
TTSManager discovers OpenAITTS class
       ↓
initialize() checks OPENAI_API_KEY, creates AsyncOpenAI client
       ↓
supports_word_timing returns False → WORD_HIGHLIGHT_MODE set to 0
       ↓
_producer_loop calls generate_audio_with_timing()
       ↓
generate_audio() calls client.audio.speech.create() → saves mp3
       ↓
get_raw_timing_data() returns [] → no word timing
       ↓
_audio_player_loop plays mp3 via ffplay
        ↓
_new_sentence_started fires → highlights entire sentence (no word tracking)
```

```
User switches back to edge (after using openai)
       ↓
supports_word_timing returns True
       ↓
Previously saved WORD_HIGHLIGHT_MODE restored (if exists)
       ↓
Word-level highlighting resumes with prior user setting
```

## File Changes

- `lue/tts/base.py` — Add `supports_word_timing` property
- `lue/tts/openai_tts.py` — New file: OpenAI TTS provider
- `lue/config.py` — Add `"openai": "alloy"` to TTS_VOICES
- `lue/reader.py` — Auto-adapt WORD_HIGHLIGHT_MODE based on provider capability
- `pyproject.toml` — Add `openai` optional dependency
- `VOICES.md` — Document OpenAI voices and pricing
