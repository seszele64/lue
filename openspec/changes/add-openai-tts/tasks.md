# Tasks

## 1. TTSBase Interface Extension
- [x] Add `supports_word_timing` property to `TTSBase` (default returns `True`)
- [x] Add property documentation explaining its purpose

## 2. OpenAI TTS Provider Implementation
- [x] Create `lue/tts/openai_tts.py` with `OpenAITTS` class
- [x] Implement `name` property returning `"openai"`
- [x] Implement `output_format` property returning `"mp3"`
- [x] Implement `__init__` with default voice from config
- [x] Implement `initialize()` checking for `openai` package and `OPENAI_API_KEY`
- [x] Implement `generate_audio()` using `AsyncOpenAI` client
- [x] Implement `get_raw_timing_data()` returning empty list
- [x] Implement `warm_up()` with a short text generation
- [x] Override `supports_word_timing` to return `False`

## 3. Configuration Updates
- [x] Add `"openai": "alloy"` to `TTS_VOICES` in `lue/config.py`

## 4. Dependency Management
- [x] Add `openai = ["openai>=1.0.0"]` to `[project.optional-dependencies]` in `pyproject.toml`

## 5. Highlight Mode Auto-Adaptation
- [x] Store previous WORD_HIGHLIGHT_MODE when switching to non-timing provider
- [x] Auto-set WORD_HIGHLIGHT_MODE to 0 when provider has supports_word_timing = False
- [x] Restore previous mode when switching back to timing-capable provider
- [x] Update reader initialization to handle new property

## 6. Documentation
- [x] Add OpenAI TTS section to VOICES.md with available voices, pricing, and setup instructions

## 7. Validation & Testing
- [ ] Run `openspec validate add-openai-tts` and fix any issues
- [ ] Verify provider appears in `--tts` CLI choices
- [ ] Test initialization with and without OPENAI_API_KEY
- [ ] Test audio generation with a short text
- [ ] Verify sentence-level highlighting works correctly with OpenAI TTS
