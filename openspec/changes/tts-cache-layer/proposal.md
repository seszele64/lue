# Proposal: TTS Audio Result Cache

## Intent
Cache generated TTS audio on disk to avoid redundant API calls and model inference when the same text is re-encountered. This eliminates 1-5s per-sentence generation latency on cache hits during re-reading, navigation back, pause/resume, and speed changes.

## Scope
**In scope:**
- SHA-256 based content-hash cache key from (sanitized_text, engine_name, voice)
- Disk storage in user cache directory with JSON sidecar for timing metadata
- LRU eviction with configurable max size (default 100MB)
- Cache hit/miss transparent to producer loop — cache checked before TTS call
- Atomic writes to prevent corruption on interrupted writes
- Graceful degradation: cache errors fall through to normal generation

**Out of scope:**
- Cross-session cache sharing between different users
- Cache invalidation on TTS engine version changes (manual cleanup only)
- In-memory caching (disk only)
- Cache warming/pre-population
- Cache statistics dashboard

## Approach
Create a `TTSCache` class in a new `lue/tts_cache.py` module. The cache key is SHA-256 of `f"{engine}:{voice}:{sanitized_text}"`, truncated to 16 hex chars. Each cache entry consists of the audio file (mp3/wav) and a JSON sidecar storing timing_info (total_duration, word_timings, word_mapping). The cache sits as a facade in the producer loop, not inside TTSBase — checked before `generate_audio_with_timing()`.

On hit: copy cached audio to buffer slot, return cached timing. On miss: generate, then store both audio and timing sidecar. LRU eviction triggered on every store; removes oldest entries by access time until under max_size_mb. Atomic writes via `os.replace()` from temp file.
