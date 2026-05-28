# Tasks

## 1. TTSCache Class (lue/tts_cache.py)
- [x] Create `CacheHit` dataclass with `audio_path`, `timing_info`, `duration` fields
- [x] Create `TTSCache` class with `__init__(cache_dir, max_size_bytes, engine_name, output_format)`
- [x] Implement `_cache_key(text)` — SHA-256 hex digest truncated to 16 chars from `f"{engine_name}:{voice}:{sanitized_text}"`
- [x] Implement `lookup(text)` — check audio file + JSON sidecar existence; verify `text` field in sidecar matches requested text for collision detection
- [x] Implement `store(text, audio_path, timing_info)` — copy audio file to cache dir + write `timing_info` as JSON sidecar + update LRU index; use temp file + `os.replace()` for atomic writes
- [x] Implement `_load_index()` — read `.cache_index.json` (creates empty dict if missing or corrupted)
- [x] Implement `_save_index()` — write `.cache_index.json` atomically via temp file + `os.replace()`
- [x] Implement `evict_if_needed()` — LRU: sort entries by `last_accessed`, remove oldest until `total_size <= max_size_bytes`; update index after eviction
- [x] Implement `size_bytes()` — sum of all file sizes from the cache index
- [x] Implement `clear()` — remove all cached files and index
- [x] Add error handling: all file I/O wrapped in try/except; cache errors logged and never propagate to caller
- [x] Add logging for cache hits, misses, stores, evictions with sentence text preview (first 40 chars)
- [x] Add startup cache index validation: if `.cache_index.json` is corrupted, log warning, rebuild from disk scan

## 2. Cache Integration into Producer Loop (lue/audio.py)
- [x] Instantiate `TTSCache` in `play_from_current_position()` before producer starts
- [x] In `_producer_loop`: before calling `generate_audio_with_timing()`, compute cache key and call `cache.lookup()`
- [x] On cache hit: copy cached audio file to buffer slot via `shutil.copy2()`; use cached `timing_info` and `duration` from `CacheHit`; skip TTS generation
- [x] On cache miss: generate audio via `tts_model.generate_audio_with_timing()` as before; after generation, call `cache.store()` with `audio_path` and `timing_info`
- [x] Ensure cache operations never block queue put (no `await` inside critical path; cache file I/O is sync and fast)
- [x] Handle cache errors gracefully: any exception during cache lookup/store falls through to normal generation path

## 3. Configuration (lue/config.py)
- [x] Add `TTS_CACHE_ENABLED = True` (when `False`, skip all cache operations)
- [x] Add `TTS_CACHE_DIR = os.path.join(AUDIO_DATA_DIR, "tts_cache")`
- [x] Add `TTS_CACHE_MAX_SIZE_MB = 100`
- [x] Add `TTS_CACHE_MAX_SIZE_BYTES = TTS_CACHE_MAX_SIZE_MB * 1024 * 1024`
- [x] Call `os.makedirs(TTS_CACHE_DIR, exist_ok=True)` at config module load

## 4. Verification
- [x] Test with Edge TTS: generate same sentence twice, verify second call is a cache hit (no TTS API call)
- [x] Test with Kokoro TTS: same test, verify cache works with thread-pool-backed engine
- [x] Test LRU eviction: fill cache past 100MB limit, verify oldest files are removed
- [x] Test cache collision: create mock scenario where two different texts hash to same key, verify text verification in sidecar catches it
- [x] Test atomic writes: simulate kill during store, verify no partial/corrupt cache files remain
- [x] Test disk full: simulate ENOSPC during store, verify graceful fallback (logged warning, generation completes)
- [x] Test corrupted cache index: write invalid JSON to `.cache_index.json`, verify rebuild on next startup
- [x] Test cache disabled: `TTS_CACHE_ENABLED=False`, verify no cache files created, no lookups performed
- [x] Test navigation/re-read: play a paragraph, navigate back, verify all sentences served from cache
- [x] Run `openspec validate tts-cache-layer`
