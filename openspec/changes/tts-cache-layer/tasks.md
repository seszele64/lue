# Tasks

## 1. TTSCache Class (lue/tts_cache.py)
- [ ] Create `CacheHit` dataclass with `audio_path`, `timing_info`, `duration` fields
- [ ] Create `TTSCache` class with `__init__(cache_dir, max_size_bytes, engine_name, output_format)`
- [ ] Implement `_cache_key(text)` — SHA-256 hex digest truncated to 16 chars from `f"{engine_name}:{voice}:{sanitized_text}"`
- [ ] Implement `lookup(text)` — check audio file + JSON sidecar existence; verify `text` field in sidecar matches requested text for collision detection
- [ ] Implement `store(text, audio_path, timing_info)` — copy audio file to cache dir + write `timing_info` as JSON sidecar + update LRU index; use temp file + `os.replace()` for atomic writes
- [ ] Implement `_load_index()` — read `.cache_index.json` (creates empty dict if missing or corrupted)
- [ ] Implement `_save_index()` — write `.cache_index.json` atomically via temp file + `os.replace()`
- [ ] Implement `evict_if_needed()` — LRU: sort entries by `last_accessed`, remove oldest until `total_size <= max_size_bytes`; update index after eviction
- [ ] Implement `size_bytes()` — sum of all file sizes from the cache index
- [ ] Implement `clear()` — remove all cached files and index
- [ ] Add error handling: all file I/O wrapped in try/except; cache errors logged and never propagate to caller
- [ ] Add logging for cache hits, misses, stores, evictions with sentence text preview (first 40 chars)
- [ ] Add startup cache index validation: if `.cache_index.json` is corrupted, log warning, rebuild from disk scan

## 2. Cache Integration into Producer Loop (lue/audio.py)
- [ ] Instantiate `TTSCache` in `play_from_current_position()` before producer starts
- [ ] In `_producer_loop`: before calling `generate_audio_with_timing()`, compute cache key and call `cache.lookup()`
- [ ] On cache hit: copy cached audio file to buffer slot via `shutil.copy2()`; use cached `timing_info` and `duration` from `CacheHit`; skip TTS generation
- [ ] On cache miss: generate audio via `tts_model.generate_audio_with_timing()` as before; after generation, call `cache.store()` with `audio_path` and `timing_info`
- [ ] Ensure cache operations never block queue put (no `await` inside critical path; cache file I/O is sync and fast)
- [ ] Handle cache errors gracefully: any exception during cache lookup/store falls through to normal generation path

## 3. Configuration (lue/config.py)
- [ ] Add `TTS_CACHE_ENABLED = True` (when `False`, skip all cache operations)
- [ ] Add `TTS_CACHE_DIR = os.path.join(AUDIO_DATA_DIR, "tts_cache")`
- [ ] Add `TTS_CACHE_MAX_SIZE_MB = 100`
- [ ] Add `TTS_CACHE_MAX_SIZE_BYTES = TTS_CACHE_MAX_SIZE_MB * 1024 * 1024`
- [ ] Call `os.makedirs(TTS_CACHE_DIR, exist_ok=True)` at config module load

## 4. Verification
- [ ] Test with Edge TTS: generate same sentence twice, verify second call is a cache hit (no TTS API call)
- [ ] Test with Kokoro TTS: same test, verify cache works with thread-pool-backed engine
- [ ] Test LRU eviction: fill cache past 100MB limit, verify oldest files are removed
- [ ] Test cache collision: create mock scenario where two different texts hash to same key, verify text verification in sidecar catches it
- [ ] Test atomic writes: simulate kill during store, verify no partial/corrupt cache files remain
- [ ] Test disk full: simulate ENOSPC during store, verify graceful fallback (logged warning, generation completes)
- [ ] Test corrupted cache index: write invalid JSON to `.cache_index.json`, verify rebuild on next startup
- [ ] Test cache disabled: `TTS_CACHE_ENABLED=False`, verify no cache files created, no lookups performed
- [ ] Test navigation/re-read: play a paragraph, navigate back, verify all sentences served from cache
- [ ] Run `openspec validate tts-cache-layer`
