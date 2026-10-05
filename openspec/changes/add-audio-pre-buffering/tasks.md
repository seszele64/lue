# Tasks

## 1. Configuration Constants
- [ ] Add `MAX_QUEUE_SIZE = 8` (up from 4) in `lue/config.py`
- [ ] Add `AUDIO_BUFFERS` entries for indices 6-7 (buffer_6, buffer_7) to support larger queue
- [ ] Add `MIN_BUFFER_SECONDS = 15` — minimum buffered audio duration before playback starts
- [ ] Add `MAX_CONCURRENT_GENERATIONS = 2` — asyncio.Semaphore limit for parallel TTS
- [ ] Add `CACHE_DIR` — path for TTS content-hash cache (default: `AUDIO_DATA_DIR/cache`)
- [ ] Add `MAX_CACHE_SIZE_MB = 500` — LRU eviction threshold

## 2. TTS Cache (`lue/cache.py`)
- [ ] Create `TTSCache` class with `__init__(cache_dir, max_size_mb)` constructor
- [ ] Implement `_make_key(sanitized_text, voice, speed)` → SHA256 hex digest
- [ ] Implement `has(key)` → bool (checks file existence)
- [ ] Implement `get(key, output_path)` → copies cached file to output_path; updates mtime
- [ ] Implement `put(key, source_path)` → copies source to cache; triggers eviction if needed
- [ ] Implement `_evict_if_needed()` → removes oldest files by mtime until under max_size_mb
- [ ] Implement `_dir_size_mb()` → du-like size calculation for cache directory
- [ ] Add error handling: cache misses/puts fail gracefully, never block TTS generation
- [ ] Add logging for cache hits, misses, stores, and evictions

## 3. Pre-Buffer Playback Start (`lue/audio.py`)
- [ ] Add `reader.pre_buffer_ready` — `asyncio.Event` for producer→player coordination
- [ ] Add `reader.buffered_duration` — running total of audio duration in queue (float)
- [ ] Modify `_producer_loop()`: increment `buffered_duration` on each queue put; set `pre_buffer_ready` when >= MIN_BUFFER_SECONDS
- [ ] Modify `play_from_current_position()`: spawn producer first, await `pre_buffer_ready` (with 30s timeout), then spawn player
- [ ] Ensure `stop_and_clear_audio()` resets `pre_buffer_ready` and `buffered_duration`
- [ ] Test: normal start, cached start (instant), short document, timeout fallback

## 4. Parallel TTS Generation (`lue/audio.py`)
- [ ] Add `asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)` to producer loop
- [ ] Refactor `_producer_loop()`: iterate sentences with sequence numbers (0, 1, 2, ...)
- [ ] Implement `_generate_single(reader, text, seq, buffer_slot)` — async function wrapping TTS gen + cache + ffprobe
- [ ] Implement ordered output: maintain `pending_tasks: dict[int, asyncio.Task]` and `next_seq_to_enqueue`
- [ ] On each loop iteration: check if `pending_tasks[next_seq_to_enqueue]` is done → enqueue result → advance
- [ ] When semaphore available and next sequence slot is free: spawn new `_generate_single` task
- [ ] Ensure buffer slot allocation is safe across concurrent tasks (round-robin or allocated per-task)
- [ ] Handle task exceptions: log error, skip failed sentence, advance sequence without enqueuing
- [ ] Test: sequential ordering preserved, concurrent generation works (verify with logging/timing)

## 5. Cache Integration into Producer
- [ ] Instantiate `TTSCache` in `reader` during `_initialize_state()`
- [ ] In `_generate_single()` (or inline in producer): compute cache key → check cache → reuse or generate
- [ ] On cache hit: copy cached audio to buffer slot, run ffprobe for duration, use cached timing if available
- [ ] On cache miss: generate audio, save to cache, then continue as before
- [ ] Handle cache errors gracefully: fall through to normal generation on any cache exception
- [ ] Test: repeated text hits cache, unique text generates fresh, cache eviction works under limit

## 6. Integration and Verification
- [ ] Run `openspec validate add-audio-pre-buffering` to verify artifact format
- [ ] Test with Edge TTS backend: verify pre-buffer reduces startup gap, queue stays non-empty
- [ ] Test with Kokoro TTS backend: verify parallel generation works with thread-pool-backed backend
- [ ] Test with OpenAI TTS backend: verify cache and pre-buffer work with streaming API backend
- [ ] Test navigation (next/prev/jump/click): verify audio restarts correctly with pre-buffer
- [ ] Test pause/resume: verify pre-buffer re-fills on resume, no double-play
- [ ] Test speed changes: verify pre-buffer and queue work with adjusted playback speeds
- [ ] Test error handling: simulate network failure, verify graceful degradation (skip sentence, continue)
- [ ] Test with short document: verify no pre-buffer timeout deadlock
- [ ] Verify no regressions: existing UI, navigation, progress saving, word highlighting still work
