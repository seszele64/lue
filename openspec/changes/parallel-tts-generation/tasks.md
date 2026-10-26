# Tasks

## 1. Core Classes (lue/tts_parallel.py)
- [ ] Create _GenerationResult dataclass: sentence_idx (c,p,s), audio_path, duration, timing_info, success
- [ ] Create _OrderedBuffer class: dict-based storage, submit(result), pop_next() → _GenerationResult | None
- [ ] _OrderedBuffer: set_expected(pos) to advance, pending_count property
- [ ] Create ParallelTTSGen class: __init__(tts_model, max_concurrent, cache=None)
- [ ] Initialize asyncio.Semaphore(max_concurrent), _OrderedBuffer, asyncio.Event for completion signaling
- [ ] Implement submit(sentence_idx, text) → stores pending task, no immediate dispatch
- [ ] Implement async dispatch() → spawns workers for pending tasks up to semaphore limit
- [ ] Implement async _generate_one(sentence_idx, text) → internal worker: cache check → TTS gen → buffer result
- [ ] Implement async drain_next() → blocks until next in-order result available, returns _GenerationResult
- [ ] Implement shutdown() → cancel pending workers, clear buffer
- [ ] Cache integration in _generate_one: cache.lookup() → if hit, resolve instantly; if miss, generate then cache.store()
- [ ] Unique temp file per worker: la_{seq_num}.{format} in LOOKAHEAD_TEMP_DIR, using monotonically increasing session sequence counter
- [ ] Error handling in worker: catch exceptions, set success=False, buffer failed result for skip
- [ ] Logging for batch dispatch, completions, cache hits, errors

## 2. Kokoro GPU Detection
- [ ] Add _detect_kokoro_gpu() function: check for torch.cuda.is_available()
- [ ] In ParallelTTSGen.__init__: if tts_model.name == "kokoro" and GPU detected, default max_concurrent=1
- [ ] Log warning when GPU Kokoro concurrency is reduced
- [ ] Make GPU detection configurable: allow override via TTS_MAX_CONCURRENT config

## 3. Configuration (lue/config.py)
- [ ] Add TTS_PARALLEL_ENABLED = True
- [ ] Add TTS_MAX_CONCURRENT = int(os.environ.get("LUE_TTS_MAX_CONCURRENT", "3"))
- [ ] Clamp TTS_MAX_CONCURRENT to 1-8 range
- [ ] Add TTS_MAX_CONCURRENT_FALLBACK = 1 (used when parallel not available)

## 4. Lookahead Buffer Integration (lue/lookahead_buffer.py)
- [ ] Modify _generate_and_enqueue() to use ParallelTTSGen for batch generation
- [ ] Submit batch of sentences (up to queue space remaining, capped at max_concurrent * 2)
- [ ] Drain results in order via drain_next()
- [ ] Handle failed results: skip sentence, advance position, continue
- [ ] Fall back to sequential generation if ParallelTTSGen import fails or TTS_PARALLEL_ENABLED=False
- [ ] Pass TTSCache instance to ParallelTTSGen for cache integration

## 5. Ordering and Concurrency Verification
- [ ] Test sequential ordering: generate 5 sentences concurrently via ParallelTTSGen, verify drain_next() returns them in chapter/paragraph/sentence order regardless of completion order
- [ ] Test semaphore behavior: submit 10 sentences with max_concurrent=3, verify never more than 3 active TTS calls simultaneously
- [ ] Test _OrderedBuffer correctness: submit results in reverse order, verify pop_next() only returns when expected position is available
- [ ] Test ordered buffer edge case: submit same position twice, verify no duplicate key error
- [ ] Test drain_next() blocking: verify it awaits correctly when expected position not yet available, then returns immediately when it arrives

## 6. Component Integration Verification
- [ ] Test that LookaheadBuffer refill() correctly uses ParallelTTSGen for batch submission
- [ ] Test batch sizing: verify batch size = min(remaining_slots, max_concurrent * 2)
- [ ] Test that TTSCache integration works within parallel workers: cache hit → instant resolution, cache miss → TTS generation
- [ ] Test that cache.store() is called after each miss within the worker
- [ ] Test graceful fallback: simulate ParallelTTSGen import error, verify LookaheadBuffer falls back to sequential generation
- [ ] Test TTS_PARALLEL_ENABLED=False: verify sequential code path used, no ParallelTTSGen created
- [ ] Test shutdown: call shutdown() during active generation, verify all workers cancelled, no orphaned tasks

## 7. Error Handling in Parallel Context
- [ ] Test one worker fails while others succeed: verify failed sentence is skipped, others enqueued in order
- [ ] Test all workers fail simultaneously: verify all marked success=False, buffer handles all skips, continues
- [ ] Test worker failure during cache.store(): verify generation result is still delivered even if caching fails
- [ ] Test semaphore release on worker failure: verify semaphore is released even when _generate_one raises (via async with)
- [ ] Test GPU detection: mock torch available/unavailable, verify max_concurrent auto-adjusted
- [ ] Test GPU detection with import error (torch not installed): verify graceful fallback without crash

## 8. End-to-End Verification
- [ ] Test full pipeline: TTSCache + ParallelTTSGen + LookaheadBuffer working together, zero gaps during 3-minute playback
- [ ] Test concurrent generation with Edge TTS: verify 3 parallel requests, all complete correctly
- [ ] Test concurrent generation with Kokoro (GPU): verify concurrency auto-reduces to 1
- [ ] Test concurrent generation with Kokoro (CPU): verify 3 concurrent generations work
- [ ] Test temp file uniqueness: concurrent workers use distinct file paths (no collisions), all cleaned after enqueue
- [ ] Stress test: rapid navigation while 3 workers are mid-generation, verify clean shutdown and restart
- [ ] Run openspec validate parallel-tts-generation
