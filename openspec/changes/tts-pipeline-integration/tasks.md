# Tasks

## 1. Pipeline Factory (`lue/tts_pipeline.py`)
- [ ] Create `TTSPipeline` class or `create_pipeline()` factory function that wires `TTSCache` → `ParallelTTSGen` → `LookaheadBuffer`
- [ ] Accept parameters: `reader`, `tts_model`, config overrides
- [ ] Initialize `TTSCache` first (from `lue/tts_cache.py`)
- [ ] Initialize `ParallelTTSGen` with `tts_model` and cache instance
- [ ] Initialize `LookaheadBuffer` with `parallel_gen` and `reader`
- [ ] Return fully wired pipeline with `start()`/`stop()` methods
- [ ] Handle optional components gracefully: if `TTSCache` import fails, create without cache; if `ParallelTTSGen` import fails, create without parallel; `LookaheadBuffer` is always required
- [ ] Add logging for pipeline creation (components available, configuration values)

## 2. Pipeline Lifecycle (`lue/audio.py`)
- [ ] Replace ad-hoc init in `play_from_current_position()` with `create_pipeline()` call
- [ ] Start sequence: `pipeline.start()` → `buffer.start_producer()` → wait for `PREBUFFER_MIN_ITEMS` → start player
- [ ] Stop sequence: `buffer.stop()` → `parallel_gen.shutdown()` → signal player to stop
- [ ] Navigation teardown: full `pipeline.stop()` → create new pipeline at new position → `pipeline.start()`
- [ ] Pause/resume: pipeline preserved across pause; producer suspended, player stopped; on resume, producer restarts refill
- [ ] Speed change: pipeline destroyed and recreated (speed affects timing/duration which invalidates cached timing; audio files remain valid)
- [ ] End-of-book: sentinel propagates through buffer → player → pipeline enters finished state
- [ ] Add a `pipeline_state` property: `IDLE`, `BUFFERING`, `PLAYING`, `PAUSED`, `FINISHED`, `ERROR`

## 3. Cross-Component Error Propagation
- [ ] Test error chain: `cache.store()` fails with disk full → logged → generation continues uncached → buffer gets result
- [ ] Test error chain: parallel worker fails for one sentence → marked `success=False` → buffer skips it → advances position → continues
- [ ] Test error chain: buffer hits consecutive error limit → stops producer → sentinel queued → player finishes remaining items
- [ ] Test error chain: `parallel_gen.shutdown()` called during mid-generation → workers cancelled → temp files cleaned → pipeline stops cleanly
- [ ] Ensure no error is silently swallowed: every error path has a log statement with context (sentence position, error type)
- [ ] Add `_pipeline_error` callback to reader: if pipeline enters `ERROR` state, notify UI

## 4. Temp File Coordination
- [ ] Verify cache files (persistent, in `TTS_CACHE_DIR`) are never cleaned by lookahead temp cleanup
- [ ] Verify lookahead temp files (session-only, in `LOOKAHEAD_TEMP_DIR`) are cleaned on navigation and exit
- [ ] Verify parallel generation temp files (in `LOOKAHEAD_TEMP_DIR` during generation) are cleaned after enqueue or on error
- [ ] Test: full navigation cycle leaves no stale files in either directory
- [ ] Add startup cleanup: on module load, scan `LOOKAHEAD_TEMP_DIR` and delete any leftover files from crashed sessions

## 5. Configuration Consolidation
- [ ] Group all TTS pipeline configs in `config.py` under a clear "TTS Pipeline" comment block
- [ ] Review naming consistency: all `TTS_CACHE_*`, `TTS_PARALLEL_*`, `LOOKAHEAD_*` keys
- [ ] Add inline doc comments explaining each key's purpose and valid range
- [ ] Ensure all env var mappings are documented (`LUE_LOOKAHEAD_SENTENCES`, `LUE_TTS_MAX_CONCURRENT`, etc.)
- [ ] Default `MAX_QUEUE_SIZE` to `LOOKAHEAD_SENTENCES` (deprecate `MAX_QUEUE_SIZE` as standalone config)
- [ ] Document `PREBUFFER_MIN_SECONDS` interaction: secondary threshold alongside `PREBUFFER_MIN_ITEMS`; if either is met, player starts. If `PREBUFFER_MIN_SECONDS` is desired as primary, set `PREBUFFER_MIN_ITEMS` very high.

## 6. UI Buffer Status (`lue/reader.py`)
- [ ] Add `buffer_depth` property on reader that returns current/target from the pipeline
- [ ] Add `SHOW_BUFFER_STATUS = False` config key
- [ ] When enabled, display "Buf: N/M" in TUI status area (e.g., right side of status bar)
- [ ] Buffer status read must be O(1) and non-blocking (no lock, just read `queue.qsize()`)
- [ ] Update buffer display on each UI refresh cycle (no separate task needed)
- [ ] When disabled, no buffer computation or display overhead

## 7. Performance Benchmarks
- [ ] Measure inter-sentence gap duration with all components active (target: < 100ms)
- [ ] Measure queue depth over time during 5-minute playback (target: stays above 80% of target)
- [ ] Measure cache hit ratio during re-read of previously played book (target: > 80% for repeated text)
- [ ] Measure startup latency from play command to first audio (target: < 2s cold, < 0.5s warm/cached)
- [ ] Measure navigation recovery time from jump to audio restart (target: < 2s)
- [ ] Compare against baseline (current feature/audio-pre-buffering branch) and document improvement
- [ ] Profile memory usage: RSS before/after full book playback (target: no growth after cleanup)

## 8. End-to-End Verification
- [ ] Full play-through of 10+ minute book with all components, verify zero audible gaps between sentences
- [ ] Repeated navigation (jump back/forward 20 times), verify no stale audio, no crashes, correct position
- [ ] Pause/resume cycle 10 times, verify correct position restoration, no double-play
- [ ] Speed change cycle (0.5x → 1.0x → 1.5x → 2.0x), verify correct speed, no gaps
- [ ] Stress test: rapid navigation + speed changes while producer is mid-generation (3+ parallel workers active)
- [ ] Memory leak check: play full book, monitor RSS, verify no growth after pipeline cleanup
- [ ] Run `openspec validate tts-pipeline-integration`
