# Tasks: Lookahead Pre-Fetch Buffer

## 1. Configuration (lue/config.py)
- [ ] Add `LOOKAHEAD_SENTENCES = int(os.environ.get("LUE_LOOKAHEAD_SENTENCES", "30"))`
- [ ] Clamp `LOOKAHEAD_SENTENCES` to range 5-100
- [ ] Add `LOOKAHEAD_MAX_ERRORS = int(os.environ.get("LUE_LOOKAHEAD_MAX_ERRORS", "5"))`
- [ ] Add `LOOKAHEAD_TEMP_DIR = os.path.join(AUDIO_DATA_DIR, "lookahead")`
- [ ] `os.makedirs(LOOKAHEAD_TEMP_DIR, exist_ok=True)` and cleanup old files on module load
- [ ] Ensure `MAX_QUEUE_SIZE` defaults to `LOOKAHEAD_SENTENCES` if not explicitly overridden

## 2. LookaheadBuffer Class (lue/lookahead_buffer.py)
- [ ] Create `LookaheadBuffer` class with `__init__(reader, target_sentences, min_start_items)`
- [ ] Initialize `asyncio.Queue(maxsize=target_sentences)`
- [ ] Initialize `asyncio.Condition` for backpressure
- [ ] Track `lookahead_pos` (c, p, s), `is_running` flag, `is_at_end` flag, `error_count`
- [ ] Implement `async start_producer()` — launch producer coroutine
- [ ] Implement `async stop()` — set `is_running=False`, `notify_all`, drain queue, cleanup temp files
- [ ] Implement `async get()` — wraps `queue.get()` with timeout, returns `None` on stop
- [ ] Implement `async task_done()` — wraps `queue.task_done()`, notifies condition
- [ ] Implement `async refill()` — main producer loop: generates until `target_sentences` met or `is_at_end`
- [ ] Implement `_advance_lookahead()` — calls `reader._advance_position`, handles `None` (end) and wrap boundaries
- [ ] Implement `_generate_and_enqueue()` — get text at pos, sanitize, generate audio, put on queue
- [ ] Implement `_get_temp_path(seq_num)` — unique temp file path in `LOOKAHEAD_TEMP_DIR`
- [ ] Implement `_cleanup_temp_files()` — delete all `la_*` files for this session
- [ ] Implement `_handle_error()` — increment `error_count`, skip sentence, check consecutive limit
- [ ] Maintain monotonically increasing sequence counter per session for temp file naming
- [ ] Add logging for buffer state changes, generation events, errors

## 3. Producer Loop Integration (lue/audio.py)
- [ ] Replace current queue management in `_producer_loop` with `LookaheadBuffer.refill()`
- [ ] Modify `play_from_current_position()`: create `LookaheadBuffer`, start producer via `buffer.start_producer()`
- [ ] Wait for `PREBUFFER_MIN_ITEMS` (unchanged), then start player
- [ ] Modify `_player_loop`: use `buffer.get()` instead of `audio_queue.get()`, call `buffer.task_done()`
- [ ] Handle `None` sentinel from `buffer.get()` (end-of-book signal)
- [ ] Remove direct `audio_queue.put/get` calls from producer/player (use buffer)

## 4. Navigation and Cleanup
- [ ] In `stop_and_clear_audio()`: call `buffer.stop()` before cancelling tasks
- [ ] In `_restart_audio_after_navigation()`: create fresh `LookaheadBuffer` at new position
- [ ] Ensure temp files cleaned on `buffer.stop()`, navigation, and app exit
- [ ] Handle edge case: stop called during active generation (cancel in-flight, clean temp)

## 5. Edge Cases and Error Handling
- [ ] Short book (total sentences < lookahead target): fill to end, set `is_at_end`, continue normally
- [ ] End of book: `_advance_lookahead` returns `None`, `is_at_end=True`, sentinel queued
- [ ] Consecutive errors: stop after `LOOKAHEAD_MAX_ERRORS`, put sentinel, finish gracefully
- [ ] Condition timeout: if producer waits too long with no consumption signal, log warning
- [ ] Queue full during rapid navigation: `buffer.stop()` clears before new buffer created

## 6. Verification
- [ ] Test normal playback: queue stays at 25-30 items, no gaps between sentences
- [ ] Test with slow TTS (Kokoro): buffer builds during playback, refills correctly
- [ ] Test navigation: buffer cleared, new sentences generated from correct position
- [ ] Test short document: all sentences generated, playback ends cleanly
- [ ] Test error recovery: simulated TTS failure, sentence skipped, playback continues
- [ ] Test consecutive error limit: 5 failures in a row stops producer gracefully
- [ ] Test env var: `LUE_LOOKAHEAD_SENTENCES=10`, verify queue depth follows config
- [ ] Test temp file cleanup: verify no stale files after navigation, pause, exit
- [ ] Run `openspec validate lookahead-pre-fetch-buffer`
