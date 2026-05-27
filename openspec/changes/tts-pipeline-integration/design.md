# Design: TTS Pipeline Integration and End-to-End Verification

## Technical Approach

The three components compose as a layered architecture:

```
LookaheadBuffer (orchestrator)
    └── ParallelTTSGen (engine)
            └── TTSCache (accelerator)
```

- **LookaheadBuffer** is the top-level orchestrator. It holds a `ParallelTTSGen` instance and uses it for all sentence generation. The buffer calls `parallel_gen.submit()` for batch generation, drains results via `drain_next()`, and enqueues them for playback. All temp file management (session-only) lives in LookaheadBuffer.
- **ParallelTTSGen** is the generation engine. It holds an optional `TTSCache` instance. Before dispatching TTS work to a worker, it checks the cache. On hit, it returns cached audio immediately. On miss, it generates via the TTS backend and stores the result.
- **TTSCache** is the acceleration layer. It is stateless (content-addressed by SHA256 hash) and persists across sessions. Cache files live in a dedicated directory separate from lookahead temp files.

Initialization happens once in `play_from_current_position()`. Navigation triggers a full teardown and recreation: `buffer.stop()` propagates to `parallel_gen.shutdown()`, then temp file cleanup, then new pipeline creation.

## Architecture Decisions

### Decision 1: Orchestrator Pattern — LookaheadBuffer is the Top-Level Coordinator
- **Context:** Three independently developed components need to work together. The cache is purely reactive, the parallel engine is a worker pool, and the buffer manages playback timing.
- **Decision:** LookaheadBuffer acts as the orchestrator, holding ParallelTTSGen and TTSCache as dependencies.
- **Rationale:** The buffer is closest to the playback loop and best positioned to make timing decisions (when to pre-buffer, when to refill, when to stop). ParallelTTSGen and TTSCache remain independently testable since they have no circular dependencies.
- **Consequence:** Buffer code needs minor refactoring to accept a `ParallelTTSGen` instance instead of a raw callable.

### Decision 2: Single Initialization Point — `play_from_current_position()`
- **Context:** Currently, pipeline components are initialized across `_initialize_state()`, `play_from_current_position()`, and inline in the producer loop.
- **Decision:** All pipeline creation happens in a single code path: `play_from_current_position()`.
- **Rationale:** Single point of control ensures correct ordering, simplifies teardown, and makes navigation recreation trivial.
- **Consequence:** `_initialize_state()` will no longer set up audio queue or buffer objects. Navigation functions call `play_from_current_position()` which handles everything.

### Decision 3: Graceful Degradation Chain — Each Component is Optional
- **Context:** Users may have partial installations or missing dependencies (e.g., no cache directory, no concurrent worker support).
- **Decision:** If ParallelTTSGen is unavailable, LookaheadBuffer falls back to sequential generation. If TTSCache is unavailable, ParallelTTSGen operates without caching. Both degradations are silent and automatic.
- **Rationale:** The pipeline must never prevent playback. Every optional dependency has a "missing" mode that works.
- **Consequence:** All `import` statements for optional components are wrapped in try/except. Component availability is checked at pipeline construction time.

### Decision 4: `asyncio.Condition` Instead of Sleep/Retry for Backpressure
- **Context:** The producer needs to know when the player has consumed items so it can refill.
- **Decision:** Use `asyncio.Condition` with the queue's lock. Producer waits on condition; player notifies after each `task_done()`.
- **Rationale:** Zero CPU waste. Producer is suspended precisely when the queue is full and woken precisely when an item is consumed. No polling, no race conditions.
- **Consequence:** Slightly more complex initialization (need shared condition), but significantly better than sleep/retry loops.

### Decision 5: Unique Temp Files (la_0001.mp3) Instead of Rotating Slots
- **Context:** The current system uses fixed buffer slots (`buffer_0.mp3` through `buffer_7.mp3`) that are overwritten cyclically.
- **Decision:** Use unique temporary filenames per sentence (e.g., `la_0001.mp3`, `la_0002.mp3`) in a dedicated `lookahead/` directory.
- **Rationale:** No slot conflicts between concurrent generation tasks. No stale-read risk (old slot content mistaken for current). Simpler cleanup: just delete the directory. Self-documenting via sequence numbers.
- **Consequence:** Slightly more files on disk, but each is small (< 100KB) and cleanup is trivial.

### Decision 6: Full Invalidation on Navigation Instead of Partial Reuse
- **Context:** When the user jumps to a new position, some cached sentences might still be valid (same chapter, forward jump).
- **Decision:** Destroy the entire buffer and recreate from scratch on every navigation.
- **Rationale:** Simpler to implement and verify. Reuse requires tracking which temp files are still valid, which introduces subtle ordering bugs. Navigation is user-initiated and infrequent; the ~100ms recreation cost is negligible compared to TTS latency.
- **Consequence:** Cache hit for previously generated sentences still provides acceleration — the old audio files in the persistent cache directory are reused across navigations. Only the lookahead temp files are destroyed.

## Data Flow

### Startup Sequence
```
reader.play_from_current_position()
    ↓
create TTSCache(cache_dir, max_size_mb)
    ↓
create ParallelTTSGen(tts_model, max_concurrent, cache)
    ↓
create LookaheadBuffer(reader, target_sentences, parallel_gen)
    ↓
buffer.start_producer()     ← launches async producer coroutine
    ↓
producer fills queue → wait buffer.qsize() >= PREBUFFER_MIN_ITEMS
    ↓
start player task            ← player loop calls buffer.get()
    ↓
continuous refill: player consumes → condition notify → producer wakes → generates → puts
```

### Steady State
```
Player consumes sentence via buffer.get()
    ↓
Player calls buffer.task_done()
    ↓
Buffer's condition.notify() wakes producer
    ↓
Producer calls parallel_gen.submit(text, seq_num)
    ↓
ParallelTTSGen checks TTSCache.has(key):
    ├─ HIT:  cache.get(key, temp_path) → instant → return
    └─ MISS: generate via TTS → cache.put(key, temp_path) → store → return
    ↓
Producer calls drain_next() → gets completed item in order
    ↓
Producer queue.put() → item available for player
    ↓
Loop repeats: queue stays near LOOKAHEAD_SENTENCES
```

### Navigation
```
reader._restart_audio_after_navigation()
    ↓
buffer.stop()               ← sets is_running=False, notify_all
    ↓
parallel_gen.shutdown()     ← cancels pending workers
    ↓
cache.cleanup_temp_refs()   ← no-op for stateless cache
    ↓
buffer._cleanup_temp_files() ← deletes all la_* files
    ↓
create new pipeline         ← same as startup, at new position
    ↓
restart
```

### End of Book
```
producer calls _advance_lookahead()
    ↓
reader._advance_position() returns None
    ↓
is_at_end = True
    ↓
sentinel queued to buffer
    ↓
player consumes sentinel → terminates
```

### Error Handling
```
worker TTS failure
    ↓
parallel_gen._handle_error():
    error_count += 1
    if error_count >= LOOKAHEAD_MAX_ERRORS:
        → stop producer gracefully
        → queue sentinel for player
        → log final error
    else:
        → log error
        → skip sentence
        → advance to next
        → continue
```

## State Machine

The full pipeline operates in one of four states:

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
            ┌──────────────┐                       ┌──────────────┐
    ┌──────►│ QUEUE_EMPTY  │───────item put────────►│ QUEUE_PARTIAL│
    │       │              │◄──────item consume─────│              │
    │       └──────┬───────┘                       └──────┬───────┘
    │              │                                       │
    │              │ qsize < target                    target reached
    │              │                                       │
    │              ▼                                       ▼
    │       ┌──────────────┐                       ┌──────────────┐
    │       │ QUEUE_FULL   │───────item consume───►│ QUEUE_PARTIAL│
    │       │ (producer    │       → wake producer  │ (playback    │
    │       │  suspended)  │                        │  continues)  │
    │       └──────────────┘                       └──────┬───────┘
    │              ▲                                       │
    │              │                         last item consumed
    │              │                                       │
    └──────────────┘                               ┌──────────────┐
                                                   │ QUEUE_DRAINING│
                                                   │ (end-of-book) │
                                                   └──────────────┘
```

**State transitions:**
| Transition | From | To | Trigger |
|---|---|---|---|
| Pipeline created, first item generated | IDLE | QUEUE_EMPTY | buffer.start_producer() |
| Queue gets its first item | QUEUE_EMPTY | QUEUE_PARTIAL | producer.put() |
| Queue reaches target_sentences | QUEUE_PARTIAL | QUEUE_FULL | producer.put(), qsize == maxsize |
| Player consumes an item from full queue | QUEUE_FULL | QUEUE_PARTIAL | buffer.task_done(), consumer wakes producer |
| Producer refills while below max | QUEUE_PARTIAL | remains QUEUE_PARTIAL | producer.put() (1 ≤ qsize < max) |
| Player consumes from partial queue | QUEUE_PARTIAL | remains QUEUE_PARTIAL | buffer.get() (1 ≤ qsize < max) |
| Player consumes last item | QUEUE_PARTIAL | QUEUE_EMPTY | buffer.get(), qsize == 0 |
| Player consumes sentinel | QUEUE_PARTIAL | QUEUE_DRAINING | buffer.get() returns sentinel |
| Navigation stop | any | DESTROYED | buffer.stop() called |
| End-of-book, player terminates | QUEUE_DRAINING | FINISHED | player task completes |

## File Changes

| File | Change |
|---|---|
| `lue/lookahead_buffer.py` | Modify `__init__` to accept `ParallelTTSGen`; replace raw generator reference with `parallel_gen.submit()` / `drain_next()`; add `stop()` method; add `_cleanup_temp_files()`; add condition-based backpressure |
| `lue/tts_parallel.py` | Modify to accept optional `TTSCache`; add cache check before generation; add `shutdown()` method; maintain `error_count`; add `submit()` and `drain_next()` for ordered result collection |
| `lue/audio.py` | Replace ad-hoc queue/producer management in `play_from_current_position()` with unified pipeline initialization; modify `_restart_audio_after_navigation()` to teardown and recreate; `stop_and_clear_audio()` calls `buffer.stop()`; remove old `_producer_loop` queue management code |
| `lue/config.py` | Add `LOOKAHEAD_SENTENCES` (env `LUE_LOOKAHEAD_SENTENCES`, default 30, clamp 5–100); add `LOOKAHEAD_MAX_ERRORS` (env `LUE_LOOKAHEAD_MAX_ERRORS`, default 5); add `LOOKAHEAD_TEMP_DIR`; group all TTS pipeline config keys together |
| `lue/tts_cache.py` | No interface changes needed; verify cache API is compatible with `ParallelTTSGen` usage pattern |
| `lue/tts_pipeline.py` | **NEW FILE — required (factory function).** Contains `create_pipeline()` factory function that wires the three components together in correct order. Keeps `audio.py` from growing too large. |
