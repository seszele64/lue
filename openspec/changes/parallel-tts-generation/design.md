# Design: Parallel TTS Generation

## Technical Approach

Create `lue/tts_parallel.py` with three main components:

1. **`_GenerationResult`** — A dataclass encapsulating the output of a single TTS generation: sentence position `(chapter, paragraph, sentence)`, audio file path, audio duration in seconds, timing info, and success status.

2. **`_OrderedBuffer`** — An in-memory buffer that accepts out-of-order completions and returns them in strict sentence order. Internally uses a dict keyed by `(c, p, s)` tuple. Tracks the next expected position via `_next_pos`. Callers submit results as they complete; `pop_next()` returns the result only when it matches `_next_pos`, advancing the position. Uses `asyncio.Event` to signal to `drain_next()` when the next in-order result becomes available. This avoids busy-waiting and lets the consumer await efficiently.

3. **`ParallelTTSGen`** — The orchestrator class. Accepts batch submissions from LookaheadBuffer via `submit(sentence_idx, text)` which queues pending generation requests without immediate dispatch. A separate `dispatch()` call spawns async worker tasks up to the semaphore limit. Each worker (`_generate_one`) first checks the optional TTSCache — cache hits resolve instantly and do not consume a semaphore slot. Cache misses proceed through the TTS engine. Results are placed into `_OrderedBuffer`. The consumer calls `drain_next()` which blocks until the next in-order result is available. Temp file naming uses session-sequence-based identifiers (`la_{seq_num}.{format}`) with a monotonically increasing per-session sequence counter for uniqueness.

## Architecture Decisions

### Decision 1: Slot-Based Ordered Buffer vs. Channel-Based
- **Option A (Slot-based dict):** Use a dict keyed by `(c, p, s)` to store out-of-order results. Consumer iterates from `next_pos`, popping contiguous results.
- **Option B (Channel per slot):** Use N result channels (one per concurrent worker), route each worker's output to a dedicated channel.
- **Verdict: Option A** — Simpler implementation, no worker-to-channel routing needed. The dict naturally handles arbitrary completion order. Single data structure is easier to reason about, clear on navigation, and debug.

### Decision 2: asyncio.Semaphore vs. ThreadPoolExecutor
- **Option A (async Semaphore):** Use `asyncio.Semaphore(max_concurrent)` to gate concurrent async workers.
- **Option B (ThreadPoolExecutor):** Submit blocking TTS calls to a thread pool; manage ordering externally.
- **Verdict: Option A** — Lue's TTS engines already support async (EdgeTTS has native async, Kokoro runs its own internal thread pool). An async semaphore is the natural fit for async-native engines. No thread pool management overhead.

### Decision 3: GPU Detection for Kokoro
- **Option A (Auto-detect GPU):** Check `torch.cuda.is_available()` when Kokoro is active; if GPU detected, default `max_concurrent=1`.
- **Option B (User config only):** Leave GPU concurrency management entirely to the user via env var.
- **Verdict: Option A** — On GPU, multiple concurrent Kokoro instances serialize on the same CUDA device anyway, adding overhead. Defaulting to 1 is the safe, sensible default. Users can still override via `LUE_TTS_MAX_CONCURRENT`.

### Decision 4: Full Invalidation on Navigation vs. Partial Reuse
- **Option A (Full invalidation):** On any position jump, stop the buffer, cancel all pending workers, clear the buffer, clean temp files, create a fresh buffer at the new position.
- **Option B (Partial reuse):** Cancel workers for skipped positions but keep completed results if they overlap with the new window.
- **Verdict: Option A** — Navigation typically jumps to a completely different position. Partial reuse adds complexity for negligible gain. Full invalidation is simpler, safer (no stale state), and easy to reason about.

### Decision 5: Sentence-Count Lookahead vs. Time-Based
- **Option A (Sentence count):** Maintain N sentences in the pre-generation buffer, regardless of their audio duration.
- **Option B (Time-based):** Maintain M seconds of audio in the buffer, adaptively generating shorter/longer sentences.
- **Verdict: Option A** — Sentence count is simpler to implement, matches the existing LookaheadBuffer API, and provides sufficient coverage (30 sentences at typical reading speed provides ~60-120s of buffer). Time-based lookahead can be added later as an enhancement.

## Data Flow

### Startup Sequence
```
LookaheadBuffer needs sentences
        ↓
Creates ParallelTTSGen(reader, LOOKAHEAD_SENTENCES, PREBUFFER_MIN_ITEMS)
        ↓
submit() called for each sentence in batch
        ↓
dispatch() spawns worker tasks (up to semaphore limit)
        ↓
Workers generate: _generate_one(sentence_idx, text)
        ↓
_cache check → hit: instant, miss: TTS generate
        ↓
Results placed in _OrderedBuffer via _OrderedBuffer.submit()
        ↓
Consumer calls drain_next() → blocks until next in-order result
        ↓
Returns _GenerationResult for each sentence in sequence
        ↓
Audio enqueued on lookahead queue in correct order
```

### Steady State
```
Player consumes sentence from queue
        ↓
LookaheadBuffer calls drain_next()
        ↓
_OrderedBuffer returns next result (or blocks if not ready)
        ↓
ParallelTTSGen.refill() is triggered → more workers spawned
        ↓
Queue maintained at target depth
```

### Navigation
```
User jumps to new position
        ↓
buffer.stop() called → is_running=False → notify_all
        ↓
Pending workers cancelled (asyncio.CancelledError caught)
        ↓
Temp files cleaned via _cleanup_temp_files()
        ↓
_OrderedBuffer cleared
        ↓
New ParallelTTSGen created at new position
        ↓
Fresh start: submit → dispatch → fill → play
```

### End of Book
```
_advance_position() returns None
        ↓
is_at_end = True
        ↓
Sentinel queued (None result)
        ↓
Consumer receives sentinel → signals LookaheadBuffer end
        ↓
Remaining buffered sentences played normally
```

### Error Handling
```
Worker: generate_audio() raises exception
        ↓
Error logged with sentence position
        ↓
error_count incremented
        ↓
success=False _GenerationResult buffered
        ↓
Consumer: drain_next() → failed result → skip sentence
        ↓
If error_count >= LOOKAHEAD_MAX_ERRORS (5):
        → producer stops gracefully
        → sentinel queued
```

## State Machine

```
            ┌──────────────────────────────────────────────────┐
            │                                                  │
            ▼                                                  │
    ┌──────────────┐     submit()/dispatch()     ┌──────────────┐
    │  IDLE        │────────────────────────────►│  FILLING     │
    │  (empty)     │                              │  (< target)  │
    └──────┬───────┘                              └──────┬───────┘
           │                                              │
           │ stop()                                       │ all submitted
           │ or nav                                       │ or refill()
           ▼                                              ▼
    ┌──────────────┐     drain reaches         ┌──────────────┐
    │  DRAINING    │◄──────────────────────────►│  FULL        │
    │  (cancelling)│     target                 │  (at target) │
    └──────┬───────┘                              └──────┬───────┘
           │                                              │
           │ cleanup                                      │ consumer
           │ complete                                     │ consumes
           ▼                                              ▼
    ┌──────────────┐                              ┌──────────────┐
    │  (back to    │                              │  REFILLING   │
    │   IDLE or    │                              │  (consumer   │
    │   new nav)   │                              │   triggered) │
    └──────────────┘                              └──────┬───────┘
                                                         │
                                                         │ complete
                                                         ▼
                                                    ┌──────────────┐
                                                    │  FULL        │
                                                    └──────────────┘
```

### State Descriptions

| State | Description | Entry | Exit |
|---|---|---|---|
| **IDLE** | No sentences loaded. Buffer empty. | Created or after stop() | submit() called |
| **FILLING** | Producer active, queue below target. | dispatch() called | Queue reaches target |
| **FULL** | Queue at target size. Producer suspended. | drain doesn't reduce below target | Consumer removes item |
| **REFILLING** | Queue below target after consumption. Producer refills. | Consumer drains one | Refill reaches target |
| **DRAINING** | Buffer being cleared (navigation). All workers cancelled. | stop() called | Cleanup complete |

## Thread Safety
- **Single-threaded event loop:** All operations run in Lue's async event loop. No thread safety concerns.
- **asyncio.Semaphore:** Async-safe. Used only in producer context.
- **_OrderedBuffer:** Only accessed from the event loop. Dict operations are atomic at the Python level.
- **Temp files:** Each worker generates a unique temp path via `la_{seq_num}.{format}` with a per-session monotonically increasing counter. No file path collisions. Cleanup is sequential (stop method cancels all workers first, then cleans).
- **No locks needed:** Python's asyncio is single-threaded within a loop. Shared mutable state (the dict) is only accessed from coroutines running on the same loop, never concurrently at the Python bytecode level.
- **Cancellation safety:** `asyncio.CancelledError` is caught in `_generate_one` to ensure temp files are cleaned up even if the worker is cancelled mid-flight.

## File Changes

### New Files
- `lue/tts_parallel.py` — `ParallelTTSGen`, `_OrderedBuffer`, `_GenerationResult`

### Modified Files
- `lue/audio.py` — Producer loop replaced with `LookaheadBuffer.refill()` pattern; `_player_loop` uses `buffer.get()` and `buffer.task_done()`
- `lue/config.py` — New environment variable configurations: `TTS_PARALLEL_ENABLED`, `TTS_MAX_CONCURRENT`, `TTS_MAX_CONCURRENT_FALLBACK`
- `lue/lookahead_buffer.py` — Integration with `ParallelTTSGen` for batch submission and ordered drain
