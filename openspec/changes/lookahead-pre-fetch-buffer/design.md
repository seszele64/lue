# Design: Lookahead Pre-Fetch Buffer

## Technical Approach

Create a new `lue/lookahead_buffer.py` module containing the `LookaheadBuffer` class. The buffer manages a deep pipeline of pre-generated TTS audio using a single `asyncio.Queue(maxsize=LOOKAHEAD_SENTENCES)`. Coordination between the producer (which fills the buffer) and the player (which consumes it) uses an `asyncio.Condition` for precise, CPU-efficient signaling.

The producer runs a refill loop that maintains the queue at the target size. When the player consumes items via `task_done()`, the condition is notified, waking the producer to generate more sentences. Temp audio files use unique sequence-numbered names (`la_0001.mp3`) in a dedicated `lookahead/` directory under the cache path. On navigation, the entire buffer is destroyed and recreated at the new position, ensuring no stale state carries over.

## Architecture Decisions

### Decision: Single Queue vs Two-Tier Queue
- **Options:** Single `asyncio.Queue(maxsize=N)` vs a two-tier design (small ready-queue + larger pre-gen pool)
- **Pros of single queue:** Simpler implementation, all queued items are already "ready" for playback, no separate management of pools
- **Cons of single queue:** Producer and player share the same backing queue, slightly less flexible
- **Verdict:** Single queue — simplicity wins. All items in the queue are at max one sentence ahead of being needed.

### Decision: asyncio.Condition vs Sleep/Retry
- **Options:** `asyncio.Condition` (notify/wait) vs a polling loop with `asyncio.sleep()`
- **Pros of Condition:** Zero CPU waste when waiting, immediate wake-up when signal fires, precise producer/player coordination
- **Cons of Condition:** Slightly more complex setup, needs careful lock management
- **Verdict:** Use `asyncio.Condition` — the precision and efficiency gains justify the minor complexity.

### Decision: Unique Temp Files vs Expanded Rotating Buffers
- **Options:** Unique `la_XXXX.mp3` files per sentence vs expanding the existing `buffer_0`–`buffer_5` rotating slot system
- **Pros of unique files:** No slot conflicts, self-cleaning on navigation, no risk of overwriting files still in use
- **Cons of unique files:** More files on disk at any time, need cleanup logic
- **Verdict:** Unique temp files — simpler and safer. The dedicated lookahead/ directory is cleaned on navigation and exit.

### Decision: Full Invalidation on Navigation vs Partial Reuse
- **Options:** Destroy and recreate the entire buffer on navigation vs selectively reusing valid items
- **Pros of full invalidation:** Simpler, impossible to have stale/out-of-order state, straightforward error handling
- **Cons of full invalidation:** Lost work on previously generated sentences, slight delay on navigation
- **Verdict:** Full invalidation — simplicity and correctness are paramount. Navigation is infrequent enough that regeneration cost is negligible.

### Decision: Sentence-Count vs Time-Based Lookahead
- **Options:** Configurable sentence count target vs time-based (e.g., buffer 60 seconds of audio)
- **Pros of sentence-count:** Simple to reason about, no need to estimate sentence durations ahead of time, deterministic queue sizing
- **Cons of sentence-count:** Sentence duration can vary (short vs long sentences)
- **Verdict:** Sentence-count — sufficient for this use case. Time-based lookahead is explicitly out of scope.

## Data Flow

### Startup Sequence
```
play_from_current_position() called
  → Create LookaheadBuffer(target=LOOKAHEAD_SENTENCES, min_start=PREBUFFER_MIN_ITEMS)
  → buffer.start_producer()  — spawns async producer coroutine
  → Producer fills to PREBUFFER_MIN_ITEMS
  → Player spawns (waited: items >= PREBUFFER_MIN_ITEMS)
  → Producer continues filling to LOOKAHEAD_SENTENCES target
  → Steady state: player consumes, producer refills
```

### Steady State
```
Player calls buffer.get() → dequeues item from asyncio.Queue
  → Player plays audio
  → Player calls buffer.task_done()
    → task_done() calls queue.task_done()
    → Notifies asyncio.Condition
    → Producer wakes (was waiting on Condition)
    → Producer generates next sentence
    → Producer calls queue.put()  
    → If queue was full, producer goes back to waiting on Condition
```

### Navigation
```
User triggers navigation (next/prev/jump/click)
  → stop_and_clear_audio()
    → buffer.stop() 
      → Sets is_running = False
      → Notifies all waiters on Condition
      → Drains queue
      → Cleans up temp files in lookahead/
  → Cancel producer/player tasks
  → Update reader position
  → play_from_current_position() with new position
    → Creates fresh LookaheadBuffer
    → Repeats startup sequence
```

### End of Book
```
_advance_lookahead() returns None (no more sentences)
  → is_at_end = True  
  → Producer puts None sentinel on queue
  → Player receives None → sets playback_finished_event
  → Player loop exits cleanly
```

### Error Handling
```
TTS generation fails for a sentence
  → _handle_error() increments error_count
  → Logs error, skips sentence
  → Calls _advance_lookahead() to move past failed sentence
  → If error_count >= LOOKAHEAD_MAX_ERRORS (5):
    → Logs critical error
    → Puts None sentinel on queue
    → Stops producer gracefully
```

## State Machine

```
                        ┌──────────────────────────────────────┐
                        │                                      │
                        ▼                                      │
              ┌─────────────────┐                             │
              │   QUEUE_EMPTY   │                             │
              │  (0 items)      │──── producer generates ─────│──┐
              └────────┬────────┘                             │  │
                       │                                      │  │
              producer generates                              │  │
                       │                                      │  │
                       ▼                                      │  │
              ┌─────────────────┐     player consumes         │  │
              │  QUEUE_PARTIAL   │◄────────────────────────────┘  │
              │ (1..target-1)   │──── producer generates ────────┘
              └────────┬────────┘                                  
                       │                                          
              producer fills to target                             
                       │                                          
                       ▼                                          
              ┌─────────────────┐                                  
              │   QUEUE_FULL    │                                  
              │ (target items)  │──── player consumes ────────────┐
              └────────┬────────┘                                 │
                       │                                          │
              player consumes (one slot freed)                    │
                       │                                          │
                       ▼                                          │
              ┌─────────────────┐     navigation                  │
              │ QUEUE_DRAINING  │────────────────────────────►    │
              │ (buffer.stop()) │    (new QUEUE_EMPTY)            │
              └─────────────────┘                                 │
                                                                  │
        End of book:                                              │
        QUEUE_PARTIAL or QUEUE_FULL → None sentinel               │
        → Player receives None → playback_finished_event          │
                                                                  │
        Error limit reached:                                      │
        QUEUE_PARTIAL → None sentinel → graceful stop             │
                                                                  │
        All transitions from stop/nav/end reset to QUEUE_EMPTY ──┘
```

## File Changes

| File | Change |
|------|--------|
| `lue/lookahead_buffer.py` | **New file.** `LookaheadBuffer` class with queue, condition, producer, refill logic, temp file management, error handling |
| `lue/config.py` | Add `LOOKAHEAD_SENTENCES`, `LOOKAHEAD_MAX_ERRORS`, `LOOKAHEAD_TEMP_DIR` env var configs |
| `lue/audio.py` | Replace producer loop queue management with `buffer.refill()`. Modify `play_from_current_position()` to use LookaheadBuffer. Modify `_player_loop` to use `buffer.get()` / `buffer.task_done()`. Add lookahead temp file cleanup in `stop_and_clear_audio()`. |
