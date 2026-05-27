# Proposal: Lookahead Pre-Fetch Buffer

## Intent
Implement a configurable sliding-window lookahead buffer that continuously pre-generates TTS audio for N sentences ahead of playback, eliminating buffer underrun during continuous reading. Unlike the current fill-and-stop approach (max 12 items), the lookahead buffer maintains a deep pipeline of 30 sentences (configurable via LUE_LOOKAHEAD_SENTENCES env var), refilling as the player consumes items.

## Scope
**In scope:**
- Single-queue lookahead buffer with configurable target sentence count (default 30, range 5-100)
- asyncio.Condition-based backpressure between producer and player
- Sliding window: refill as player consumes, stop only at end-of-book
- Navigation invalidation: full buffer destruction and recreation on position jump
- Unique temp files per sentence (la_0001.mp3) in dedicated lookahead/ directory
- Producer starts before player; player starts after PREBUFFER_MIN_ITEMS met
- Error handling: skip failed sentences, stop after 5 consecutive errors
- Environment variable configuration: LUE_LOOKAHEAD_SENTENCES, LUE_LOOKAHEAD_MAX_ERRORS

**Out of scope:**
- Time-based lookahead (seconds instead of sentence count)
- Adaptive/dynamic lookahead depth
- Priority-based generation ordering
- Cross-chapter lookahead beyond book boundaries
- UI status indicators for buffer depth

## Dependencies
- **tts-cache-layer** (optional): TTSCache accelerates pre-fetch fill when sentences are re-encountered. Buffer functions correctly without cache.

## Approach
Create a `LookaheadBuffer` class in `lue/lookahead_buffer.py` that wraps an `asyncio.Queue(maxsize=target_sentences)` with an `asyncio.Condition` for backpressure. The buffer replaces the current ad-hoc queue management in the producer loop. Producer calls `buffer.refill()` which generates until queue reaches target. Player calls `buffer.get()` and `buffer.task_done()`, which signals the condition to wake the producer. On navigation, the buffer is stopped, cleared, and recreated at the new position. Temp audio files are stored in `~/.cache/lue/lookahead/` with sequence numbers, cleaned up on navigation and app exit.
