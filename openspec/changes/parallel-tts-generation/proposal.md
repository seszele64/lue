# Proposal: Parallel TTS Generation

## Intent
Generate multiple TTS sentences concurrently using asyncio.Semaphore to increase throughput, reducing the time to fill the lookahead buffer and eliminating buffer underrun during slow TTS backends. Builds on the LookaheadBuffer (Phase 2) and TTSCache (Phase 1).

## Scope
**In scope:**
- ParallelTTSGen class with asyncio.Semaphore(max_concurrent) dispatch
- Ordered output via _OrderedBuffer — sentences enqueued in correct sequence regardless of completion order
- Integration with LookaheadBuffer: batch submission, ordered drain
- Integration with TTSCache: cache hits resolve instantly, misses go through TTS engine
- Configurable max concurrency (TTS_MAX_CONCURRENT, default 3)
- Per-engine concurrency awareness (GPU Kokoro defaults to 1)
- Graceful degradation: single concurrent worker fallback on import errors

**Out of scope:**
- Cross-sentence optimization (sharing context, batching API calls)
- Dynamic concurrency adjustment based on latency
- Priority-based ordering (e.g., closer sentences first)
- Thread-pool concurrency for blocking TTS engines (already handled at engine level)

## Dependencies
- **tts-cache-layer** (optional): TTSCache enables instant resolution for cached sentences within parallel workers.
- **lookahead-pre-fetch-buffer** (required): ParallelTTSGen is used by LookaheadBuffer for batch generation.

## Approach
Create `ParallelTTSGen` in `lue/tts_parallel.py`. Accept batch submissions from the LookaheadBuffer. Each batch item is dispatched as an async task gated by asyncio.Semaphore. Results stored in `_OrderedBuffer` — a dict keyed by `(c, p, s)` that holds out-of-order completions. `drain_next()` returns the next in-order result, blocking via `asyncio.Event` until available. Cache hits bypass the semaphore entirely (instant resolution). For Kokoro on GPU, detect GPU and default concurrency to 1.
