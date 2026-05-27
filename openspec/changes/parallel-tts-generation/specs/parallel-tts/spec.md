# Delta for Parallel TTS

## DEPENDS ON

This change depends on the following requirements from the **lookahead-pre-fetch-buffer** spec:

- Continuous Lookahead Generation (buffer management, queue lifecycle)
- Condition-Based Backpressure (producer/consumer coordination)
- Navigation Buffer Invalidation (buffer teardown on navigation)
- Player Start After Pre-Buffer (playback readiness gating)
- End of Book Termination (sentinel signalling)
- Sentence Failure Skip (error handling, consecutive failure policy)
- Temp File Management (temp directory lifecycle, cleanup)

## ADDED Requirements

### Requirement: GPU Concurrency Detection
The system **SHALL** auto-detect GPU availability for Kokoro TTS and default `max_concurrent` to 1 when a GPU is available.

#### Scenario: GPU detected, concurrency reduced
- Given: Kokoro TTS is the active model
- And: `torch.cuda.is_available()` returns `True`
- When: `ParallelTTSGen` is initialized
- Then: `max_concurrent` is set to 1 by default
- And: A log warning indicates concurrency was reduced for GPU Kokoro

#### Scenario: No GPU detected, configured concurrency used
- Given: Kokoro TTS is the active model
- And: `torch.cuda.is_available()` returns `False`
- When: `ParallelTTSGen` is initialized
- Then: `max_concurrent` uses the configured value (default 3)
- And: No GPU warning is logged

### Requirement: Batch Submission Interface
The system **SHALL** accept batches of sentences via `submit()` and return ordered results via `drain_next()`.

#### Scenario: Batch of sentences submitted and drained in order
- Given: A batch of 5 sentences has been submitted via `submit()`
- When: `dispatch()` is called and all workers complete
- Then: `drain_next()` returns each result in the exact order submitted
- And: The sequence `(c, p, s)` matches the submission order
- And: No result is returned before its predecessor

### Requirement: Graceful Error Isolation
The system **SHALL** isolate worker failures per sentence; one failure does not affect other concurrent generations.

#### Scenario: One worker fails, others succeed
- Given: 3 sentences are generating concurrently
- When: The second sentence fails (e.g., network error)
- Then: The first and third sentences complete and are buffered
- And: The second sentence is marked as `success=False` in the result
- And: The producer continues with the next batch after the failed sentence

### Requirement: Fallback to Sequential Generation
The system **SHALL** fall back to sequential generation when the parallel module is unavailable or `TTS_PARALLEL_ENABLED` is `False`.

#### Scenario: Import error falls back to sequential
- Given: The `lue/tts_parallel.py` module has an import error or is missing
- When: The producer attempts to initialize `ParallelTTSGen`
- Then: An error is logged indicating parallel TTS unavailable
- And: The system falls back to sequential generation (one sentence at a time)
- And: Playback continues without interruption

#### Scenario: Config disables parallel generation
- Given: `TTS_PARALLEL_ENABLED` is set to `False`
- When: The producer loop starts
- Then: The sequential generation code path is used
- And: No attempt is made to import or use `ParallelTTSGen`
- And: All existing single-sentence generation behavior is preserved

### Requirement: Concurrent Sentence Generation
The system **SHALL** generate up to `TTS_MAX_CONCURRENT` sentences simultaneously, using an `asyncio.Semaphore` to limit concurrent TTS workers.

#### Scenario: Three sentences generated concurrently
- Given: `TTS_MAX_CONCURRENT` is set to 3
- And: 3 sentences have been submitted for generation
- When: `dispatch()` is called
- Then: All 3 workers start concurrently within 100ms of each other
- And: The semaphore has 0 available slots
- And: Total wall-clock time is approximately the time of a single generation (not sequential)

#### Scenario: Concurrency limited when tasks exceed semaphore slots
- Given: `TTS_MAX_CONCURRENT` is set to 3
- And: 6 sentences have been submitted for generation
- When: `dispatch()` is called
- Then: Exactly 3 workers start immediately
- And: The remaining 3 workers wait for a semaphore slot
- And: When the first worker finishes, the fourth worker starts
- And: At no point do more than 3 workers run concurrently

### Requirement: Strict Output Ordering
The system **SHALL** enqueue completed results in an `_OrderedBuffer` and yield them strictly in chapter/paragraph/sentence order regardless of completion order.

#### Scenario: Out-of-order completions buffered until sequence is correct
- Given: 3 sentences at positions (1,1,1), (1,1,2), (1,1,3) are generating concurrently
- When: Sentence (1,1,3) completes first, then (1,1,2), then (1,1,1)
- Then: No result is returned after (1,1,3) completes
- And: No result is returned after (1,1,2) completes
- And: Only after (1,1,1) completes is the first result (1,1,1) returned
- And: (1,1,2) is returned immediately after (1,1,1)
- And: (1,1,3) is returned immediately after (1,1,2)

#### Scenario: In-order completions returned immediately
- Given: 3 sentences at positions (1,1,1), (1,1,2), (1,1,3) are generating concurrently
- When: Sentence (1,1,1) completes first, then (1,1,2), then (1,1,3)
- Then: (1,1,1) is returned immediately upon completion
- And: (1,1,2) is returned immediately upon completion
- And: (1,1,3) is returned immediately upon completion
- And: No buffering delay occurs for any result

### Requirement: Cache Integration in Workers
The system **SHALL** check the `TTSCache` inside each worker; a cache hit resolves instantly without consuming a semaphore slot.

#### Scenario: Cache hit bypasses TTS generation
- Given: `TTS_MAX_CONCURRENT` is set to 3
- And: 3 sentences are submitted, where the middle sentence is already in cache
- When: `dispatch()` is called
- Then: The cached sentence resolves instantly and is buffered immediately
- And: The semaphore slot for the cached sentence is released (not consumed)
- And: Only 2 semaphore slots are occupied at peak (for the uncached sentences)
- And: The cached sentence's result is available in the `_OrderedBuffer` immediately

#### Scenario: Cache miss proceeds to standard TTS generation
- Given: None of the submitted sentences are in the TTS cache
- When: `dispatch()` is called
- Then: Each worker performs the standard cache-miss flow (check cache, miss, generate TTS)
- And: All 3 workers proceed to TTS generation
- And: Semaphore slots are consumed for all 3 workers

### Requirement: Temp File Isolation in Workers
The system **SHALL** use session-sequence-based temp file paths per concurrent worker to prevent file collisions during parallel generation.

#### Scenario: Concurrent workers use distinct file paths
- Given: 3 sentences are generating concurrently via `_generate_one()`
- When: Each worker writes audio to a temp file
- Then: Each worker creates a file with a session-sequence-based name (e.g., `la_0001.mp3`, `la_0002.mp3`)
- And: `seq_num` is a monotonically increasing sequence counter per session
- And: No two workers write to the same file path
- And: All distinct paths reside in the configured temp directory
- And: A race condition is never possible during file write
