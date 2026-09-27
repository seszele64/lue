# Delta for Audio Pipeline

## ADDED Requirements

### Requirement: Pre-Buffer Audio Before Playback Start
The audio pipeline **SHALL** generate and buffer a minimum duration of audio before starting playback, preventing startup silence while TTS synthesis completes.

#### Scenario: Normal playback start with adequate pre-buffer
- Given: a document is loaded and TTS model is initialized
- When: playback starts (user presses play or auto-starts on load)
- Then: the producer generates sentences until at least MIN_BUFFER_SECONDS of audio duration are queued
- And: the player task is started only after the pre-buffer threshold is met
- And: the first sentence plays within 200ms of player task spawning (no startup gap)

#### Scenario: Pre-buffer threshold is met immediately (audio already cached)
- Given: the first N sentences are already in the TTS cache
- When: playback starts
- Then: cached audio files are enqueued without regeneration
- And: the pre-buffer threshold is satisfied without waiting for network/TTS latency
- And: playback begins immediately

#### Scenario: Very short document where total audio < buffer threshold
- Given: a document with total audio duration less than MIN_BUFFER_SECONDS
- When: playback starts
- Then: the producer generates all sentences
- And: the player starts after all sentences are generated
- And: no artificial delay is introduced beyond what generation requires

### Requirement: Increased Queue Capacity
The audio queue **SHALL** support at least 8 sentences of look-ahead to provide deeper buffering against TTS generation latency spikes.

#### Scenario: Queue fills to new capacity during continuous playback
- Given: the producer generates sentences faster than the player consumes them
- When: playback is active and the queue has 8 items
- Then: the producer blocks (awaits sleep) until a slot opens
- And: no queue overflow occurs
- And: all generated audio files are correctly played in order

#### Scenario: Queue drains partially during slow generation
- Given: TTS generation is slow (3-5s per sentence)
- When: the player consumes sentences faster than the producer generates them
- Then: the queue level decreases but does not reach zero for at least 4 consecutive sentences
- And: no audible gap occurs as long as queue contains at least 1 item

### Requirement: Parallel TTS Sentence Generation
The producer **SHALL** generate up to MAX_CONCURRENT_GENERATIONS sentences in parallel using asyncio.Semaphore, while preserving strict sentence ordering in the output queue.

#### Scenario: Three sentences generated concurrently
- Given: three sentences are queued for generation
- When: the producer loop processes them
- Then: up to MAX_CONCURRENT_GENERATIONS TTS requests are in-flight simultaneously
- And: sentences are placed in the audio queue in correct sequential order regardless of completion order
- And: each concurrent task uses a unique AUDIO_BUFFERS file slot

#### Scenario: Generation fails for one concurrent task
- Given: two sentences are generating concurrently and one fails with a network error
- When: the failed sentence's generation task raises an exception
- Then: the failed sentence is skipped with an error logged
- And: the successful sentence is enqueued at its correct sequence position
- And: the producer continues with the next sentence

### Requirement: Content-Hash TTS Audio Cache
The system **SHALL** cache generated TTS audio files by content hash to avoid redundant API calls for repeated text.

#### Scenario: Repeated sentence is served from cache
- Given: sentence "Chapter One" was previously generated and cached
- When: the same sentence is encountered again later (e.g., chapter heading)
- Then: the cached audio file is used instead of calling the TTS API
- And: the cache hit is transparent to the player (same file path, duration, timing)
- And: no TTS API cost is incurred for the cached sentence

#### Scenario: Cache miss generates and stores new audio
- Given: a sentence has never been generated before
- When: the producer encounters it
- Then: the TTS API is called to generate audio
- And: the generated audio file is stored in the cache directory
- And: the cache key is the SHA256 hash of (sanitized_text + voice + speed)

#### Scenario: Cache directory reaches size limit
- Given: the cache directory exceeds MAX_CACHE_SIZE_MB
- When: a new file needs to be cached
- Then: the least recently accessed files are evicted until usage is below the limit
- And: eviction removes only cache files, never files currently referenced by the audio queue
- And: cache eviction logs a warning if it cannot free enough space

## MODIFIED Requirements

### Requirement: Playback Initiation
The `play_from_current_position()` function **MUST** coordinate producer pre-buffering before spawning the player task, replacing its current behavior of starting both tasks simultaneously.

#### Scenario: Playback starts after pre-buffer
- Given: `play_from_current_position()` is called
- When: the producer has generated at least MIN_BUFFER_SECONDS of audio
- Then: the player task is spawned
- And: the first sentence plays without a startup gap
- And: the existing task cancellation and cleanup logic is preserved

#### Scenario: Navigation triggers audio restart
- Given: the user navigates to a new sentence position (click, keyboard, or chapter jump)
- When: `_restart_audio_after_navigation()` is called
- Then: the pre-buffer process runs before the new player starts
- And: the old producer/player tasks are fully cancelled and cleaned up before pre-buffering begins

## REMOVED Requirements

None.
