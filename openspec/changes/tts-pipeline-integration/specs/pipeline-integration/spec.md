# Delta for TTS Pipeline Integration

## DEPENDS ON
- tts-cache-layer: TTSCache component for content-hash audio caching
- lookahead-pre-fetch-buffer: LookaheadBuffer for sliding-window pre-fetch
- parallel-tts-generation: ParallelTTSGen for concurrent generation

## ADDED Requirements

### Requirement: Optional UI Buffer Status
The pipeline **SHALL** expose buffer depth for optional TUI display when `SHOW_BUFFER_STATUS` is enabled.

#### Scenario: Buffer fill displayed during playback
- Given: `SHOW_BUFFER_STATUS` is enabled in config
- When: playback is active
- Then: the TUI status area displays current and maximum queue depth (e.g., `Buf: 28/30`)
- And: the display updates within 1 second of a consumption or generation event
- And: the buffer depth read is O(1) and non-blocking

#### Scenario: No UI overhead when disabled
- Given: `SHOW_BUFFER_STATUS` is disabled or unset
- When: playback is active
- Then: no buffer status is computed or displayed
- And: zero performance overhead from buffer monitoring
- And: the TUI status area is unaffected

### Requirement: Unified Pipeline Initialization
The pipeline **SHALL** create `TTSCache`, `ParallelTTSGen`, and `LookaheadBuffer` in correct dependency order during `play_from_current_position()`, with graceful fallback when individual components are unavailable.

#### Scenario: All three components initialized when available
- Given: all TTS pipeline dependencies are installed and configuration is valid
- When: `play_from_current_position()` is called
- Then: `TTSCache` is initialized first (no dependencies)
- And: `ParallelTTSGen` is initialized second (depends on TTS engine)
- And: `LookaheadBuffer` is initialized third (depends on both upstream components)
- And: each component receives its correct configuration subset
- And: initialization order guarantees no component is referenced before creation

#### Scenario: Graceful fallback when individual component imports fail
- Given: one or more component dependencies are missing (e.g., TTS engine not installed)
- When: `play_from_current_position()` attempts to initialize the pipeline
- Then: the failing component import is caught and logged with a meaningful message
- And: a warning is emitted indicating which feature is degraded
- And: playback continues without the unavailable component (e.g., no caching, no pre-fetch)
- And: no traceback propagates to the caller

### Requirement: Pipeline Lifecycle Management
The pipeline **SHALL** manage state transitions through `IDLE → BUFFERING → PLAYING → PAUSED → FINISHED` states.

#### Scenario: Normal playback lifecycle
- Given: the pipeline is freshly initialized (IDLE)
- When: `start_producer()` is called
- Then: state transitions to BUFFERING
- And: when PREBUFFER_MIN_ITEMS is reached, state transitions to PLAYING
- And: when the final sentence is consumed, state transitions to FINISHED
- And: each state transition is logged with a timestamp

#### Scenario: Pause/resume preserves pipeline state
- Given: the pipeline is in PLAYING state with N sentences in the queue
- When: the user pauses
- Then: state transitions to PAUSED
- And: the producer continues to refill the queue (to support seek after resume)
- And: the player task is suspended but not cancelled
- When: the user resumes
- Then: state transitions to PLAYING
- And: playback resumes from the paused position without audible gap

### Requirement: Navigation Teardown and Recreation
The pipeline **SHALL** fully tear down (`buffer.stop → parallel.shutdown → temp cleanup`) and recreate on position jump.

#### Scenario: Pipeline fully destroyed and recreated on chapter jump
- Given: the pipeline is in PLAYING state with audio generated for positions 50-80
- When: the user jumps to chapter 3 (position 120)
- Then: the LookaheadBuffer is stopped (pending consumer tasks cancelled)
- And: ParallelTTSGen is shut down (pending worker tasks cancelled)
- And: all lookahead temp files are cleaned from LOOKAHEAD_TEMP_DIR
- And: a new pipeline is initialized for position 120
- And: playback resumes from the new position

#### Scenario: Temp files from old pipeline not leaked
- Given: the pipeline was torn down during a navigation event
- When: the teardown sequence completes
- Then: no `.mp3` temp files from the old pipeline remain in LOOKAHEAD_TEMP_DIR
- And: the directory is empty except for `.gitkeep` or similar sentinel files
- And: temp file cleanup does not remove files from TTS_CACHE_DIR

### Requirement: Cross-Layer Error Propagation
The pipeline **SHALL** propagate errors from worker through parallel engine to buffer, isolating failures per-sentence without stalling the entire pipeline.

#### Scenario: Single TTS failure skips one sentence while others continue
- Given: five sentences are being generated concurrently via ParallelTTSGen
- When: one worker raises an exception (e.g., network timeout for sentence N)
- Then: the error is caught at the ParallelTTSGen layer and converted to a failure result
- And: the buffer receives a skip signal for sentence N (not a crash)
- And: the remaining four sentences continue generation normally
- And: an error is logged with sentence position and failure reason
- And: pipeline state remains PLAYING

#### Scenario: Pipeline stops gracefully after consecutive error limit across all layers
- Given: five consecutive sentences have failed TTS generation across multiple workers
- When: the error counter reaches LOOKAHEAD_MAX_ERRORS
- Then: the buffer emits a warning with the failing position range
- And: the producer enqueues a sentinel and terminates
- And: no exception propagates across layer boundaries
- And: the player finishes cleanly after consuming remaining sentences

### Requirement: Temp File Directory Isolation
The pipeline **SHALL** separate cache files (persistent, in `TTS_CACHE_DIR`) from lookahead temp files (session-only, in `LOOKAHEAD_TEMP_DIR`) in distinct directories, with no cross-cleanup collisions.

#### Scenario: Cache files survive navigation
- Given: TTSCache has written cached audio files to TTS_CACHE_DIR
- When: the user navigates to a new chapter (triggering temp cleanup)
- Then: all files in TTS_CACHE_DIR are preserved
- And: cache hits continue to work after navigation
- And: no cache files are ever deleted during temp cleanup

#### Scenario: Lookahead temp files cleaned on navigation without affecting cache
- Given: the pipeline has generated lookahead temp files in LOOKAHEAD_TEMP_DIR
- When: navigation occurs and `_cleanup_temp_files()` runs
- Then: only files within LOOKAHEAD_TEMP_DIR are deleted
- And: the cleanup glob pattern (`LOOKAHEAD_TEMP_DIR / "la_*"`) isolates lookahead files
- And: no attempt is made to access or modify TTS_CACHE_DIR during cleanup

### Requirement: Configuration Consolidation
The pipeline **SHALL** group all TTS pipeline configuration keys in a single documented section of `config.py` with consistent naming.

#### Scenario: All pipeline config keys documented with purpose and valid ranges
- Given: the configuration module is loaded
- When: the "TTS Pipeline" section is read
- Then: the section contains all pipeline-relevant keys grouped together
- And: each key has a comment documenting its purpose
- And: each numeric key documents its valid range or allowed values
- And: key names follow a consistent pattern (`LOOKAHEAD_*`, `PREBUFFER_*`, `TTS_*`)
- And: the section is clearly delimited by section header comments

## MODIFIED Requirements

### Requirement: Playback Initiation (from add-audio-pre-buffering)
The `play_from_current_position()` function **MUST** incorporate `TTSCache` and `ParallelTTSGen` into the pre-buffer pipeline, replacing its current behavior of starting both tasks simultaneously.

#### Scenario: Cached sentences fill pre-buffer instantly
- Given: the first N sentences are already in the TTS cache
- When: playback starts
- Then: cached audio files are enqueued via the pipeline without regeneration
- And: the pre-buffer threshold is satisfied without waiting for network/TTS latency
- And: playback begins immediately

#### Scenario: Parallel generation accelerates pre-buffer fill
- Given: PREBUFFER_MIN_ITEMS sentences need to be generated and none are cached
- When: the pipeline starts
- Then: up to MAX_CONCURRENT_GENERATIONS sentences are generated concurrently via ParallelTTSGen
- And: sentences are enqueued in correct order as they complete
- And: the pre-buffer threshold is met in approximately MIN_BUFFER_ITEMS / MAX_CONCURRENT_GENERATIONS generation cycles

## REMOVED Requirements

None.
