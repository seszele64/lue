# Delta for Lookahead Buffer

## ADDED Requirements

### Requirement: Continuous Lookahead Generation
The system **SHALL** maintain a queue of up to LOOKAHEAD_SENTENCES pre-generated audio items, continuously refilling as the player consumes them, eliminating buffer underrun during uninterrupted playback.

#### Scenario: Queue stays full during steady playback
- Given: the player is consuming sentences at a normal rate and the producer can keep up
- When: playback is in steady state
- Then: the queue depth stays between (LOOKAHEAD_SENTENCES - 3) and LOOKAHEAD_SENTENCES
- And: the player never waits longer than 100ms for the next audio item
- And: no audible gap occurs between consecutive sentences

#### Scenario: Refill resumes when queue drops below target
- Given: the producer is blocked (queue full) and the player consumes one item
- When: the player calls `task_done()`
- Then: the producer is notified via the condition variable
- And: the producer resumes generating the next sentence
- And: the queue is refilled back to LOOKAHEAD_SENTENCES

### Requirement: Condition-Based Backpressure
The system **SHALL** use `asyncio.Condition` to suspend the producer when the queue reaches capacity and resume it when the player consumes an item, ensuring zero CPU waste during idle periods.

#### Scenario: Producer blocks on full queue
- Given: the queue contains LOOKAHEAD_SENTENCES items and the producer has just attempted to enqueue another
- When: the queue is full
- Then: the producer awaits on the condition variable (suspends without polling)
- And: CPU usage drops to near-zero during the wait
- And: no `asyncio.sleep()` or busy-wait loop is involved

#### Scenario: Player consumption wakes producer
- Given: the producer is suspended on the condition (queue was full)
- When: the player dequeues an item and calls `task_done()`
- Then: the condition is notified
- And: the producer wakes within one event loop iteration
- And: the producer resumes generating the next sentence

### Requirement: Navigation Buffer Invalidation
The system **SHALL** destroy the current lookahead buffer and create a new one at the target position whenever the reader position jumps due to navigation (next chapter, previous sentence, direct position click, or similar).

#### Scenario: Chapter jump clears old buffer
- Given: a lookahead buffer has 20+ items generated from Chapter 3
- When: the user jumps to Chapter 5
- Then: `buffer.stop()` is called immediately
- And: the queue is drained completely
- And: all temp files in the lookahead/ directory are deleted
- And: a new LookaheadBuffer is created starting at Chapter 5, sentence 0

#### Scenario: New buffer generates from correct position
- Given: the reader position has changed to a new chapter and sentence
- When: a fresh LookaheadBuffer is created at the new position
- Then: the first generated sentence corresponds to the new reader position exactly
- And: no sentences from the previous position are played
- And: the sequence counter for temp filenames resets

### Requirement: Player Start After Pre-Buffer
The system **SHALL NOT** start the player until either PREBUFFER_MIN_ITEMS sentences are queued OR PREBUFFER_MIN_SECONDS of audio duration is buffered (whichever comes first).

#### Scenario: Player waits for minimum items
- Given: `play_from_current_position()` has been called and the producer has started
- When: the queue has fewer than PREBUFFER_MIN_ITEMS items
- Then: the player task is not yet spawned
- And: `play_from_current_position()` awaits until PREBUFFER_MIN_ITEMS is reached
- And: a timeout of 30 seconds prevents indefinite blocking

#### Scenario: Player starts immediately when cache provides items
- Given: all PREBUFFER_MIN_ITEMS sentences are already generated (e.g., from a previous session or cache)
- When: the producer quickly fills the initial items
- Then: the player spawns without additional delay beyond the generation itself
- And: the first sentence starts playing within 200ms of the player task starting

#### Scenario: Seconds-based threshold triggers player start
- Given: a document with very long sentences where PREBUFFER_MIN_ITEMS=3 would mean 30+ seconds
- When: the producer generates enough audio that total buffered duration >= PREBUFFER_MIN_SECONDS before reaching PREBUFFER_MIN_ITEMS
- Then: the player starts when the seconds threshold is met, even if item count is below minimum
- And: PREBUFFER_MIN_SECONDS acts as a safety net to prevent excessive pre-buffer delay on long sentences

### Requirement: End of Book Termination
The system **SHALL** detect when the reader has reached the end of the book and signal the player to terminate playback gracefully.

#### Scenario: Producer stops at book end
- Given: the producer is generating sentences and `_advance_lookahead()` returns `None`
- When: no more sentences exist in the book
- Then: `is_at_end` is set to `True`
- And: a `None` sentinel is placed on the queue
- And: the producer coroutine exits cleanly without errors

#### Scenario: Player finishes after last sentence
- Given: the player is consuming items and encounters a `None` sentinel
- When: `buffer.get()` returns `None`
- Then: the player stops waiting for new items
- And: any remaining active playback tasks complete
- And: `playback_finished_event` is set
- And: the player loop exits

### Requirement: Sentence Failure Skip
The system **SHALL** skip sentences whose TTS generation fails, log the error, and continue with the next sentence. After LOOKAHEAD_MAX_ERRORS consecutive failures, the system **SHALL** stop the producer gracefully.

#### Scenario: Single failure skipped, playback continues
- Given: a sentence fails TTS generation (e.g., network timeout or API error)
- When: `_handle_error()` is called
- Then: the error is logged with the sentence position and error details
- And: `error_count` is incremented by 1
- And: `_advance_lookahead()` is called to skip the failed sentence
- And: the producer continues generating the next sentence
- And: playback continues without interruption (no gap item on queue)

#### Scenario: Consecutive failures stop producer
- Given: 4 sentences have already failed consecutively
- When: the 5th consecutive sentence also fails
- Then: `error_count` reaches LOOKAHEAD_MAX_ERRORS
- And: a critical error is logged indicating the producer is stopping
- And: a `None` sentinel is placed on the queue
- And: the producer coroutine exits
- And: the player finishes gracefully after consuming remaining items

### Requirement: Temp File Management
The system **SHALL** manage temporary audio files using unique sequence-numbered filenames in a dedicated directory, with cleanup on navigation and application exit.

#### Scenario: Temp files created with unique names
- Given: a lookahead buffer session has started
- When: a new sentence is generated
- Then: the audio file is saved as `la_XXXX.ext` in the `LOOKAHEAD_TEMP_DIR`
- And: `XXXX` is a monotonically increasing zero-padded sequence number (0001, 0002, ...)
- And: the sequence number is unique within the session

#### Scenario: Temp files deleted on navigation
- Given: a lookahead buffer session has generated multiple temp files
- When: `buffer.stop()` is called due to navigation
- Then: all `la_*` files in `LOOKAHEAD_TEMP_DIR` matching the current session are deleted
- And: no file deletion errors interrupt the navigation flow
- And: the directory itself is not deleted (only the files)

#### Scenario: Temp files cleaned on app exit
- Given: the application is shutting down while a lookahead buffer exists
- When: cleanup routines run during shutdown
- Then: all `la_*` files for this session are deleted
- And: the lookahead/ directory is empty after exit

### Requirement: Environment Variable Configuration
The system **SHALL** read lookahead buffer parameters from environment variables with sensible defaults, allowing users to tune buffer depth without code changes.

#### Scenario: Env var overrides default
- Given: `LUE_LOOKAHEAD_SENTENCES=15` is set in the environment
- When: the configuration module loads
- Then: `LOOKAHEAD_SENTENCES` is set to 15
- And: the lookahead buffer uses this value for its max queue size

#### Scenario: Value clamped to valid range
- Given: `LUE_LOOKAHEAD_SENTENCES=200` is set in the environment (above maximum)
- When: the configuration module loads
- Then: `LOOKAHEAD_SENTENCES` is clamped to 100 (the upper bound)
- And: a warning is logged indicating the value was clamped

#### Scenario: Value below minimum clamped
- Given: `LUE_LOOKAHEAD_SENTENCES=2` is set in the environment (below minimum)
- When: the configuration module loads
- Then: `LOOKAHEAD_SENTENCES` is clamped to 5 (the lower bound)
- And: a warning is logged indicating the value was clamped
