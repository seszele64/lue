# Delta for TTS Cache

## ADDED Requirements

### Requirement: Content-Hash TTS Audio Cache
The system **SHALL** hash (engine, voice, sanitized_text) via SHA-256 to produce a deterministic 16-character hex key used for cache lookups, enabling redundant API calls to be skipped on repeated text.

#### Scenario: same text produces same key
- Given: a sentence "Chapter One" with engine "edge" and voice "en-US-JennyNeural"
- When: the cache key is computed from (engine, voice, sanitized_text)
- Then: the same sentence always produces the same 16-character hex digest

#### Scenario: different engine produces different key
- Given: the same sentence "Chapter One" is processed with engine "edge" and then engine "kokoro"
- When: the cache key is computed each time
- Then: the keys are different because the engine name is included in the hash input

### Requirement: Cache Hit — Transparent Reuse
The system **SHALL** serve cached audio on key match, copying the cached file to the output path and returning cached timing_info without calling the TTS engine.

#### Scenario: repeated sentence served from cache
- Given: sentence "Hello World" was previously generated and cached
- When: the same sanitized text, engine, and voice are requested again
- Then: the cached audio file is copied to the output path
- And: no TTS API call or model inference occurs
- And: playback proceeds with zero generation latency

#### Scenario: cache hit preserves word-level timing data
- Given: a cached entry contains word-level timing in its sidecar
- When: the cache entry is served
- Then: the cached timing_info dict is returned intact, including word_timings and word_mapping
- And: word highlighting continues to function correctly from cached data

### Requirement: Cache Miss — Generate and Store
The system **SHALL** generate audio via the TTS engine on cache miss, then store both the audio file and a JSON timing sidecar in the cache directory with atomic writes.

#### Scenario: new sentence generated and cached
- Given: a sentence has never been generated before
- When: the TTS engine produces audio and timing data
- Then: the audio file is written to the cache with a unique key-based filename
- And: a JSON sidecar with full timing_info is written alongside the audio
- And: both writes are performed atomically (temp file + os.replace)

### Requirement: LRU Eviction
The system **SHALL** remove least recently accessed entries when total cache size exceeds TTS_CACHE_MAX_SIZE_BYTES, using per-entry last_accessed timestamps from the cache index.

#### Scenario: cache exceeds limit triggers eviction
- Given: the cache size is near the threshold (e.g., 95 MB of a 100 MB limit)
- When: a new file is stored that pushes the total over the limit
- Then: the least recently accessed entries are deleted until total is below the limit
- And: the entry just stored is never evicted in the same operation

#### Scenario: eviction preserves files referenced by active audio queue
- Given: a cached entry is currently in the active audio queue (scheduled for playback)
- When: eviction is triggered
- Then: the referenced entry is excluded from eviction
- And: playback of that entry proceeds normally

### Requirement: Atomic Writes
The system **SHALL** use a temp file in the same directory followed by os.replace for all cache writes to prevent file corruption from interrupted writes or process crashes.

#### Scenario: write completes atomically
- Given: a cache write operation is in progress
- When: the audio data is fully written to a temporary file
- Then: os.replace atomically moves the temp file to the final cache path
- And: the cache index is also written atomically
- And: if the process crashes during the temp write, only a temp file is lost (not the existing cache entry)

### Requirement: Graceful Degradation
The system **SHALL** fall through to normal generation on any cache error, ensuring the user never experiences a playback failure due to a cache problem.

#### Scenario: disk full falls back to generation
- Given: the cache directory is on a full disk
- When: cache.store() raises a write error
- Then: the error is logged
- And: the sentence is generated and played normally (without caching)
- And: playback continues unaffected

#### Scenario: corrupted cache file regenerated
- Given: a cached audio file exists but is corrupted (zero-sized, partial, or unplayable)
- When: the cache entry is looked up and fails validation
- Then: the corrupted entry is logged and treated as a cache miss
- And: the sentence is freshly generated via the TTS engine
- And: the new result overwrites the corrupted cache entry

### Requirement: Cache Configuration
The system **SHALL** accept TTS_CACHE_ENABLED, TTS_CACHE_DIR, and TTS_CACHE_MAX_SIZE_MB from configuration, with the cache directory auto-created on startup.

#### Scenario: cache disabled skips all operations
- Given: TTS_CACHE_ENABLED is False
- When: the producer loop processes sentences
- Then: no cache lookup, store, or eviction operations occur
- And: all sentences are generated via the TTS engine every time

#### Scenario: cache directory auto-created
- Given: the configured TTS_CACHE_DIR does not exist
- When: the application starts
- Then: the directory is created (including parent directories)
- And: the application continues with a fresh empty cache
