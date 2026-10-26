# Proposal: TTS Pipeline Integration and End-to-End Verification

## Intent
Integrate the three independently developed TTS pipeline components (TTS Cache, Lookahead Buffer, Parallel Generation) into a cohesive end-to-end system, resolving any interface mismatches, ensuring correct composition under all playback states, and verifying the full pipeline against the original goals of eliminating audio gaps between sentences.

## Scope
**In scope:**
- End-to-end integration of TTSCache, LookaheadBuffer, and ParallelTTSGen
- Unified initialization in reader._initialize_state() or play_from_current_position()
- Cross-component state synchronization: cache aware of lookahead temp files, parallel aware of cache
- Full lifecycle testing: startup, steady playback, navigation, pause/resume, speed changes, end-of-book
- Error propagation across components: cache miss → parallel worker failure → lookahead skip
- UI feedback: optional buffer-status indicator in TUI
- Performance benchmarking: measure gap duration, queue depth, cache hit ratio
- Configuration consolidation: all TTS pipeline settings in one config section

**Out of scope:**
- TTS engine changes (Edge, Kokoro, OpenAI unchanged)
- Player/ffplay changes
- Word highlighting changes
- Progress persistence changes
- UI redesign beyond buffer status indicator

## Dependencies
- **tts-cache-layer** (required): Provides TTSCache for content-hash audio caching.
- **lookahead-pre-fetch-buffer** (required): Provides LookaheadBuffer for sliding-window pre-fetch.
- **parallel-tts-generation** (required): Provides ParallelTTSGen for concurrent generation.

## Approach
Unify initialization: create a `TTSPipeline` class (or configure in reader) that wires TTSCache → ParallelTTSGen → LookaheadBuffer in the correct order. The LookaheadBuffer orchestrates: it accepts a ParallelTTSGen (with optional cache) and uses it for batch generation. On startup, create pipeline, start buffer producer, wait for pre-buffer threshold, start player. On navigation, destroy and recreate. End-to-end tests verify no gaps under load, correct ordering, correct cleanup.
