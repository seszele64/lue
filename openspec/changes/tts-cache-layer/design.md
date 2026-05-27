# Design: TTS Audio Result Cache

## Technical Approach

The TTS audio cache is implemented as a standalone `TTSCache` class in a new `lue/tts_cache.py` module. The cache sits transparently in the producer loop between sentence extraction and TTS engine invocation — neither `TTSBase` nor individual engine implementations require modification.

1. **Cache Key**: SHA-256 of `f"{engine}:{voice}:{sanitized_text}"`, truncated to 16 hex characters. This produces a deterministic key before any audio generation occurs, unlike content-addressed schemes that require the generated file first.
2. **Storage Layout**: Each cached entry consists of an audio file (`<key>.<ext>`) and a JSON sidecar (`<key>.json`) storing the full `timing_info` dict (total_duration, word_timings, word_mapping). A shared `.cache_index.json` tracks all entries with file_size, last_accessed timestamp, and access count for LRU eviction.
3. **LRU Eviction**: Triggered synchronously on every `store()` call. The index is sorted by `last_accessed` (ascending), and oldest entries are removed until total size is under `max_size_bytes`. Entries currently referenced by the active audio queue are excluded from eviction.
4. **Atomic Writes**: All file writes (audio, sidecar, index) use a temp file in the same directory followed by `os.replace()` to prevent corruption from interrupted writes or crashes.

## Architecture Decisions

### Decision: Cache Placement in Producer Loop, Not TTSBase
- Pros: Engines remain completely unaware of caching; cache has access to both audio output and timing_info (already produced), no API changes to TTSBase
- Cons: Each producer loop implementation must separately integrate cache calls (only one in practice)
- Verdict: Place in producer loop — the cache needs both audio_path and timing_info dict, which only the producer loop has at the same time

### Decision: SHA-256 Key (Deterministic) Instead of Content-Addressable
- Pros: Key is known before generation begins, can check cache before calling TTS API, avoids unnecessary API calls for cache misses that are really hits
- Cons: Requires consistent normalization of engine name, voice, and text; hash collisions are theoretically possible (16-char truncation — 2^64 space — negligible risk)
- Verdict: SHA-256 deterministic key — the whole purpose is to skip the API call on hit, which requires the key to be computable before generation

### Decision: Synchronous LRU Eviction on Store Instead of Background Thread
- Pros: No background thread management, no need for file locking, simple and fast (eviction takes <10ms for typical cache sizes)
- Cons: Slightly delays cache store operation if many entries need eviction
- Verdict: Synchronous on store — fast enough for typical usage (file deletion + JSON update under 10ms)

### Decision: JSON Sidecar With Text Verification Instead of Content-Addressable Check
- Pros: Full timing_info preserved alongside audio, enables word-level highlighting from cache; sidecar text field enables collision detection; sidecar can be read independently of audio file
- Cons: Additional disk I/O for JSON read on every cache hit; sidecar could become out of sync with audio file
- Verdict: JSON sidecar — the timing_info is essential for word highlighting and must be preserved; text verification catches the vanishingly rare hash collision

### Decision: Atomic Writes via temp file + os.replace
- Pros: Prevents corruption from interrupted writes or process crashes; POSIX-standard atomic operation; no partial files visible to readers
- Cons: Temporary doubling of disk space during write (briefly, until os.replace); requires all writes to use the same filesystem
- Verdict: os.replace — the simplest correct approach on POSIX systems; disk space is not a concern for small audio files

## Data Flow

```
Cache Lookup (on every sentence before generation):
  Producer loop has: sanitized_text, engine_name, voice
       ↓
  Compute cache key: SHA-256(f"{engine_name}:{voice}:{sanitized_text}")[:16]
       ↓
  cache.lookup(sanitized_text)
       ↓
  Check for cache key directory file: {cache_dir}/{engine_name}/{key}.{format}
       ↓
  File exists?
      YES → Read .json sidecar
           ↓
         Verify sidecar["text"] == sanitized_text ?
             YES → HIT: Copy audio to buffer file with shutil.copy2()
                   → Load timing_info from sidecar
                   → Update last_accessed in cache_index.json
                   → Return (buffer_path, timing_info, duration)
             NO  → Collision: log warning, treat as MISS
      NO  → MISS: Return None

Cache Store (after successful TTS generation):
  Producer loop has: audio_path, timing_info dict
       ↓
  cache.store(sanitized_text, audio_path, timing_info)
       ↓
  Compute cache key (same formula)
       ↓
  Ensure cache directory exists: {cache_dir}/{engine_name}/
       ↓
  Write audio: copy audio_path to temp file → os.replace to {key}.{format}
       ↓
  Write timing: json.dumps to temp file → os.replace to {key}.json
       ↓
  Update cache_index.json: add/update entry with file_size and last_accessed
       ↓
  evict_if_needed():
    While total_size > max_size_bytes:
      Find entry with oldest last_accessed in index
      Delete audio file: {key}.{format}
      Delete sidecar: {key}.json
      Remove entry from index
      Update total_size
       ↓
  Save updated cache_index.json (atomic: temp → os.replace)

Error Paths:
  cache.lookup() raises exception (disk error, corrupted file)
       ↓
  Log warning, return None → treat as MISS → generate normally
       ↓
  cache.store() raises exception (disk full, permission error)
       ↓
  Log warning, skip cache → sentence plays normally, just not cached
       ↓
  Corrupted index: .cache_index.json missing or invalid JSON
       ↓
  On _load_index(): log warning, return empty dict → rebuild from disk scan
```

## Cache Index Structure

The `.cache_index.json` file maintains metadata for LRU eviction:

```json
{
  "edge/a1b2c3d4e5f6g7h8": {
    "size": 45230,
    "last_accessed": "2026-05-27T10:30:00Z"
  },
  "kokoro/x9y8z7w6v5u4t3s2": {
    "size": 128400,
    "last_accessed": "2026-05-27T10:29:00Z"
  }
}
```

On startup: `_load_index()` reads the file and validates it. If missing or corrupted, a warning is logged and an empty index is used. The index is rebuilt by scanning disk files on the next eviction cycle.

On lookup: `last_accessed` is updated for the accessed entry and the index is saved.

On eviction: entries are sorted by `last_accessed` (ascending), and the oldest are removed until `total_size <= max_size_bytes`.

## File Changes

- **`lue/tts_cache.py`** — New file: `TTSCache` class, `CacheHit` dataclass, index management helpers
- **`lue/audio.py`** — Modify `_producer_loop` to check `cache.lookup()` before `generate_audio_with_timing()` and call `cache.store()` after generation; modify `play_from_current_position()` to instantiate `TTSCache`
- **`lue/config.py`** — Add `TTS_CACHE_ENABLED`, `TTS_CACHE_DIR`, `TTS_CACHE_MAX_SIZE_MB`, `TTS_CACHE_MAX_SIZE_BYTES`; auto-create cache directory on module load
