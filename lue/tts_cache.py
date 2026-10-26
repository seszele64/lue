"""TTS audio cache with SHA-256 content-hash keys and LRU eviction.

Provides a persistent disk cache for generated TTS audio files.
Each cache entry consists of an audio file (.mp3/.wav) and a JSON
sidecar storing timing_info for word-level highlighting support.
"""

import glob
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


@dataclass
class CacheHit:
    """Result of a successful cache lookup."""
    audio_path: str
    timing_info: dict
    duration: float


class TTSCache:
    """Persistent disk cache for TTS audio with LRU eviction.

    Each cache entry is keyed by SHA-256(engine:voice:text)[:16]
    and consists of:
      - {key}.{format}  — the audio file
      - {key}.json      — timing_info sidecar with text for collision detection

    A shared .cache_index.json tracks file sizes and last_accessed
    timestamps for LRU eviction decisions.
    """

    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int,
        engine_name: str,
        voice: str,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_bytes
        self.engine_name = engine_name
        self.voice = voice

        self._engine_dir = os.path.join(cache_dir, engine_name)
        self._index_path = os.path.join(cache_dir, ".cache_index.json")

        os.makedirs(self._engine_dir, exist_ok=True)
        self._index: Dict[str, dict] = self._load_index()

        # Track the most recently stored key to prevent evicting it
        self._last_stored_key: Optional[str] = None

    # --- Public API ---------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """Compute deterministic 16-char hex key from (engine, voice, text)."""
        input_str = f"{self.engine_name}:{self.voice}:{text}"
        return hashlib.sha256(input_str.encode("utf-8")).hexdigest()[:16]

    def lookup(self, text: str) -> Optional[CacheHit]:
        """Check cache for previously generated audio.

        Returns CacheHit on match, None on miss or error.
        Errors are logged and treated as cache misses.
        """
        key = self._cache_key(text)
        text_preview = text[:40]

        try:
            # Find the audio file (extension unknown from key alone)
            pattern = os.path.join(self._engine_dir, f"{key}.*")
            matches = glob.glob(pattern)
            audio_files = [
                m for m in matches if not m.endswith(".json")
            ]
            if not audio_files:
                log.debug("Cache MISS: '%s...'", text_preview)
                return None

            audio_path = audio_files[0]
            sidecar_path = os.path.join(self._engine_dir, f"{key}.json")

            if not os.path.exists(sidecar_path):
                log.debug("Cache MISS (no sidecar): '%s...'", text_preview)
                return None

            # Read sidecar
            with open(sidecar_path, "r", encoding="utf-8") as fh:
                sidecar = json.load(fh)

            # Collision detection: verify stored text matches requested text
            stored_text = sidecar.get("text", "")
            if stored_text != text:
                log.warning(
                    "Cache HASH COLLISION: '%s...' vs stored '%s...'",
                    text_preview,
                    stored_text[:40],
                )
                return None

            # Validate audio file exists and is non-empty
            if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
                log.warning(
                    "Cache CORRUPTED entry: '%s...' (file missing or empty)",
                    text_preview,
                )
                return None

            # Update access time in index
            if key in self._index:
                self._index[key]["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self._save_index()

            timing_info = sidecar.get("timing_info", {})
            duration = timing_info.get("total_duration", 0.0)

            log.debug("Cache HIT: '%s...'", text_preview)
            return CacheHit(
                audio_path=audio_path,
                timing_info=timing_info,
                duration=duration,
            )

        except Exception as e:
            log.warning("Cache ERROR in lookup: %s", e)
            return None

    def store(
        self,
        text: str,
        source_audio_path: str,
        timing_info: dict,
    ) -> bool:
        """Store generated audio in the cache.

        Copies audio file and writes timing_info sidecar atomically.
        Triggers LRU eviction if needed.

        Returns True on success, False on error.
        """
        key = self._cache_key(text)
        text_preview = text[:40]

        try:
            # Determine extension from source file
            ext = os.path.splitext(source_audio_path)[1]  # e.g. ".mp3"
            if not ext:
                ext = ".mp3"

            audio_dest = os.path.join(self._engine_dir, f"{key}{ext}")
            sidecar_dest = os.path.join(self._engine_dir, f"{key}.json")

            # Atomic audio write
            audio_tmp = audio_dest + ".tmp"
            shutil.copy2(source_audio_path, audio_tmp)
            os.replace(audio_tmp, audio_dest)

            # Atomic sidecar write
            sidecar_data: Dict[str, Any] = {
                "text": text,
                "timing_info": timing_info,
                "engine": self.engine_name,
                "voice": self.voice,
            }
            sidecar_tmp = sidecar_dest + ".tmp"
            with open(sidecar_tmp, "w", encoding="utf-8") as fh:
                json.dump(sidecar_data, fh, indent=2)
            os.replace(sidecar_tmp, sidecar_dest)

            # Update index
            file_size = os.path.getsize(audio_dest)
            self._index[key] = {
                "size": file_size,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
            }

            # Track so eviction doesn't remove just-stored entry
            self._last_stored_key = key

            # Evict if needed, then save index
            self._evict_if_needed()
            self._save_index()

            log.debug(
                "Cache STORE: '%s...' (%d bytes)",
                text_preview,
                file_size,
            )
            return True

        except Exception as e:
            log.warning("Cache ERROR in store: %s", e)
            return False

    def size_bytes(self) -> int:
        """Total cache size in bytes (from index)."""
        return sum(
            entry.get("size", 0) for entry in self._index.values()
        )

    def clear(self) -> None:
        """Remove all cached files and reset the index."""
        try:
            # Delete all audio and sidecar files in engine directory
            if os.path.isdir(self._engine_dir):
                for filename in os.listdir(self._engine_dir):
                    filepath = os.path.join(self._engine_dir, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except OSError as e:
                        log.warning(
                            "Cache ERROR clearing file %s: %s",
                            filename,
                            e,
                        )

            # Delete index file
            if os.path.exists(self._index_path):
                try:
                    os.remove(self._index_path)
                except OSError as e:
                    log.warning("Cache ERROR removing index: %s", e)

            self._index = {}
            log.info("Cache cleared (%s)", self._engine_dir)

        except Exception as e:
            log.warning("Cache ERROR in clear: %s", e)

    # --- Internal methods ---------------------------------------------

    def _load_index(self) -> Dict[str, dict]:
        """Load cache index from disk.

        Returns empty dict if file is missing or corrupted.
        Logs warning and rebuilds from disk scan if corrupted.
        """
        if not os.path.exists(self._index_path):
            log.debug("Cache INDEX not found, starting fresh")
            return {}

        try:
            with open(self._index_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data).__name__}")

            # Validate entries: remove entries pointing to missing files
            valid_entries: Dict[str, dict] = {}
            for key, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                pattern = os.path.join(self._engine_dir, f"{key}.*")
                matches = glob.glob(pattern)
                audio_exists = any(
                    m for m in matches if not m.endswith(".json")
                )
                if audio_exists:
                    valid_entries[key] = entry
                else:
                    log.debug(
                        "Cache INDEX: removing stale entry %s (no audio file)",
                        key,
                    )

            if len(valid_entries) < len(data):
                log.info(
                    "Cache INDEX pruned: %d stale entries removed",
                    len(data) - len(valid_entries),
                )

            return valid_entries

        except (json.JSONDecodeError, ValueError) as e:
            log.warning(
                "Cache INDEX corrupted, rebuilding from disk: %s", e
            )
            return {}

        except Exception as e:
            log.warning("Cache ERROR loading index: %s", e)
            return {}

    def _save_index(self) -> None:
        """Write index to disk atomically."""
        try:
            index_tmp = self._index_path + ".tmp"
            with open(index_tmp, "w", encoding="utf-8") as fh:
                json.dump(self._index, fh, indent=2)
            os.replace(index_tmp, self._index_path)
        except Exception as e:
            log.warning("Cache ERROR saving index: %s", e)

    def _evict_if_needed(self) -> None:
        """Remove oldest entries until total size is under max_size_bytes."""
        try:
            total = self.size_bytes()
            if total <= self.max_size_bytes:
                return

            freed_bytes = 0
            evicted_count = 0

            # Sort by last_accessed ascending (oldest first)
            sorted_entries = sorted(
                self._index.items(),
                key=lambda item: item[1].get("last_accessed", ""),
            )

            for key, entry in sorted_entries:
                # Never evict the entry we just stored
                if key == self._last_stored_key:
                    continue

                # Delete audio file
                pattern = os.path.join(self._engine_dir, f"{key}.*")
                matches = glob.glob(pattern)
                for filepath in matches:
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except OSError:
                        pass

                entry_size = entry.get("size", 0)
                freed_bytes += entry_size
                total -= entry_size
                del self._index[key]
                evicted_count += 1

                if total <= self.max_size_bytes:
                    break

            if evicted_count > 0:
                freed_mb = freed_bytes / (1024 * 1024)
                log.info(
                    "Cache EVICT: %d entries removed (%.1f MB freed)",
                    evicted_count,
                    freed_mb,
                )

        except Exception as e:
            log.warning("Cache ERROR in eviction: %s", e)
