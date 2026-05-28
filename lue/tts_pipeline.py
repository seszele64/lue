"""TTS Pipeline — unified lifecycle for cache, parallel generation, and lookahead.

Wraps :class:`TTSCache`, :class:`ParallelTTSGen`, and :class:`LookaheadBuffer`
in a single :class:`TTSPipeline` with correct dependency ordering and graceful
degradation when individual components are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .tts_cache import TTSCache
    from .tts_parallel import ParallelTTSGen
    from .lookahead_buffer import LookaheadBuffer

log = logging.getLogger(__name__)


class PipelineState(Enum):
    """Lifecycle states of the TTS pipeline."""
    IDLE = auto()        # Not started yet
    BUFFERING = auto()   # Producer active, queue below pre-buffer threshold
    PLAYING = auto()     # Pre-buffer met, player consuming
    PAUSED = auto()      # Playback paused, pipeline preserved
    FINISHED = auto()    # End-of-book reached, playback complete
    ERROR = auto()       # Unrecoverable error, pipeline stopped


@dataclass
class TTSPipeline:
    """Unified TTS pipeline with component dependency ordering.

    Initializes :class:`TTSCache` → :class:`ParallelTTSGen` →
    :class:`LookaheadBuffer` in correct order, supporting graceful
    degradation when individual components are unavailable.

    Parameters
    ----------
    reader:
        Lue Reader instance.
    cache:
        Optional pre-initialized TTSCache. If None and TTS_CACHE_ENABLED,
        creates one automatically.
    parallel_gen:
        Optional pre-initialized ParallelTTSGen. If None, creates one
        (with cache if available).
    buffer:
        Optional pre-initialized LookaheadBuffer. If None, creates one.
    """

    reader: object
    cache: Optional[TTSCache] = None
    parallel_gen: Optional[ParallelTTSGen] = None
    buffer: Optional[LookaheadBuffer] = None
    state: PipelineState = field(default=PipelineState.IDLE)

    # Component availability flags (set during __post_init__)
    cache_available: bool = field(default=False)
    parallel_available: bool = field(default=False)

    def __post_init__(self) -> None:
        """Initialize pipeline components in dependency order.

        Order: TTSCache → ParallelTTSGen → LookaheadBuffer
        Each step has graceful fallback if unavailable.
        """
        # --- Step 1: TTSCache (no dependencies) ---
        if self.cache is None and config.TTS_CACHE_ENABLED:
            try:
                from .tts_cache import TTSCache

                engine_name = getattr(self.reader.tts_model, 'name', 'unknown')
                voice = getattr(self.reader.tts_model, 'voice', 'default')
                self.cache = TTSCache(
                    cache_dir=config.TTS_CACHE_DIR,
                    max_size_bytes=config.TTS_CACHE_MAX_SIZE_BYTES,
                    engine_name=engine_name,
                    voice=voice,
                )
                self.cache_available = True
                log.info("TTSCache initialized (engine=%s, voice=%s)", engine_name, voice)
            except ImportError as e:
                log.warning("TTSCache import failed: %s — caching disabled", e)
            except Exception as e:
                log.warning("TTSCache init failed: %s — caching disabled", e)
        elif self.cache is not None:
            self.cache_available = True

        # --- Step 2: ParallelTTSGen (depends on cache) ---
        if self.parallel_gen is None:
            try:
                from .tts_parallel import ParallelTTSGen

                self.parallel_gen = ParallelTTSGen(
                    reader=self.reader,
                    tts_model=self.reader.tts_model,
                    cache=self.cache if self.cache_available else None,
                )
                self.parallel_available = True
                log.info(
                    "ParallelTTSGen initialized (max_concurrent=%d)",
                    self.parallel_gen._max_concurrent,
                )
            except ImportError as e:
                log.warning(
                    "ParallelTTSGen import failed: %s — falling back to sequential", e
                )
            except Exception as e:
                log.warning(
                    "ParallelTTSGen init failed: %s — falling back to sequential", e
                )
        else:
            self.parallel_available = True

        # --- Step 3: LookaheadBuffer (depends on both) ---
        if self.buffer is None:
            from .lookahead_buffer import LookaheadBuffer

            self.buffer = LookaheadBuffer(
                reader=self.reader,
                target_sentences=config.LOOKAHEAD_SENTENCES,
                min_start_items=config.PREBUFFER_MIN_ITEMS,
                max_errors=config.LOOKAHEAD_MAX_ERRORS,
            )

        log.info(
            "TTSPipeline created: cache=%s, parallel=%s, state=%s",
            self.cache_available,
            self.parallel_available,
            self.state.name,
        )

    # --- Public API ---

    def start(self) -> object:  # returns asyncio.Task
        """Start the pipeline producer.

        Transitions state to BUFFERING and launches the LookaheadBuffer
        producer task.

        Returns:
            asyncio.Task: The producer task.
        """
        if self.buffer is None:
            raise RuntimeError("Cannot start: no buffer")

        self.state = PipelineState.BUFFERING
        task = self.buffer.start_producer()
        log.info("TTSPipeline started (state=%s)", self.state.name)
        return task

    async def stop(self) -> None:
        """Stop the pipeline: buffer → parallel → cache cleanup.

        Transitions to IDLE. Safe to call multiple times.
        """
        if self.state == PipelineState.IDLE:
            return

        log.info("TTSPipeline stopping (was %s)...", self.state.name)

        # Stop buffer (cancels producer, drains queue, cleans temp files)
        if self.buffer is not None:
            try:
                await self.buffer.stop()
            except Exception:
                pass

        # Shut down parallel generator (cancels workers)
        if self.parallel_gen is not None and self.parallel_available:
            try:
                await self.parallel_gen.shutdown()
            except Exception:
                pass

        self.state = PipelineState.IDLE
        log.info("TTSPipeline stopped (state=%s)", self.state.name)

    def set_playing(self) -> None:
        """Transition to PLAYING state (pre-buffer threshold met)."""
        self.state = PipelineState.PLAYING

    def set_paused(self) -> None:
        """Transition to PAUSED state."""
        self.state = PipelineState.PAUSED

    def set_finished(self) -> None:
        """Transition to FINISHED state."""
        self.state = PipelineState.FINISHED

    def set_error(self) -> None:
        """Transition to ERROR state."""
        self.state = PipelineState.ERROR
        log.error("TTSPipeline entered ERROR state")

    @property
    def buffer_depth(self) -> int:
        """Current number of buffered (ready) items. O(1), non-blocking."""
        if self.buffer is None:
            return 0
        return self.buffer.qsize

    @property
    def buffer_target(self) -> int:
        """Target buffer size."""
        if self.buffer is None:
            return 0
        return self.buffer._target

    @property
    def is_parallel(self) -> bool:
        """Whether parallel generation is active."""
        return self.parallel_available and self.parallel_gen is not None


def create_pipeline(reader) -> TTSPipeline:
    """Factory function: create and return a fully wired TTSPipeline.

    Convenience wrapper that constructs a pipeline with all three
    components in correct order with graceful degradation.

    Args:
        reader: Lue Reader instance.

    Returns:
        TTSPipeline: Fully initialized pipeline (may have some components
        unavailable due to import/init failures, but buffer is always present).
    """
    return TTSPipeline(reader=reader)
