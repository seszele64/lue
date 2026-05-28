"""Parallel TTS generation using asyncio workers gated by a Semaphore.

Provides ordered output guaranteed by an internal ``_OrderedBuffer``
that accepts out-of-order completions and returns them in strict
submission order. Integrates with :class:`TTSCache` for instant cache-hit
resolution and :class:`LookaheadBuffer` for deep pre-fetch pipelines.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from . import config, content_parser

log = logging.getLogger(__name__)

# --- Constants -----------------------------------------------------------

_DEFAULT_MAX_CONCURRENT = 3
_POP_TIMEOUT = 1.0


# ---------------------------------------------------------------------------
# Component 1: _GenerationResult
# ---------------------------------------------------------------------------


@dataclass
class _GenerationResult:
    """Result of generating audio for a single sentence.

    Attributes:
        sentence_idx: Tuple ``(chapter_idx, paragraph_idx, sentence_idx)``
            identifying the sentence position in the book.
        audio_path: Absolute path to the generated audio file on disk.
        duration: Length of the generated audio in seconds.
        timing_info: Dict with ``word_timings``, ``total_duration``, etc.
        success: ``True`` if generation completed without error.
    """

    sentence_idx: tuple
    audio_path: str
    duration: float
    timing_info: dict
    success: bool


# ---------------------------------------------------------------------------
# Component 2: _OrderedBuffer
# ---------------------------------------------------------------------------


class _OrderedBuffer:
    """In-memory buffer that returns results in strict submission order.

    Each result is stored under a monotonically incrementing submission
    index (not its ``(c, p, s)`` position).  The consumer drains results
    starting at index 0 and advancing by 1 each time :meth:`pop_next`
    succeeds, guaranteeing strict FIFO ordering regardless of the order
    in which workers complete.
    """

    def __init__(self) -> None:
        self._buffer: dict[int, _GenerationResult] = {}
        self._next_pos: int = 0
        self._submission_counter: int = 0
        self._pending_count: int = 0
        self._event: asyncio.Event = asyncio.Event()
        self._lock: asyncio.Lock = asyncio.Lock()

    # -- Public API -------------------------------------------------------

    async def submit(self, result: _GenerationResult) -> None:
        """Submit a completed result.

        The result is stored at the current ``_submission_counter`` index
        (which is then incremented).  If this index matches ``_next_pos``
        the internal event is signalled so that :meth:`pop_next` wakes up.
        """
        async with self._lock:
            idx = self._submission_counter
            self._submission_counter += 1
            self._buffer[idx] = result
            self._pending_count += 1

            if idx == self._next_pos:
                self._event.set()

    async def pop_next(self) -> Optional[_GenerationResult]:
        """Block until the next in-order result is available, then return it.

        Uses a loop with :func:`asyncio.wait_for` on the internal event
        so that it does not block indefinitely when the producer has
        stopped.  Returns ``None`` when the buffer is empty **and** no
        pending items remain.
        """
        while True:
            async with self._lock:
                if self._next_pos in self._buffer:
                    result = self._buffer.pop(self._next_pos)
                    self._next_pos += 1
                    self._pending_count -= 1
                    # If the next position is already buffered, keep the
                    # event set so the next call returns immediately.
                    if self._next_pos in self._buffer:
                        self._event.set()
                    else:
                        self._event.clear()
                    return result

                # No pending items at all → drained
                if self._pending_count == 0:
                    return None

            # Wait for the next completion
            try:
                await asyncio.wait_for(self._event.wait(), timeout=_POP_TIMEOUT)
            except asyncio.TimeoutError:
                # Re-check under lock; could have been signalled between
                # the lock release above and wait_for.
                continue

    def clear(self) -> None:
        """Reset the buffer.

        Discards all buffered results, resets counters, and creates a
        fresh event.  Intended for use on navigation.
        """
        self._buffer.clear()
        self._next_pos = 0
        self._submission_counter = 0
        self._pending_count = 0
        # Replace the event so any waiting pop_next gets a fresh start.
        self._event = asyncio.Event()

    @property
    def pending_count(self) -> int:
        """Number of results submitted but not yet drained by the consumer."""
        return self._pending_count


# ---------------------------------------------------------------------------
# GPU detection helper
# ---------------------------------------------------------------------------


def _detect_kokoro_gpu() -> bool:
    """Return ``True`` if a CUDA-capable GPU is available via PyTorch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Component 3: ParallelTTSGen
# ---------------------------------------------------------------------------


class ParallelTTSGen:
    """Orchestrates parallel TTS generation with ordered output delivery.

    Manages a pool of concurrent worker coroutines gated by an
    ``asyncio.Semaphore``.  Cache hits resolve instantly **without**
    consuming a semaphore slot.  Completed results are submitted to an
    internal :class:`_OrderedBuffer` which guarantees the consumer
    receives them in strict submission order.

    Parameters
    ----------
    reader:
        Lue Reader instance (provides chapter/paragraph/sentence access).
    tts_model:
        TTS model instance (e.g. :class:`~lue.tts.edge_tts.EdgeTTS`,
        :class:`~lue.tts.kokoro_tts.KokoroTTS`).
    max_concurrent:
        Maximum number of simultaneous TTS generation workers.
        Defaults to ``config.TTS_MAX_CONCURRENT``.
    cache:
        Optional :class:`~lue.tts_cache.TTSCache` instance.  When
        provided, cache hits bypass the semaphore entirely.
    """

    def __init__(
        self,
        reader,
        tts_model,
        max_concurrent: int | None = None,
        cache=None,
    ) -> None:
        self._reader = reader
        self._tts_model = tts_model

        # Determine effective concurrency limit ---------------------------
        if max_concurrent is not None:
            effective = max_concurrent
        else:
            effective = config.TTS_MAX_CONCURRENT

        # Kokoro GPU detection: reduce to 1 on GPU, unless user overrode.
        model_name = getattr(tts_model, "name", "")
        if model_name == "kokoro" and _detect_kokoro_gpu():
            # The default in config is 3.  If the user has explicitly set
            # a different value via env-var, respect their choice.
            if config.TTS_MAX_CONCURRENT == _DEFAULT_MAX_CONCURRENT:
                log.warning(
                    "Kokoro GPU detected, reducing max_concurrent from %d to 1",
                    effective,
                )
                effective = 1
            else:
                log.info(
                    "Kokoro GPU detected but TTS_MAX_CONCURRENT=%d "
                    "(env override) — using configured value",
                    config.TTS_MAX_CONCURRENT,
                )

        self._max_concurrent = effective
        self._semaphore = asyncio.Semaphore(effective)
        self._buffer = _OrderedBuffer()
        self._cache = cache

        # Pending sentences waiting to be dispatched.
        self._pending: list[tuple[tuple, str]] = []  # (sentence_idx, text)

        # Tracks spawned worker tasks for lifecycle management.
        self._worker_tasks: list[asyncio.Task] = []

        # State
        self._is_running = True
        self._seq_counter = 0
        self._submission_counter = 0
        self._error_count = 0

        log.info("ParallelTTSGen created (max_concurrent=%d)", effective)

    # -- Public API -------------------------------------------------------

    def submit(self, sentence_idx: tuple, text: str) -> None:
        """Queue a sentence for generation.

        Does **not** dispatch immediately; call :meth:`dispatch` to spawn
        worker tasks for all queued sentences.

        Parameters
        ----------
        sentence_idx:
            ``(chapter_idx, paragraph_idx, sentence_idx)`` tuple.
        text:
            The sentence text to generate audio for.
        """
        self._pending.append((sentence_idx, text))

    async def dispatch(self) -> None:
        """Spawn worker tasks for all currently pending sentences.

        Logs the number of sentences dispatched.  Workers are created via
        :func:`asyncio.create_task` and tracked in ``_worker_tasks``.
        """
        items = self._pending[:]
        self._pending.clear()

        for sentence_idx, text in items:
            task = asyncio.create_task(self._generate_one(sentence_idx, text))
            self._worker_tasks.append(task)

        log.info(
            "Dispatched %d workers (max_concurrent=%d)",
            len(items),
            self._max_concurrent,
        )

    async def drain_next(self) -> Optional[_GenerationResult]:
        """Block until the next in-order result is ready, then return it.

        Returns
        -------
        _GenerationResult or None
            The next buffered result, or ``None`` when the buffer is fully
            drained and no pending work remains.
        """
        return await self._buffer.pop_next()

    async def shutdown(self) -> None:
        """Cancel all running workers and reset the buffer.

        Safe to call multiple times.
        """
        self._is_running = False

        if self._worker_tasks:
            for t in self._worker_tasks:
                t.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()

        self._buffer.clear()
        self._pending.clear()
        log.info("ParallelTTSGen shut down")

    @property
    def pending_submitted(self) -> int:
        """Total number of unprocessed items across all stages.

        Includes pending (not yet dispatched), in-flight (worker tasks),
        and buffered (completed but not yet drained) items.
        """
        return (
            len(self._pending)
            + len(self._worker_tasks)
            + self._buffer.pending_count
        )

    # -- Internal helpers -------------------------------------------------

    async def _generate_one(self, sentence_idx: tuple, text: str) -> None:
        """Core worker: check cache, acquire semaphore, generate, submit.

        If a cache hit occurs the result is submitted immediately without
        consuming a semaphore slot.  Otherwise the semaphore is acquired
        for TTS generation.
        """
        sanitized = content_parser.sanitize_text_for_tts(text)

        # --- Cache check (BEFORE semaphore) ------------------------------
        if self._cache is not None:
            try:
                hit = self._cache.lookup(sanitized)
                if hit is not None:
                    result = _GenerationResult(
                        sentence_idx=sentence_idx,
                        audio_path=hit.audio_path,
                        duration=hit.duration,
                        timing_info=hit.timing_info,
                        success=True,
                    )
                    await self._buffer.submit(result)
                    return
            except Exception:
                log.warning("Cache lookup failed for '%s...'", text[:50])

        # --- Acquire semaphore for actual generation ---------------------
        async with self._semaphore:
            self._seq_counter += 1
            output_format = self._tts_model.output_format
            temp_path = os.path.join(
                config.LOOKAHEAD_TEMP_DIR,
                f"la_{self._seq_counter:04d}.{output_format}",
            )

            timing_info: dict = {}
            duration: float = 0.0
            success: bool = False

            try:
                # Attempt generation with timing
                tts = self._tts_model
                if hasattr(tts, "generate_audio_with_timing"):
                    try:
                        timing_info = await tts.generate_audio_with_timing(
                            sanitized, temp_path
                        )
                    except Exception as e:
                        log.error(
                            "TTS timing generation failed for '%s...': %s",
                            text[:50],
                            e,
                        )
                        await tts.generate_audio(sanitized, temp_path)
                else:
                    await tts.generate_audio(sanitized, temp_path)

                # Determine duration
                if timing_info:
                    duration = timing_info.get("total_duration", 0.0)
                if not duration or duration <= 0:
                    import subprocess

                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            temp_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                        )
                        stdout, _ = await proc.communicate()
                        if proc.returncode == 0:
                            duration = float(stdout.decode().strip())
                    except Exception:
                        pass

                # Fallback timing_info if still missing
                if not timing_info:
                    from .timing_calculator import process_tts_timing_data

                    timing_info = process_tts_timing_data(text, [], duration)

                # Store in cache
                if self._cache is not None and timing_info:
                    try:
                        self._cache.store(sanitized, temp_path, timing_info)
                    except Exception:
                        log.warning(
                            "Cache store failed for '%s...'", text[:50]
                        )

                # Submit success
                result = _GenerationResult(
                    sentence_idx=sentence_idx,
                    audio_path=temp_path,
                    duration=duration,
                    timing_info=timing_info,
                    success=True,
                )
                await self._buffer.submit(result)
                self._error_count = 0

            except asyncio.CancelledError:
                # Clean up temp file on cancellation
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise

            except Exception as e:
                self._error_count += 1
                log.error(
                    "TTS generation error (%d) for sentence %s "
                    "('%s...'): %s",
                    self._error_count,
                    sentence_idx,
                    text[:50],
                    e,
                )

                # Clean up temp file on error
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

                # Submit failed result so the consumer can handle the gap.
                result = _GenerationResult(
                    sentence_idx=sentence_idx,
                    audio_path="",
                    duration=0.0,
                    timing_info={},
                    success=False,
                )
                await self._buffer.submit(result)

    async def _generate_one_sequential(
        self, sentence_idx: tuple, text: str
    ) -> _GenerationResult:
        """Generate audio for a single sentence without semaphore gating.

        Returns the result directly instead of submitting to the buffer.
        Used as a fallback when parallel generation is disabled.

        Parameters
        ----------
        sentence_idx:
            ``(chapter_idx, paragraph_idx, sentence_idx)`` tuple.
        text:
            The sentence text to generate.

        Returns
        -------
        _GenerationResult
            The generation result (success or failure).
        """
        sanitized = content_parser.sanitize_text_for_tts(text)

        # --- Cache check -------------------------------------------------
        if self._cache is not None:
            try:
                hit = self._cache.lookup(sanitized)
                if hit is not None:
                    return _GenerationResult(
                        sentence_idx=sentence_idx,
                        audio_path=hit.audio_path,
                        duration=hit.duration,
                        timing_info=hit.timing_info,
                        success=True,
                    )
            except Exception:
                log.warning("Cache lookup failed for '%s...'", text[:50])

        # --- Generate audio ----------------------------------------------
        self._seq_counter += 1
        output_format = self._tts_model.output_format
        temp_path = os.path.join(
            config.LOOKAHEAD_TEMP_DIR,
            f"la_{self._seq_counter:04d}.{output_format}",
        )

        timing_info: dict = {}
        duration: float = 0.0

        try:
            tts = self._tts_model
            if hasattr(tts, "generate_audio_with_timing"):
                try:
                    timing_info = await tts.generate_audio_with_timing(
                        sanitized, temp_path
                    )
                except Exception as e:
                    log.error(
                        "TTS timing generation failed for '%s...': %s",
                        text[:50],
                        e,
                    )
                    await tts.generate_audio(sanitized, temp_path)
            else:
                await tts.generate_audio(sanitized, temp_path)

            # Duration
            if timing_info:
                duration = timing_info.get("total_duration", 0.0)
            if not duration or duration <= 0:
                import subprocess

                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        temp_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    stdout, _ = await proc.communicate()
                    if proc.returncode == 0:
                        duration = float(stdout.decode().strip())
                except Exception:
                    pass

            if not timing_info:
                from .timing_calculator import process_tts_timing_data

                timing_info = process_tts_timing_data(text, [], duration)

            # Cache store
            if self._cache is not None and timing_info:
                try:
                    self._cache.store(sanitized, temp_path, timing_info)
                except Exception:
                    log.warning("Cache store failed for '%s...'", text[:50])

            return _GenerationResult(
                sentence_idx=sentence_idx,
                audio_path=temp_path,
                duration=duration,
                timing_info=timing_info,
                success=True,
            )

        except asyncio.CancelledError:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        except Exception as e:
            log.error(
                "TTS generation error for sentence %s ('%s...'): %s",
                sentence_idx,
                text[:50],
                e,
            )
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return _GenerationResult(
                sentence_idx=sentence_idx,
                audio_path="",
                duration=0.0,
                timing_info={},
                success=False,
            )
