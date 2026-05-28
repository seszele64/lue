"""Lookahead pre-fetch buffer for continuous TTS audio generation.

Manages a deep pipeline of pre-generated audio sentences using an
asyncio.Queue with asyncio.Condition-based backpressure between
the producer (generator) and player (consumer).
"""

import asyncio
import glob
import logging
import os
import re
from typing import Optional, Tuple

from . import config, content_parser

log = logging.getLogger(__name__)

# Reuse these patterns from audio.py for fragment merging
ABBREVIATION_PATTERN = (
    r"\b(Mr|Mrs|Ms|Dr|Prof|Rev|Hon|Jr|Sr|Cpl|Sgt|Gen|Col|Capt|Lt|Pvt|"
    r"vs|viz|Co|Inc|Ltd|Corp|St|Ave|Blvd)\."
)

# Type alias for position tuples
Position = Tuple[int, int, int]  # (chapter_idx, paragraph_idx, sentence_idx)
QueueItem = Tuple[str, int, int, int, float, dict]  # (path, c, p, s, duration, timing_info)


class LookaheadBuffer:
    """Deep pre-fetch buffer for TTS audio sentences.

    Maintains a queue of up to ``target_sentences`` pre-generated audio
    items. The producer fills the queue continuously, pausing via
    asyncio.Condition when full. The player consumes and notifies the
    producer to refill. Temp audio files use unique sequence-numbered
    names in a dedicated directory, cleaned on navigation or exit.
    """

    def __init__(
        self,
        reader,                       # Lue reader instance
        target_sentences: int = 30,
        min_start_items: int = 3,
        max_errors: int = 5,
    ) -> None:
        self._reader = reader
        self._target = target_sentences
        self._min_start = min_start_items
        self._max_errors = max_errors

        # Core async primitives
        self._queue: asyncio.Queue[Optional[QueueItem]] = asyncio.Queue(
            maxsize=target_sentences
        )
        self._condition = asyncio.Condition()

        # State tracking
        self._is_running = False
        self._is_at_end = False
        self._error_count = 0
        self._seq_counter = 0
        self._buffered_duration = 0.0

        # Start position cloned from reader
        self._lookahead_pos: Position = (
            reader.chapter_idx,
            reader.paragraph_idx,
            reader.sentence_idx,
        )

        # Parallel TTS generator (lazy init in _refill)
        self._parallel_gen = None

        # Task handle
        self._producer_task: Optional[asyncio.Task] = None

        log.debug(
            "LookaheadBuffer created: target=%d, min_start=%d, pos=(%d,%d,%d)",
            target_sentences,
            min_start_items,
            *self._lookahead_pos,
        )

    # --- Public API ---------------------------------------------------

    @property
    def qsize(self) -> int:
        """Current number of ready items in the queue."""
        return self._queue.qsize()

    def start_producer(self) -> asyncio.Task:
        """Launch the producer coroutine as a background task.

        Returns the created asyncio.Task.
        """
        if self._producer_task is not None and not self._producer_task.done():
            log.warning("Producer already running, skipping start")
            return self._producer_task

        self._is_running = True
        self._producer_task = asyncio.create_task(self._refill())
        log.debug("LookaheadBuffer producer started")
        return self._producer_task

    async def stop(self) -> None:
        """Stop the producer, drain the queue, and clean up temp files."""
        log.debug("LookaheadBuffer stopping...")
        self._is_running = False

        # Wake any waiting producer
        async with self._condition:
            self._condition.notify_all()

        # Cancel producer task if still running
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            try:
                await asyncio.wait_for(self._producer_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._producer_task = None

        # Shut down parallel generator if it was created
        if self._parallel_gen is not None:
            try:
                await self._parallel_gen.shutdown()
            except Exception:
                pass
            self._parallel_gen = None

        # Drain queue
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                self._queue.task_done()
                # If it's a real item (not sentinel), clean up its temp file
                if item is not None:
                    temp_path = item[0]
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
            except asyncio.QueueEmpty:
                break

        # Reset buffered duration
        self._buffered_duration = 0.0

        # Clean up any remaining temp files in the lookahead directory
        self._cleanup_temp_files()

        log.debug("LookaheadBuffer stopped")

    async def get(self) -> Optional[QueueItem]:
        """Get the next audio item from the queue.

        Returns None as a sentinel when the buffer is stopped or at
        end-of-book. Raises asyncio.CancelledError if cancelled.
        """
        if not self._is_running and self._queue.empty():
            return None

        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            return item
        except asyncio.TimeoutError:
            if not self._is_running:
                return None
            # Still running but queue empty — retry
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                return item
            except asyncio.TimeoutError:
                return None

    def task_done(self) -> None:
        """Mark the current queue item as processed and signal the producer.

        This is a synchronous method — both queue.task_done() and
        condition.notify() must be called within the condition context
        that is acquired by the player.
        """
        try:
            self._queue.task_done()
        except ValueError:
            pass  # too many task_done calls

    async def signal_refill(self) -> None:
        """Notify the producer that the player consumed an item.

        Must be called by the player after task_done() to wake the
        producer when the queue drops below target.
        """
        async with self._condition:
            self._condition.notify()

    # --- Producer internals -------------------------------------------

    async def _refill(self) -> None:
        """Main producer loop with optional parallel batch generation."""

        # --- Lazy init parallel generator ---
        parallel_gen = None
        max_concurrent = 1
        if config.TTS_PARALLEL_ENABLED:
            try:
                from .tts_parallel import ParallelTTSGen

                parallel_gen = ParallelTTSGen(
                    reader=self._reader,
                    tts_model=self._reader.tts_model,
                    cache=getattr(self._reader, "tts_cache", None),
                )
                max_concurrent = parallel_gen._max_concurrent
                log.info(
                    "ParallelTTSGen initialized (max_concurrent=%d)", max_concurrent
                )
            except ImportError as e:
                log.warning(
                    "Parallel TTS import failed: %s — falling back to sequential", e
                )
            except Exception as e:
                log.warning(
                    "Parallel TTS init failed: %s — falling back to sequential", e
                )

        # Store on self so stop() can shut it down
        self._parallel_gen = parallel_gen

        log.debug(
            "LookaheadBuffer refill loop starting (parallel=%s)",
            parallel_gen is not None,
        )

        is_parallel = parallel_gen is not None

        try:
            while self._is_running:
                # If queue is at target, wait for consumer to free slots
                if self._qsize_safe() >= self._target:
                    async with self._condition:
                        await self._condition.wait()
                    continue

                # Check if we've reached the end
                if self._is_at_end:
                    await self._put_sentinel()
                    break

                if is_parallel:
                    # === PARALLEL PATH ===
                    success = await self._refill_batch_parallel(
                        parallel_gen, max_concurrent
                    )
                    if not success and self._error_count >= self._max_errors:
                        log.critical(
                            "LookaheadBuffer: %d consecutive errors, stopping",
                            self._error_count,
                        )
                        await self._put_sentinel()
                        break
                else:
                    # === SEQUENTIAL PATH (existing) ===
                    success = await self._generate_and_enqueue()
                    if not success and self._error_count >= self._max_errors:
                        log.critical(
                            "LookaheadBuffer: %d consecutive errors, stopping",
                            self._error_count,
                        )
                        await self._put_sentinel()
                        break

        except asyncio.CancelledError:
            log.debug("LookaheadBuffer refill cancelled")
            if parallel_gen is not None:
                try:
                    await parallel_gen.shutdown()
                except Exception:
                    pass
        except Exception as e:
            log.error("LookaheadBuffer refill error: %s", e, exc_info=True)
        finally:
            log.debug("LookaheadBuffer refill loop exited")

    def _extract_sentence_text(
        self, pos: Position
    ) -> Tuple[Optional[str], Optional[Position], bool]:
        """Extract text at the given position, merging abbreviation fragments.

        Returns:
            (text, next_position, is_merged):
            - text: The sentence text (possibly merged with next), or None if out of bounds
            - next_position: The position AFTER the extracted sentence(s), or None at end-of-book
            - is_merged: True if this text merged two sentences (fragment + continuation)
        """
        c, p, s = pos

        try:
            sentences = content_parser.split_into_sentences(
                self._reader.chapters[c][p]
            )
            text = sentences[s]
        except IndexError:
            return None, None, False

        if not text or not text.strip():
            next_pos = self._advance_lookahead(from_pos=pos)
            return "", next_pos, False

        # Fragment merging (same logic as audio.py producer loop)
        merged = False
        is_abbrev_fragment = re.fullmatch(ABBREVIATION_PATTERN, text.strip())
        if is_abbrev_fragment and s + 1 < len(sentences):
            text += " " + sentences[s + 1]
            merged = True

        # Advance from the given pos: single step normally, double step if merged
        next_pos = self._advance_lookahead(from_pos=pos)
        if merged and next_pos is not None:
            next_pos = self._advance_lookahead(from_pos=next_pos)

        return text, next_pos, merged

    def _advance_past(self, sentence_pos: Position, merged: bool) -> None:
        """Update self._lookahead_pos past the given sentence(s).

        If merged is True, advances past 2 sentences (the fragment + its continuation).
        """
        next_pos = self._advance_lookahead()
        if merged and next_pos is not None:
            next_pos = self._advance_lookahead()

        if next_pos is None:
            self._is_at_end = True
        else:
            self._lookahead_pos = next_pos

    async def _refill_batch_parallel(
        self, parallel_gen, max_concurrent: int
    ) -> bool:
        """Batch-submit sentences to ParallelTTSGen, dispatch, and drain results.

        Returns True if at least one sentence was successfully enqueued,
        False if all failed.
        """
        remaining = self._target - self._qsize_safe()
        batch_size = min(remaining, max_concurrent * 2)

        # --- Collect batch: extract sentence texts with fragment merging ---
        batch: list[tuple] = []  # [(sentence_pos, text, is_merged), ...]
        pos = self._lookahead_pos
        for _ in range(batch_size):
            text, next_pos, merged = self._extract_sentence_text(pos)
            if text is None:
                # End of book
                self._is_at_end = True
                break
            batch.append((pos, text, merged))
            if next_pos is None:
                self._is_at_end = True
                break
            pos = next_pos

        if not batch:
            if self._is_at_end:
                await self._put_sentinel()
            return False

        # --- Submit batch ---
        for sentence_pos, text, _merged in batch:
            parallel_gen.submit(sentence_pos, text)

        # --- Dispatch ---
        await parallel_gen.dispatch()

        # --- Drain results in order ---
        any_success = False
        for idx, (sentence_pos, _, merged) in enumerate(batch):
            try:
                result = await parallel_gen.drain_next()
            except Exception as e:
                log.error("drain_next failed: %s", e)
                self._error_count += 1
                # Advance position
                self._advance_past(sentence_pos, merged)
                continue

            if result is None:
                # Buffer drained unexpectedly — means shutdown or end
                break

            if result.success:
                # Enqueue the result item
                c, p, s = result.sentence_idx
                item = (
                    result.audio_path,
                    c,
                    p,
                    s,
                    result.duration,
                    result.timing_info,
                )
                try:
                    await asyncio.wait_for(
                        self._queue.put(item),
                        timeout=1.0,
                    )
                    self._buffered_duration += result.duration
                    self._error_count = 0
                    any_success = True
                except asyncio.TimeoutError:
                    log.error(
                        "Queue put timeout for sentence %s", result.sentence_idx
                    )
                    self._error_count += 1
            else:
                # Failed result — skip this sentence
                self._error_count += 1
                log.warning(
                    "Skipping failed sentence %s (error %d/%d)",
                    result.sentence_idx,
                    self._error_count,
                    self._max_errors,
                )

            # Advance position (accounting for merged fragments)
            self._advance_past(sentence_pos, merged)

        return any_success

    async def _generate_and_enqueue(self) -> bool:
        """Generate one sentence at self._lookahead_pos and enqueue it.

        Returns True on success, False on failure.
        """
        c, p, s = self._lookahead_pos

        # Get text with fragment merging via helper
        text, next_pos, merged = self._extract_sentence_text(self._lookahead_pos)

        # Out of bounds — advance
        if text is None:
            if next_pos is None:
                self._is_at_end = True
            return True

        # Empty/blank text — advance
        if not text or not text.strip():
            if next_pos is None:
                self._is_at_end = True
            else:
                self._lookahead_pos = next_pos
            return True

        original_text = text

        # Sanitize for TTS
        sanitized_text = content_parser.sanitize_text_for_tts(original_text)

        # Temp file path
        self._seq_counter += 1
        output_format = self._reader.tts_model.output_format
        temp_path = os.path.join(
            config.LOOKAHEAD_TEMP_DIR,
            f"la_{self._seq_counter:04d}.{output_format}",
        )

        timing_info = None
        duration = 0.0
        cache_hit = False

        try:
            # --- Cache lookup ---
            if (
                config.TTS_CACHE_ENABLED
                and hasattr(self._reader, "tts_cache")
                and self._reader.tts_cache is not None
            ):
                try:
                    hit = self._reader.tts_cache.lookup(sanitized_text)
                    if hit is not None:
                        import shutil
                        shutil.copy2(hit.audio_path, temp_path)
                        timing_info = hit.timing_info
                        duration = hit.duration
                        cache_hit = True
                except Exception:
                    pass

            # --- TTS generation (on miss) ---
            if not cache_hit:
                tts = self._reader.tts_model
                if hasattr(tts, "generate_audio_with_timing"):
                    try:
                        timing_info = await tts.generate_audio_with_timing(
                            sanitized_text, temp_path
                        )
                    except Exception as e:
                        log.error(
                            "TTS timing generation failed for '%s...': %s",
                            original_text[:50],
                            e,
                        )
                        await tts.generate_audio(sanitized_text, temp_path)
                else:
                    await tts.generate_audio(sanitized_text, temp_path)

                # Get duration
                if timing_info is not None:
                    duration = timing_info.get("total_duration", 0)
                    if not duration or duration <= 0:
                        duration = await self._get_audio_duration(temp_path)
                else:
                    duration = await self._get_audio_duration(temp_path)

                # Store in cache
                if (
                    config.TTS_CACHE_ENABLED
                    and hasattr(self._reader, "tts_cache")
                    and self._reader.tts_cache is not None
                    and timing_info is not None
                ):
                    try:
                        self._reader.tts_cache.store(
                            sanitized_text, temp_path, timing_info
                        )
                    except Exception:
                        pass

            # Fallback timing_info if still None
            if timing_info is None:
                from .timing_calculator import process_tts_timing_data

                timing_info = process_tts_timing_data(original_text, [], duration)

            # Enqueue
            await asyncio.wait_for(
                self._queue.put(
                    (temp_path, c, p, s, duration, timing_info)
                ),
                timeout=1.0,
            )

            # Track buffered duration
            self._buffered_duration += duration

            # Reset error count on success
            self._error_count = 0

            # Advance position using next_pos from _extract_sentence_text
            if next_pos is None:
                self._is_at_end = True
            else:
                self._lookahead_pos = next_pos

            return True

        except asyncio.CancelledError:
            # Clean up temp file on cancellation
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

        except Exception as e:
            self._handle_error(e, original_text)
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            # Advance past failed sentence using next_pos from _extract_sentence_text
            if next_pos is None:
                self._is_at_end = True
            else:
                self._lookahead_pos = next_pos
            return False

    def _advance_lookahead(
        self, from_pos: Optional[Position] = None
    ) -> Optional[Position]:
        """Advance a position by one sentence.

        Args:
            from_pos: The position to advance from. If None, uses
                       ``self._lookahead_pos``.

        Returns the next position tuple, or None if at end of book.
        """
        current = self._lookahead_pos if from_pos is None else from_pos
        result = self._reader._advance_position(current, wrap=False)
        if result == (0, 0, 0):
            # wrap=False should return None when at end, but _advance_position
            # returns (0,0,0) on wrap. Check if we're truly at end.
            return None
        return result

    def _handle_error(self, error: Exception, text: str) -> None:
        """Log the error, increment error counter, advance position."""
        self._error_count += 1
        log.error(
            "LookaheadBuffer TTS error (%d/%d) for '%s...': %s",
            self._error_count,
            self._max_errors,
            text[:50],
            error,
        )

    async def _put_sentinel(self) -> None:
        """Place a None sentinel on the queue to signal end-of-stream."""
        try:
            await asyncio.wait_for(self._queue.put(None), timeout=1.0)
            log.debug("LookaheadBuffer: sentinel queued")
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    def _qsize_safe(self) -> int:
        """Get queue size without raising exceptions."""
        try:
            return self._queue.qsize()
        except Exception:
            return 0

    async def _get_audio_duration(self, file_path: str) -> float:
        """Get audio duration via ffprobe (fallback)."""
        import subprocess

        try:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                return 0.0
            return float(stdout.decode().strip())
        except Exception:
            return 0.0

    def _cleanup_temp_files(self) -> None:
        """Delete all lookahead temp files for this session."""
        pattern = os.path.join(config.LOOKAHEAD_TEMP_DIR, "la_*")
        for filepath in glob.glob(pattern):
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except OSError:
                pass
