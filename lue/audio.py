"""Audio pipeline for Lue.

Two paths coexist (spec-v3 §9.5):

* **Persistent sink** (default): a single long-lived ``ffplay`` sink fed raw
  ``s16le/48000/2`` PCM over stdin by :func:`lue.audio_sink.writer_task`, with
  an authoritative :class:`~lue.audio_sink.SinkClock` and a
  :class:`~lue.audio_sink.SegmentTable` driving the highlight.
* **Legacy per-file**: the original ``_producer_loop`` / ``_player_loop`` that
  spawns one ``ffplay`` per sentence.  Preserved behind
  ``config.USE_PERSISTENT_SINK`` / ``--legacy-audio`` for one release.

Both paths share :func:`stop_and_clear_audio` (the 10-step teardown, §5.2) and
the same reader state names.
"""

import asyncio
import os
import re
import shutil
import logging

from . import config, content_parser, audio_sink

# This pattern is used to both clean text for TTS and detect sentence fragments.
ABBREVIATION_PATTERN = r'\b(Mr|Mrs|Ms|Dr|Prof|Rev|Hon|Jr|Sr|Cpl|Sgt|Gen|Col|Capt|Lt|Pvt|vs|viz|Co|Inc|Ltd|Corp|St|Ave|Blvd)\.'
INITIAL_PATTERN = r'\b([A-Z])\.(?=\s[A-Z])'


# Word mapping functionality moved to timing_calculator.py
# Import it here for backward compatibility
from .timing_calculator import create_word_mapping as _create_word_mapping


# ── Playback facade ────────────────────────────────────────────────────────


class AudioPlaybackController:
    """Owns the generation token and the two audio queues.

    The reader exposes ``audio_generation`` / ``audio_queue`` / ``decode_queue``
    as delegating properties so existing call sites keep working unchanged.
    """

    def __init__(self):
        self.generation = 0
        # Legacy per-file queue (``MAX_QUEUE_SIZE`` items).
        self.audio_queue = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)
        # Persistent-sink decode-job queue (``MAX_QUEUED_SENTENCES`` items).
        self.decode_queue = asyncio.Queue(maxsize=config.MAX_QUEUED_SENTENCES)


def clean_tts_text(text: str) -> str:
    """
    Removes periods from specific English abbreviations and single initials
    to prevent unnatural pauses in TTS engines. Also removes loose punctuation
    marks that are not connected to any word.
    """
    # Remove periods from abbreviations and initials
    text = re.sub(ABBREVIATION_PATTERN, r'\1', text)
    text = re.sub(INITIAL_PATTERN, r'\1 ', text)

    # Remove loose punctuation marks that are standalone (not connected to words)
    # This pattern matches punctuation that is surrounded by whitespace or at string boundaries
    text = re.sub(r'(?:^|\s)[.,:;!?]+(?=\s|$)', ' ', text)

    # Remove standalone dashes that are followed by quotation marks
    # This prevents TTS engines from reading "-" as "dash" in cases like: -"
    text = re.sub(r'(?:^|\s)-(?=")', ' ', text)

    # Clean up any extra whitespace that might result from removing punctuation
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── Teardown helpers (spec-v3 §5.2) ────────────────────────────────────────


async def _cancel_task(task):
    """Cancel ``task`` and await it with a bounded timeout (tolerate anything)."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=config.TASK_CANCEL_TIMEOUT_S)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    except Exception:  # noqa: BLE001 - teardown must never raise
        pass


async def _drain_queue(queue):
    if queue is None:
        return
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        try:
            queue.task_done()
        except ValueError:
            pass


def _remove_session_dir(reader):
    """Best-effort delete of the generation's session temp dir (§9.4)."""
    path = getattr(reader, '_session_dir', None)
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass
    reader._session_dir = None


async def stop_and_clear_audio(reader):
    """Single-source-of-truth teardown (spec-v3 §5.2).

    Runs the ten steps in order.  Shared by ``_restart_audio_after_navigation``,
    ``_handle_pause_toggle``, ``_switch_book`` and ``_shutdown``.
    """
    # 1. Invalidate every in-flight loop.
    reader.audio_generation += 1

    # 2. Cancel + await the producer.
    await _cancel_task(getattr(reader, 'producer_task', None))
    reader.producer_task = None

    # 3. Cancel + await the writer (may be inside drain(); CancelledError is ok).
    await _cancel_task(getattr(reader, 'writer_task', None))
    reader.writer_task = None

    # 4. Cancel + await decoder tasks (normally empty on the sink path), then
    #    make sure any inline decoder process is gone.
    for task in list(getattr(reader, 'decoder_tasks', None) or []):
        await _cancel_task(task)
    reader.decoder_tasks = []
    decoder_proc = getattr(reader, '_current_decoder', None)
    if decoder_proc is not None:
        try:
            if decoder_proc.returncode is None:
                decoder_proc.kill()
        except (ProcessLookupError, AttributeError):
            pass
        try:
            await asyncio.wait_for(decoder_proc.wait(), timeout=config.DECODE_TERM_TIMEOUT_S)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        reader._current_decoder = None

    # 5-6. Close stdin (delivers EOF) then terminate/kill the sink.
    sink = getattr(reader, 'sink', None)
    if sink is not None:
        try:
            sink.close_stdin()
        except Exception:  # noqa: BLE001
            pass
        try:
            await sink.terminate()
        except Exception:  # noqa: BLE001
            pass
        try:
            if sink.proc is not None and sink.proc.returncode is None:
                await sink.kill()
        except Exception:  # noqa: BLE001
            pass

    # 7. Cancel + await the supervisor.
    await _cancel_task(getattr(reader, 'player_task', None))
    reader.player_task = None

    # 8. Drain both queues and any legacy processes/tasks.
    await _drain_queue(getattr(reader, 'decode_queue', None))
    await _drain_queue(getattr(reader, 'audio_queue', None))

    for process in list(getattr(reader, 'playback_processes', None) or []):
        try:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=config.SINK_KILL_TIMEOUT_S)
                except asyncio.TimeoutError:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=config.SINK_KILL_TIMEOUT_S)
        except (ProcessLookupError, AttributeError, asyncio.TimeoutError):
            pass
    reader.playback_processes = []

    for task in list(getattr(reader, 'active_playback_tasks', None) or []):
        await _cancel_task(task)
    reader.active_playback_tasks = []

    # 9. Reset state for the next generation.
    reader.sink = None
    reader.sink_clock = None
    table = getattr(reader, 'segment_table', None)
    if table is not None:
        table.clear()
    reader.playback_finished_event.clear()
    reader._sink_eof_intended = False
    reader._current_decoder = None

    # 10. Delete the session temp dir.
    _remove_session_dir(reader)


async def get_audio_duration(file_path):
    """Get the duration of an audio file (logging / validation only, §7.4)."""
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    stdout, _ = await process.communicate()
    if process.returncode != 0: return None
    try: return float(stdout.decode().strip())
    except (ValueError, TypeError): return None


# ── Start / restart ────────────────────────────────────────────────────────


async def play_from_current_position(reader):
    """Start audio playback, routing to the active pipeline (§9.5)."""
    if reader.is_paused or not reader.running or not reader.tts_model:
        return
    if config.USE_PERSISTENT_SINK:
        await _start_persistent(reader)
    else:
        await _start_legacy(reader)


def _notify_console(reader, message: str) -> None:
    """Best-effort user-visible warning; never raises."""
    console = getattr(reader, "console", None)
    if console is None:
        return
    try:
        console.print(message)
    except Exception:  # noqa: BLE001 - console must never break the pipeline
        pass


async def _start_persistent(reader):
    """Spawn sink + writer + producer + supervisor for one generation."""
    # Cancel any lingering tasks from a previous (already-invalidated) generation.
    await _cancel_task(reader.producer_task)
    await _cancel_task(reader.writer_task)
    await _cancel_task(reader.player_task)
    reader.producer_task = None
    reader.writer_task = None
    reader.player_task = None

    reader.playback_finished_event.clear()
    reader._sink_eof_intended = False

    gen = reader.audio_generation
    reader.sink_clock = audio_sink.SinkClock()
    reader.segment_table = audio_sink.SegmentTable()

    session_dir = audio_sink.make_session_dir(gen)
    reader._session_dir = session_dir

    sink = audio_sink.SinkProcess(gen)
    try:
        await sink.start()
    except (FileNotFoundError, OSError) as exc:
        logging.error("Failed to start persistent audio sink: %s", exc)
        _notify_console(
            reader,
            f"[bold red]Audio sink failed to start: {exc}[/bold red]\n"
            "[yellow]Falling back to legacy audio.[/yellow]",
        )
        reader.sink = None
        await _start_legacy(reader)
        return

    # Startup health-check (§3.1): ffplay exits immediately on a bad argv
    # (e.g. a removed/renamed option). Give it a moment, then verify it is
    # still alive before wiring up the writer.
    await asyncio.sleep(config.SINK_STARTUP_GRACE_S)
    rc = sink.proc.returncode if sink.proc is not None else None
    if rc is not None:
        logging.error(
            "Persistent audio sink exited immediately (rc=%s) for gen %s; "
            "falling back to legacy audio",
            rc, gen,
        )
        _notify_console(
            reader,
            f"[bold red]Audio sink exited immediately (rc={rc}).[/bold red]\n"
            "[yellow]Falling back to legacy audio.[/yellow]",
        )
        reader.sink = None
        await _start_legacy(reader)
        return
    reader.sink = sink

    reader.writer_task = asyncio.create_task(audio_sink.writer_task(reader, gen))
    reader.producer_task = asyncio.create_task(_producer_loop_sink(reader, gen))
    reader.player_task = asyncio.create_task(_sink_supervisor(reader, gen))


async def _start_legacy(reader):
    """Legacy per-file pipeline (``--legacy-audio``)."""
    await _cancel_task(reader.producer_task)
    await _cancel_task(reader.player_task)
    reader.producer_task = None
    reader.player_task = None
    await asyncio.sleep(0.05)
    reader.producer_task = asyncio.create_task(_producer_loop(reader))
    reader.player_task = asyncio.create_task(_player_loop(reader))


async def _sink_supervisor(reader, gen):
    """Await the sink's exit; set ``playback_finished_event`` on real completion."""
    sink = reader.sink
    if sink is None:
        return
    try:
        await sink.wait()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logging.warning("Sink wait failed (gen %s): %s", gen, exc)

    if gen != reader.audio_generation:
        return

    if not reader._sink_eof_intended:
        logging.warning("Audio sink exited before EOF (gen %s); scheduling restart", gen)
        if reader.running and not reader.is_paused:
            reader.pending_restart_task = asyncio.create_task(
                reader._restart_audio_after_navigation()
            )
    reader.playback_finished_event.set()


# ── Persistent producer (§13) ──────────────────────────────────────────────


async def _producer_loop_sink(reader, gen):
    """Generate TTS files and enqueue :class:`DecodeJob` objects."""
    if not reader.tts_model or not reader.tts_model.initialized:
        try:
            await asyncio.wait_for(reader.decode_queue.put(audio_sink.EOF_SENTINEL), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        return

    resume = reader._resume
    reader._resume = None
    if resume:
        producer_pos = resume[0]
        resume_src = float(resume[1] or 0.0)
    else:
        producer_pos = (reader.chapter_idx, reader.paragraph_idx, reader.sentence_idx)
        resume_src = 0.0

    seq = 0
    try:
        while reader.running and gen == reader.audio_generation:
            try:
                c, p, s = producer_pos
                sentences = content_parser.split_into_sentences(reader.chapters[c][p])
                text = sentences[s]
            except IndexError:
                break
            if not text or not text.strip():
                next_pos = reader._advance_position(producer_pos, wrap=False)
                if not next_pos:
                    break
                producer_pos = next_pos
                continue

            # --- Fragment merging (abbreviation-only "sentences") ---
            merged = False
            is_abbrev_fragment = re.fullmatch(ABBREVIATION_PATTERN, text.strip())
            if is_abbrev_fragment and s + 1 < len(sentences):
                text += " " + sentences[s + 1]
                merged = True

            original_text = text
            output_format = reader.tts_model.output_format
            output_filename = os.path.join(reader._session_dir, f"seg_{seq}.{output_format}")
            seq += 1

            try:
                if gen != reader.audio_generation:
                    break

                for attempt in range(3):
                    try:
                        if os.path.exists(output_filename):
                            os.remove(output_filename)
                        break
                    except OSError:
                        if attempt < 2:
                            await asyncio.sleep(0.05)

                sanitized_text = content_parser.sanitize_text_for_tts(original_text)

                timing_info = None
                if hasattr(reader.tts_model, 'generate_audio_with_timing'):
                    try:
                        timing_info = await reader.tts_model.generate_audio_with_timing(sanitized_text, output_filename)
                    except Exception as exc:  # noqa: BLE001
                        logging.error(
                            "TTS timing generation failed for text '%s...' (sanitized: '%s...'): %s",
                            original_text[:50], sanitized_text[:50], exc,
                        )
                        await reader.tts_model.generate_audio(sanitized_text, output_filename)
                else:
                    await reader.tts_model.generate_audio(sanitized_text, output_filename)

                if gen != reader.audio_generation:
                    break

                duration = await get_audio_duration(output_filename)

                if timing_info is None:
                    from .timing_calculator import process_tts_timing_data
                    timing_info = process_tts_timing_data(original_text, [], duration)

                job = audio_sink.DecodeJob(
                    generation=gen,
                    file_path=output_filename,
                    pos=producer_pos,
                    timing_info=timing_info,
                    requested_speed=reader.playback_speed,
                    duration=float(duration or 0.0),
                    src_offset=resume_src,
                )
                resume_src = 0.0

                # Bounded enqueue: backpressure while the writer drains the pipe.
                while reader.running and gen == reader.audio_generation:
                    try:
                        reader.decode_queue.put_nowait(job)
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.05)

                next_pos = reader._advance_position(producer_pos, wrap=False)
                if merged and next_pos:
                    next_pos = reader._advance_position(next_pos, wrap=False)
                if not next_pos:
                    break
                producer_pos = next_pos
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                if reader.running:
                    try:
                        sanitized_for_log = content_parser.sanitize_text_for_tts(original_text)
                        logging.error(
                            "TTS Error in producer: %s\nOriginal text: '%s...'\nSanitized text: '%s...'",
                            exc, original_text[:100], sanitized_for_log[:100],
                        )
                    except Exception:  # noqa: BLE001
                        logging.error("TTS Error in producer: %s", exc, exc_info=True)
                    await asyncio.sleep(2)
                continue
    except asyncio.CancelledError:
        pass
    finally:
        # Only signal EOF for the still-current generation (teardown bumps gen).
        if gen == reader.audio_generation:
            try:
                await asyncio.wait_for(
                    reader.decode_queue.put(audio_sink.EOF_SENTINEL), timeout=1.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass


# ── Legacy per-file loops (preserved behind the flag) ──────────────────────


async def _producer_loop(reader):
    """Producer loop to generate audio files."""
    if not reader.tts_model or not reader.tts_model.initialized:
        try: await asyncio.wait_for(reader.audio_queue.put(None), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError): pass
        return

    producer_pos = (reader.chapter_idx, reader.paragraph_idx, reader.sentence_idx)
    buffer_index = 0
    try:
        while reader.running:
            if reader.audio_queue.full():
                await asyncio.sleep(0.1)
                continue
            try:
                c, p, s = producer_pos
                sentences = content_parser.split_into_sentences(reader.chapters[c][p])
                text = sentences[s]
            except IndexError: break
            if not text or not text.strip():
                next_pos = reader._advance_position(producer_pos, wrap=False)
                if not next_pos: break
                producer_pos = next_pos
                continue

            # --- Start of fragment merging logic ---
            merged = False
            is_abbrev_fragment = re.fullmatch(ABBREVIATION_PATTERN, text.strip())

            if is_abbrev_fragment and s + 1 < len(sentences):
                text += " " + sentences[s+1]
                merged = True
            # --- End of fragment merging logic ---

            original_text = text

            output_format = reader.tts_model.output_format
            output_filename = f"{config.AUDIO_BUFFERS[buffer_index]}.{output_format}"

            try:
                if not reader.running: break

                for attempt in range(3):
                    try:
                        if os.path.exists(output_filename): os.remove(output_filename)
                        break
                    except OSError:
                        if attempt < 2: await asyncio.sleep(0.05)

                sanitized_text = content_parser.sanitize_text_for_tts(original_text)

                timing_info = None

                if hasattr(reader.tts_model, 'generate_audio_with_timing'):
                    try:
                        timing_info = await reader.tts_model.generate_audio_with_timing(sanitized_text, output_filename)
                    except Exception as e:
                        logging.error(f"TTS timing generation failed for text '{original_text[:50]}...' (sanitized: '{sanitized_text[:50]}...'): {e}")
                        await reader.tts_model.generate_audio(sanitized_text, output_filename)
                else:
                    await reader.tts_model.generate_audio(sanitized_text, output_filename)

                duration = await get_audio_duration(output_filename)

                if not reader.running: break

                if timing_info is None:
                    from .timing_calculator import process_tts_timing_data
                    timing_info = process_tts_timing_data(original_text, [], duration)

                await asyncio.wait_for(reader.audio_queue.put((output_filename, *producer_pos, duration, timing_info)), timeout=1.0)

                next_pos = reader._advance_position(producer_pos, wrap=False)
                if merged:
                    if next_pos:
                        next_pos = reader._advance_position(next_pos, wrap=False)

                if not next_pos: break
                producer_pos = next_pos
                buffer_index = (buffer_index + 1) % len(config.AUDIO_BUFFERS)
            except asyncio.CancelledError: break
            except Exception as e:
                if reader.running:
                    try:
                        sanitized_for_log = content_parser.sanitize_text_for_tts(original_text) if 'original_text' in locals() else 'N/A'
                        original_for_log = original_text if 'original_text' in locals() else 'N/A'
                        logging.error(f"TTS Error in producer: {e}\nOriginal text: '{original_for_log[:100]}...'\nSanitized text: '{sanitized_for_log[:100]}...'", exc_info=True)
                    except:
                        logging.error(f"TTS Error in producer: {e}", exc_info=True)
                    await asyncio.sleep(2)
                continue
    except asyncio.CancelledError: pass
    finally:
        try: await asyncio.wait_for(reader.audio_queue.put(None), timeout=0.5)
        except (asyncio.TimeoutError, asyncio.CancelledError): pass


async def _player_loop(reader):
    """Legacy player loop to play audio files."""
    try:
        while reader.running:
            try:
                item = await asyncio.wait_for(reader.audio_queue.get(), timeout=1.0)
                if item is None:
                    reader.audio_queue.task_done()
                    if reader.active_playback_tasks:
                        await asyncio.gather(*reader.active_playback_tasks, return_exceptions=True)
                    reader.playback_finished_event.set()
                    break
                # Unpack the queue item
                audio_file, c, p, s, duration, timing_data = item
                if isinstance(timing_data, dict):
                    timing_info = timing_data
                else:
                    timing_info = {"word_timings": timing_data, "speech_duration": duration, "total_duration": duration}

                word_timings = timing_info.get("word_timings", [])

                if not os.path.exists(audio_file):
                    reader.audio_queue.task_done()
                    continue
                try:
                    reader.loop.call_soon_threadsafe(
                        reader._post_command_sync,
                        ('_new_sentence_started', (c, p, s, duration, timing_data))
                    )
                except RuntimeError:
                    reader.audio_queue.task_done()
                    break
                try:
                    cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'error']

                    if abs(reader.playback_speed - 1.0) > 0.01:
                        speed = reader.playback_speed
                        filters = []

                        while speed > 2.0:
                            filters.append('atempo=2.0')
                            speed /= 2.0
                        while speed < 0.5:
                            filters.append('atempo=0.5')
                            speed /= 0.5
                        if abs(speed - 1.0) > 0.01:
                            filters.append(f'atempo={speed:.3f}')

                        if filters:
                            filter_chain = ','.join(filters)
                            cmd.extend(['-af', filter_chain])

                    cmd.append(audio_file)
                    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    reader.playback_processes.append(process)
                except Exception:
                    reader.audio_queue.task_done()
                    continue

                async def await_and_remove(proc, file):
                    task = asyncio.current_task()
                    try:
                        await proc.wait()
                    except Exception: pass
                    finally:
                        try:
                            if proc in reader.playback_processes: reader.playback_processes.remove(proc)
                        except ValueError: pass
                        for attempt in range(3):
                            try:
                                if os.path.exists(file): os.remove(file)
                                break
                            except OSError:
                                if attempt < 2: await asyncio.sleep(0.05)
                        if task in reader.active_playback_tasks:
                            reader.active_playback_tasks.remove(task)

                playback_task = asyncio.create_task(await_and_remove(process, audio_file))
                reader.active_playback_tasks.append(playback_task)

                # Overlap/crossfade was removed (spec-v3 §9.1): no per-sentence
                # sleep and no overlap; TTS overlap accessors return ``None``.
                reader.audio_queue.task_done()
            except asyncio.TimeoutError:
                if not reader.running: break
                continue
            except asyncio.CancelledError: break
    except asyncio.CancelledError: pass
    finally:
        for process in reader.playback_processes.copy():
            try:
                if process.returncode is None: process.terminate()
            except (ProcessLookupError, AttributeError): pass
