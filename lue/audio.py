import asyncio
import os
import re
import subprocess
import logging
from . import config, content_parser

# This pattern is used to both clean text for TTS and detect sentence fragments.
ABBREVIATION_PATTERN = r'\b(Mr|Mrs|Ms|Dr|Prof|Rev|Hon|Jr|Sr|Cpl|Sgt|Gen|Col|Capt|Lt|Pvt|vs|viz|Co|Inc|Ltd|Corp|St|Ave|Blvd)\.'
INITIAL_PATTERN = r'\b([A-Z])\.(?=\s[A-Z])'


# Word mapping functionality moved to timing_calculator.py
# Import it here for backward compatibility
from .timing_calculator import create_word_mapping as _create_word_mapping


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

async def stop_and_clear_audio(reader):
    """Stop audio playback and clear the audio queue."""
    tasks_to_cancel = []
    for task in [reader.producer_task, reader.player_task]:
        if task and not task.done():
            task.cancel()
            tasks_to_cancel.append(task)
    if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    
    reader.producer_task = None
    reader.player_task = None
    
    processes_to_kill = reader.playback_processes.copy()
    reader.playback_processes.clear()
    for process in processes_to_kill:
        try:
            if process.returncode is None:
                process.terminate()
                try: await asyncio.wait_for(process.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=0.1)
        except (ProcessLookupError, AttributeError, asyncio.TimeoutError): pass
    
    try:
        pkill_proc = await asyncio.create_subprocess_exec('pkill', '-9', '-f', 'ffplay.*buffer_', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.wait_for(pkill_proc.wait(), timeout=0.3)
    except (FileNotFoundError, asyncio.TimeoutError): pass
    
    while not reader.audio_queue.empty():
        try:
            reader.audio_queue.get_nowait()
            reader.audio_queue.task_done()
        except asyncio.QueueEmpty: break
    
    await asyncio.sleep(0.1)
    
    # More aggressive cleanup with longer delays for file system operations
    for buf_base in config.AUDIO_BUFFERS:
        for ext in ['.mp3', '.wav']:
            buf = f"{buf_base}{ext}"
            for attempt in range(5):  # Increased attempts
                try:
                    if os.path.exists(buf): 
                        os.remove(buf)
                    break
                except OSError:
                    if attempt < 4: 
                        await asyncio.sleep(0.1)  # Longer delay
    
    # Clean up any remaining lookahead temp files
    import glob
    for la_file in glob.glob(os.path.join(config.LOOKAHEAD_TEMP_DIR, "la_*")):
        try:
            if os.path.isfile(la_file):
                os.remove(la_file)
        except OSError:
            pass

    
    await asyncio.sleep(0.2)  # Longer final delay


        
async def get_audio_duration(file_path):
    """Get the duration of an audio file."""
    command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=subprocess.DEVNULL)
    stdout, _ = await process.communicate()
    if process.returncode != 0: return None
    try: return float(stdout.decode().strip())
    except (ValueError, TypeError): return None

async def play_from_current_position(reader):
    """Start the audio producer and player loops using LookaheadBuffer."""
    if not reader.is_paused and reader.running and reader.tts_model:
        # Cancel existing tasks and wait for them to complete
        for task in [reader.producer_task, reader.player_task]:
            if task and not task.done():
                task.cancel()
                try: 
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError): 
                    pass
        
        # Ensure tasks are properly cleaned up
        reader.producer_task = None
        reader.player_task = None
        
        # Small delay to ensure cleanup is complete
        await asyncio.sleep(0.05)
        
        # Create lookahead buffer
        from .lookahead_buffer import LookaheadBuffer
        reader.lookahead_buffer = LookaheadBuffer(
            reader,
            target_sentences=config.LOOKAHEAD_SENTENCES,
            min_start_items=config.PREBUFFER_MIN_ITEMS,
            max_errors=config.LOOKAHEAD_MAX_ERRORS,
        )
        reader.tts_cache = getattr(reader, 'tts_cache', None)
        reader.producer_task = reader.lookahead_buffer.start_producer()
        
        # Wait for pre-buffer threshold before starting the player.
        # Two thresholds: minimum item count OR minimum seconds of audio,
        # whichever is reached first (plus a 30-second safety timeout).
        import time as _time
        _buffer_start = _time.monotonic()
        _total_duration_buffered = 0.0
        while (reader.running 
               and reader.lookahead_buffer.qsize < config.PREBUFFER_MIN_ITEMS
               and _total_duration_buffered < config.PREBUFFER_MIN_SECONDS
               and reader.lookahead_buffer._is_running
               and not reader.producer_task.done()):
            # Check accumulated audio duration from buffered items
            _total_duration_buffered = getattr(
                reader.lookahead_buffer, '_buffered_duration', 0.0
            )
            # 30-second safety timeout
            if _time.monotonic() - _buffer_start > 30.0:
                break
            await asyncio.sleep(0.05)
        
        # Start player regardless of whether threshold was reached
        reader.player_task = asyncio.create_task(_player_loop(reader))


async def _player_loop(reader):
    """Player loop to play audio files from the lookahead buffer."""
    buf = getattr(reader, 'lookahead_buffer', None)
    use_buffer = buf is not None

    def mark_done():
        """Call task_done on the appropriate queue."""
        if use_buffer:
            buf.task_done()
        else:
            try:
                reader.audio_queue.task_done()
            except ValueError:
                pass  # too many task_done calls

    async def signal_refill():
        """Signal the lookahead buffer producer to refill."""
        if use_buffer:
            try:
                await buf.signal_refill()
            except Exception:
                pass

    try:
        while reader.running:
            try:
                if use_buffer:
                    item = await buf.get()
                else:
                    item = await asyncio.wait_for(reader.audio_queue.get(), timeout=1.0)

                if item is None:
                    mark_done()
                    if reader.active_playback_tasks:
                        await asyncio.gather(*reader.active_playback_tasks, return_exceptions=True)
                    reader.playback_finished_event.set()
                    break
                # Unpack the queue item
                audio_file, c, p, s, duration, timing_data = item
                if isinstance(timing_data, dict):
                    timing_info = timing_data
                else:
                    # Old format, timing_data is word_timings
                    timing_info = {"word_timings": timing_data, "speech_duration": duration, "total_duration": duration}

                word_timings = timing_info.get("word_timings", [])
                
                if not os.path.exists(audio_file):
                    mark_done()
                    await signal_refill()
                    continue
                if duration is None or duration <= 0:
                    mark_done()
                    await signal_refill()
                    continue
                try:
                    # Post a command to the main loop to handle the state transition atomically
                    reader.loop.call_soon_threadsafe(
                        reader._post_command_sync,
                        ('_new_sentence_started', (c, p, s, duration, timing_data))
                    )
                except RuntimeError:
                    mark_done()
                    break
                try:
                    # Build ffplay command with speed control using atempo filter
                    cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'error']
                    
                    # Add atempo filter if speed is not 1.0
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
                    process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    reader.playback_processes.append(process)
                except Exception:
                    mark_done()
                    await signal_refill()
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
                
                # Calculate dynamic overlap based on playback speed
                base_overlap = config.OVERLAP_SECONDS
                if reader.tts_model and hasattr(reader.tts_model, 'get_overlap_seconds'):
                    tts_overlap = reader.tts_model.get_overlap_seconds()
                    if tts_overlap is not None:
                        base_overlap = tts_overlap
                
                # Apply speed-based overlap reduction
                speed = reader.playback_speed
                if speed >= 3.0:
                    overlap_seconds = 0.0
                else:
                    overlap_factor = max(0.0, min(1.0, (3.0 - speed) / (3.0 - 1.0)))
                    overlap_seconds = base_overlap * overlap_factor
                
                # Adjust duration for playback speed
                actual_duration = duration / speed
                
                await asyncio.sleep(max(0.1, actual_duration - overlap_seconds))
                mark_done()
                await signal_refill()
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
