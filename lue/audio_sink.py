"""Persistent gapless audio sink (spec-v3).

This module implements the long-lived ``ffplay`` sink that is fed raw
``s16le / 48000 Hz / 2ch`` PCM over stdin by a single paced writer task, the
authoritative :class:`SinkClock`, the :class:`SegmentTable` used for word /
sentence highlighting, and the per-sentence decode chain.

Only the sink ``stdin`` StreamWriter is written here, and only by the
generation's ``writer_task`` (spec-v3 §3.1 / ADR-6).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field

from . import config

log = logging.getLogger(__name__)

BYTES_PER_SEC = config.BYTES_PER_SEC

# All byte counts are rounded down to a whole 4-byte sample frame (s16 stereo).
_FRAME = 4


def _bytes_for_ms(ms: int) -> int:
    return max(_FRAME, (int(BYTES_PER_SEC * ms / 1000.0) // _FRAME) * _FRAME)


WRITER_CHUNK_BYTES = _bytes_for_ms(config.WRITER_CHUNK_MS)
PCM_RING_MAX_BYTES = _bytes_for_ms(config.PCM_RING_MAX_MS)
PCM_LOW_WATER_BYTES = _bytes_for_ms(config.PCM_LOW_WATER_MS)
UNDERRUN_PAD_BYTES = _bytes_for_ms(config.UNDERRUN_PAD_MS)
SINK_PRIME_BYTES = _bytes_for_ms(config.SINK_PRIME_MS)

EOF_SENTINEL = object()


class SinkWriteError(Exception):
    """Raised when the sink pipe stalls (drain timeout)."""


# ── Filter / command builders (§6.2, §7.2) ─────────────────────────────────


def build_atempo_chain(speed: float) -> str:
    """Return the ``atempo`` chain (including a trailing comma) or ``""``."""
    if abs(speed - 1.0) < 1e-4:
        return ""
    parts: list[str] = []
    r = float(speed)
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    if abs(r - 1.0) > 1e-4:
        parts.append(f"atempo={r:.4f}")
    if not parts:
        return ""
    return ",".join(parts) + ","


_TRIM_FILTER = (
    "silenceremove=start_periods=1:start_duration=0.30:start_threshold=-45dB:"
    "stop_periods=-1:stop_duration=0.30:stop_threshold=-45dB,"
)


def build_decode_filter(speed: float, trim_silence: bool = False) -> str:
    chain = build_atempo_chain(speed)
    prefix = _TRIM_FILTER if trim_silence else ""
    return (
        prefix
        + chain
        + "aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo"
    )


def build_decode_command(
    src_file: str, speed: float, src_offset: float = 0.0, trim_silence: bool = False
) -> list[str]:
    """Exact decode command (spec-v3 §7.2)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if src_offset and src_offset > 0:
        cmd += ["-ss", f"{src_offset:.6f}"]
    cmd += [
        "-i",
        src_file,
        "-map",
        "0:a:0",
        "-filter:a",
        build_decode_filter(speed, trim_silence),
        "-ar",
        "48000",
        "-ch_layout",
        "stereo",
        "-f",
        "s16le",
        "pipe:1",
    ]
    return cmd


# ── Sink process (§3.1, ADR-6) ─────────────────────────────────────────────


class SinkProcess:
    """Wraps the persistent ``ffplay`` sink."""

    def __init__(self, generation: int):
        self.generation = generation
        self.proc: asyncio.subprocess.Process | None = None
        self._stdin = None
        self._closed = False

    @property
    def stdin(self):
        return self._stdin

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    async def start(self) -> "SinkProcess":
        gen = self.generation
        self.proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-window_title",
            f"lue-audio-sink-{os.getpid()}-{gen}",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ch_layout",
            "stereo",
            "-i",
            "pipe:0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._stdin = self.proc.stdin
        return self

    def close_stdin(self) -> None:
        """Idempotent; may be called by writer_task or teardown (whichever first)."""
        if self._closed:
            return
        self._closed = True
        writer = self._stdin
        if writer is None:
            return
        try:
            writer.close()
        except Exception:  # noqa: BLE001 - tolerate already-closed / broken pipe
            pass

    async def wait(self):
        if self.proc is None:
            return None
        return await self.proc.wait()

    async def terminate(self) -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        try:
            self.proc.terminate()
        except (ProcessLookupError, AttributeError):
            return
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=config.SINK_TERM_TIMEOUT_S)
        except asyncio.TimeoutError:
            await self.kill()

    async def kill(self) -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        try:
            self.proc.kill()
        except (ProcessLookupError, AttributeError):
            return
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=config.SINK_KILL_TIMEOUT_S)
        except asyncio.TimeoutError:
            pass


# ── Segment table (§4.3) ───────────────────────────────────────────────────


@dataclass
class Segment:
    generation: int
    pos: tuple | None
    out_start: float
    out_duration: float = 0.0
    src_duration: float = 0.0
    src_offset: float = 0.0
    speed: float = 1.0
    word_timings: list | None = None
    word_mapping: list | None = None
    is_silence: bool = False
    written_bytes: int = 0
    is_failed: bool = False


@dataclass
class DecodeJob:
    generation: int
    file_path: str
    pos: tuple
    timing_info: dict | None = None
    requested_speed: float = 1.0
    duration: float = 0.0
    src_offset: float = 0.0


class SegmentTable:
    def __init__(self):
        self.segments: list[Segment] = []
        self._open: Segment | None = None

    def open_segment(self, seg: Segment) -> Segment:
        self._open = seg
        self.segments.append(seg)
        return seg

    def extend(self, nbytes: int) -> None:
        if self._open is None:
            return
        self._open.written_bytes += nbytes
        self._open.out_duration = self._open.written_bytes / BYTES_PER_SEC

    def close_segment(self) -> Segment | None:
        seg = self._open
        self._open = None
        return seg

    def total_duration(self) -> float:
        total = 0.0
        for seg in self.segments:
            total = max(total, seg.out_start + seg.out_duration)
        return total

    def find(self, P: float) -> Segment | None:
        if not self.segments:
            return None
        found = self.segments[0]
        for seg in self.segments:
            if seg.out_start <= P + 1e-9:
                found = seg
            else:
                break
        return found

    def clear(self) -> None:
        self.segments.clear()
        self._open = None

    def discard_generation(self, gen: int) -> None:
        self.segments = [s for s in self.segments if s.generation != gen]
        if self._open is not None and self._open.generation == gen:
            self._open = None


# ── Sink clock (§4.3) ──────────────────────────────────────────────────────


class SinkClock:
    def __init__(self, lead: float | None = None):
        self.lead = config.SINK_LEAD if lead is None else lead
        self.t_first_write: float | None = None
        self.t_play_start: float | None = None
        self.p_end: float = 0.0

    def anchor(self, t_first_write: float) -> None:
        self.t_first_write = t_first_write
        self.t_play_start = t_first_write + self.lead

    def position(self, now: float) -> float:
        if self.t_play_start is None:
            return 0.0
        p = now - self.t_play_start
        if p < 0.0:
            return 0.0
        if self.p_end > 0.0 and p > self.p_end:
            return self.p_end
        return p

    def reset(self) -> None:
        self.t_first_write = None
        self.t_play_start = None
        self.p_end = 0.0


# ── Writer helpers ─────────────────────────────────────────────────────────


async def _write_chunk(table: SegmentTable, clock: SinkClock, stream, data: bytes) -> None:
    """Single write path for the sink stdin (sole owner: writer_task)."""
    if not data:
        return
    if clock.t_first_write is None:
        clock.anchor(asyncio.get_running_loop().time())
    stream.write(data)
    try:
        await asyncio.wait_for(stream.drain(), timeout=config.SINK_WRITE_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise SinkWriteError("sink drain timed out") from exc
    table.extend(len(data))
    clock.p_end = table.total_duration()


async def _pace(table: SegmentTable, clock: SinkClock) -> None:
    ahead = table.total_duration() - clock.position(asyncio.get_running_loop().time())
    limit = config.PCM_RING_MAX_MS / 1000.0
    if ahead > limit:
        await asyncio.sleep(min(0.05, ahead - limit))


async def _write_silence(
    reader,
    gen: int,
    table: SegmentTable,
    clock: SinkClock,
    stream,
    nbytes: int,
    silence_seg: Segment | None,
) -> tuple[bool, Segment | None]:
    if silence_seg is None:
        silence_seg = Segment(
            generation=gen,
            pos=None,
            out_start=table.total_duration(),
            speed=1.0,
            is_silence=True,
        )
        table.open_segment(silence_seg)
    remaining = nbytes
    while remaining > 0:
        if gen != reader.audio_generation:
            return False, silence_seg
        chunk = min(WRITER_CHUNK_BYTES, remaining)
        await _write_chunk(table, clock, stream, b"\x00" * chunk)
        remaining -= chunk
        await _pace(table, clock)
    silence_seg.src_duration = silence_seg.out_duration
    return True, silence_seg


async def _run_decode_once(reader, gen: int, job: DecodeJob, seg: Segment) -> tuple[int, int | str]:
    cmd = build_decode_command(
        job.file_path, job.requested_speed, job.src_offset, config.TRIM_SILENCE
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    reader._current_decoder = proc
    produced = 0
    try:
        while True:
            if gen != reader.audio_generation:
                return produced, "cancelled"
            chunk = await proc.stdout.read(WRITER_CHUNK_BYTES)
            if not chunk:
                break
            produced += len(chunk)
            await _write_chunk(reader.segment_table, reader.sink_clock, reader.sink.stdin, chunk)
            await _pace(reader.segment_table, reader.sink_clock)
        rc = await proc.wait()
        return produced, rc
    finally:
        reader._current_decoder = None
        if proc.returncode is None:
            try:
                proc.terminate()
            except (ProcessLookupError, AttributeError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=config.DECODE_TERM_TIMEOUT_S)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except (ProcessLookupError, AttributeError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=config.DECODE_TERM_TIMEOUT_S)
                except asyncio.TimeoutError:
                    pass


def _safe_remove(path: str) -> None:
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except OSError:
            time.sleep(0.02)


async def _decode_and_write(reader, gen: int, job: DecodeJob) -> None:
    table: SegmentTable = reader.segment_table
    timing = job.timing_info or {}
    seg = Segment(
        generation=gen,
        pos=job.pos,
        out_start=table.total_duration(),
        src_duration=float(job.duration or 0.0),
        src_offset=float(job.src_offset or 0.0),
        speed=float(job.requested_speed or 1.0),
        word_timings=timing.get("word_timings"),
        word_mapping=timing.get("word_mapping"),
    )
    table.open_segment(seg)
    try:
        produced, rc = await _run_decode_once(reader, gen, job, seg)
        if rc != 0 or produced == 0:
            # retry once (spec §7.5)
            if gen == reader.audio_generation:
                log.error("Decode failed (rc=%s, bytes=%s) for %s; retrying", rc, produced, job.file_path)
                table.close_segment()
                seg = Segment(
                    generation=gen,
                    pos=job.pos,
                    out_start=table.total_duration(),
                    src_duration=float(job.duration or 0.0),
                    src_offset=float(job.src_offset or 0.0),
                    speed=float(job.requested_speed or 1.0),
                    word_timings=timing.get("word_timings"),
                    word_mapping=timing.get("word_mapping"),
                    is_failed=True,
                )
                table.open_segment(seg)
                produced, rc = await _run_decode_once(reader, gen, job, seg)
        if rc != 0 or produced == 0:
            if gen == reader.audio_generation:
                log.error("Decode failed twice for %s; inserting %d ms silence",
                          job.file_path, config.DECODE_FAIL_SILENCE_MS)
                table.close_segment()
                silence_ok, _ = await _write_silence(
                    reader, gen, table, reader.sink_clock, reader.sink.stdin,
                    _bytes_for_ms(config.DECODE_FAIL_SILENCE_MS), None,
                )
                table.close_segment()
    finally:
        if table._open is seg:
            table.close_segment()
        _safe_remove(job.file_path)


# ── writer_task (§3.1, §8) ─────────────────────────────────────────────────


async def writer_task(reader, gen: int) -> None:
    sink: SinkProcess = reader.sink
    clock: SinkClock = reader.sink_clock
    table: SegmentTable = reader.segment_table
    queue = reader.decode_queue
    stream = sink.stdin
    loop = asyncio.get_running_loop()

    started = False
    silence_seg: Segment | None = None
    pad_episode_ms = 0
    warned = False
    intended_eof = False

    try:
        while True:
            if gen != reader.audio_generation:
                return
            try:
                job = await asyncio.wait_for(queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                job = None

            if job is EOF_SENTINEL:
                intended_eof = True
                try:
                    queue.task_done()
                except ValueError:
                    pass
                break

            if job is None:
                if not started:
                    continue
                ahead = table.total_duration() - clock.position(loop.time())
                if ahead * 1000.0 <= config.PCM_LOW_WATER_MS:
                    if silence_seg is None:
                        silence_seg = Segment(
                            generation=gen,
                            pos=None,
                            out_start=table.total_duration(),
                            speed=1.0,
                            is_silence=True,
                        )
                        table.open_segment(silence_seg)
                    pad_episode_ms += config.UNDERRUN_PAD_MS
                    if pad_episode_ms > config.MAX_UNDERRUN_PAD_MS and not warned:
                        log.warning(
                            "Sink starvation exceeded %d ms; padding silence (gen %s)",
                            config.MAX_UNDERRUN_PAD_MS, gen,
                        )
                        warned = True
                    remaining = UNDERRUN_PAD_BYTES
                    while remaining > 0:
                        if gen != reader.audio_generation:
                            return
                        chunk = min(WRITER_CHUNK_BYTES, remaining)
                        await _write_chunk(table, clock, stream, b"\x00" * chunk)
                        remaining -= chunk
                        await _pace(table, clock)
                    silence_seg.src_duration = silence_seg.out_duration
                continue

            # Real job: close any open silence segment first.
            if silence_seg is not None:
                table.close_segment()
                silence_seg = None
            pad_episode_ms = 0
            warned = False
            try:
                queue.task_done()
            except ValueError:
                pass
            try:
                await _decode_and_write(reader, gen, job)
            except SinkWriteError:
                raise
            started = True
            await _pace(table, clock)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, SinkWriteError) as exc:
        log.warning("Sink pipe closed / stalled (gen %s): %s", gen, exc)
        try:
            sink.close_stdin()
        except Exception:  # noqa: BLE001
            pass
        try:
            await sink.terminate()
        except Exception:  # noqa: BLE001
            pass
        return
    except asyncio.CancelledError:
        raise

    if gen != reader.audio_generation:
        return
    if intended_eof:
        reader._sink_eof_intended = True
    try:
        sink.close_stdin()
    except Exception:  # noqa: BLE001
        pass
    try:
        await asyncio.wait_for(sink.stdin.wait_closed(), timeout=config.SINK_EOF_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        pass


# ── Temp session dirs (§9.4) ───────────────────────────────────────────────


def sweep_stale_sessions() -> None:
    root = config.AUDIO_TMP_ROOT
    try:
        entries = os.listdir(root)
    except OSError:
        return
    now = time.time()
    for name in entries:
        if not name.startswith("session-"):
            continue
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) > config.AUDIO_TMP_MAX_AGE_S:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def make_session_dir(gen: int) -> str:
    path = os.path.join(config.AUDIO_TMP_ROOT, f"session-{os.getpid()}-{gen}")
    os.makedirs(path, exist_ok=True)
    return path