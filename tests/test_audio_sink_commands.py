"""Regression tests for the persistent-sink / decode command builders.

These guard against the ffmpeg n9.0.2 regression where the removed ``-ac``
option made both the ``ffplay`` sink and the ``ffmpeg`` decoder fail at
startup (silent TTS).  See diagnosis: ``lue/audio_sink.py``.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from lue import audio, audio_sink, config
from lue import __main__ as cli_main
from lue.reader import Lue
from lue.timing_calculator import select_highlight_word


# ── build_decode_command / build_decode_filter ─────────────────────────────


def test_build_decode_command_uses_ch_layout_not_ac():
    cmd = audio_sink.build_decode_command("/tmp/whatever.wav", 1.0)
    assert "-ac" not in cmd, f"legacy -ac option leaked into decode cmd: {cmd}"
    assert "-af" not in cmd
    assert cmd[cmd.index("-ch_layout") + 1] == "stereo"
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "s16le"
    assert cmd[cmd.index("-ar") + 1] == "48000"


def test_build_decode_filter_keeps_valid_aformat_option():
    # aformat's option is ``channel_layouts`` (plural); ``channel_layout`` is
    # NOT a valid aformat option on ffmpeg n9.0.2.
    flt = audio_sink.build_decode_filter(1.0)
    assert "channel_layouts=stereo" in flt
    assert "aresample=48000" in flt


# ── SinkProcess.start argv ─────────────────────────────────────────────────


async def test_sink_start_argv(monkeypatch):
    captured: dict = {}

    def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs

        async def _co():
            return SimpleNamespace(pid=1234, stdin=Mock())

        return _co()

    monkeypatch.setattr(
        audio_sink.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )

    await audio_sink.SinkProcess(1).start()
    args = captured["args"]

    assert "-ac" not in args, f"legacy -ac option leaked into sink argv: {args}"
    assert "-af" not in args
    assert "-nostdin" not in args, "ffplay must not receive -nostdin"
    assert args[args.index("-ch_layout") + 1] == "stereo"
    assert args[args.index("-f") + 1] == "s16le"
    assert args[args.index("-ar") + 1] == "48000"
    assert args[args.index("-i") + 1] == "pipe:0"
    assert captured["kwargs"]["stdin"] == asyncio.subprocess.PIPE


# ── _start_persistent startup health-check ─────────────────────────────────


def _fake_reader() -> SimpleNamespace:
    async def _noop_restart() -> None:
        """Stand-in for ``Lue._restart_audio_after_navigation``."""

    return SimpleNamespace(
        console=Mock(),
        audio_generation=1,
        producer_task=None,
        writer_task=None,
        player_task=None,
        pending_restart_task=None,
        _restart_audio_after_navigation=_noop_restart,
        playback_finished_event=asyncio.Event(),
        _sink_eof_intended=False,
        sink=None,
        sink_clock=None,
        segment_table=None,
        _session_dir=None,
        tts_model=None,
        running=True,
        is_paused=False,
        decode_queue=asyncio.Queue(),
        _resume=None,
    )


async def test_start_persistent_falls_back_when_sink_exits_immediately(
    monkeypatch, tmp_path
):
    fallback = {"called": False}

    async def fake_start_legacy(reader):
        fallback["called"] = True

    class DeadSink:
        last = None

        def __init__(self, gen):
            self.generation = gen
            self.proc = SimpleNamespace(returncode=1)
            self.closed = False
            DeadSink.last = self

        async def start(self):
            return self

        def close_stdin(self):
            self.closed = True

    monkeypatch.setattr(audio_sink, "SinkProcess", DeadSink)
    monkeypatch.setattr(audio, "_start_legacy", fake_start_legacy)
    monkeypatch.setattr(config, "SINK_STARTUP_GRACE_S", 0.0)
    monkeypatch.setattr(audio_sink, "make_session_dir", lambda gen: str(tmp_path))

    reader = _fake_reader()
    await audio._start_persistent(reader)

    assert fallback["called"] is True
    assert reader.sink is None
    # W1: the failed generation's clock/table must not survive, otherwise the
    # highlight loop keeps using the dead sink instead of the legacy path.
    assert reader.sink_clock is None
    assert reader.segment_table is None
    # W1: close the write end even though the process already exited.
    assert DeadSink.last.closed is True
    reader.console.print.assert_called()


async def test_start_persistent_clears_state_when_sink_binary_missing(
    monkeypatch, tmp_path
):
    """W1: a FileNotFoundError fallback must not leak clock/table or an fd."""

    fallback = {"called": False}

    async def fake_start_legacy(reader):
        fallback["called"] = True

    class MissingSink:
        def __init__(self, gen):
            self.generation = gen
            self.proc = None

        async def start(self):
            raise FileNotFoundError("ffplay")

    monkeypatch.setattr(audio_sink, "SinkProcess", MissingSink)
    monkeypatch.setattr(audio, "_start_legacy", fake_start_legacy)
    monkeypatch.setattr(audio_sink, "make_session_dir", lambda gen: str(tmp_path))

    reader = _fake_reader()
    await audio._start_persistent(reader)

    assert fallback["called"] is True
    assert reader.sink is None
    assert reader.sink_clock is None
    assert reader.segment_table is None


async def test_start_persistent_wires_tasks_when_sink_alive(monkeypatch, tmp_path):
    fallback = {"called": False}

    class _FakeStdin:
        """Minimal StreamWriter contract used by ``writer_task`` on EOF."""

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        async def wait_closed(self):
            return None

    async def fake_start_legacy(reader):
        fallback["called"] = True

    class LiveSink:
        def __init__(self, gen):
            self.generation = gen
            self.proc = SimpleNamespace(returncode=None)
            self.stdin = _FakeStdin()

        async def start(self):
            return self

        def close_stdin(self):
            self.stdin.close()

        async def wait(self):
            return 0

        async def terminate(self):
            pass

    monkeypatch.setattr(audio_sink, "SinkProcess", LiveSink)
    monkeypatch.setattr(audio, "_start_legacy", fake_start_legacy)
    monkeypatch.setattr(audio_sink, "make_session_dir", lambda gen: str(tmp_path))
    monkeypatch.setattr(config, "SINK_STARTUP_GRACE_S", 0.0)

    reader = _fake_reader()
    await audio._start_persistent(reader)

    assert fallback["called"] is False
    assert reader.sink is not None

    # Let the supervisor observe the (fake) sink exit.  It must take the
    # "exited before EOF" branch and schedule a restart through the reader
    # contract instead of raising AttributeError on a bare namespace.
    await reader.player_task
    assert reader.playback_finished_event.is_set()
    assert reader.pending_restart_task is not None

    tasks = [
        getattr(reader, attr)
        for attr in ("writer_task", "producer_task", "pending_restart_task")
    ]
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Teardown-cancelled tasks may only surface CancelledError or a clean exit;
    # anything else means the fake reader violated the supervisor contract.
    assert all(
        isinstance(result, (asyncio.CancelledError, type(None))) for result in results
    ), results

# ── C1: legacy player must never overlap two ffplay processes ─────────────


class _FakeFfplay:
    """Minimal stand-in for an ffplay process that outlives the pacing sleep."""

    def __init__(self, tracker, lifetime=0.25):
        self.returncode = None
        self._tracker = tracker
        self._lifetime = lifetime
        self._done = False

    async def wait(self):
        if not self._done:
            await asyncio.sleep(self._lifetime)
            self._done = True
            self._tracker["active"] -= 1
        self.returncode = 0
        return 0

    def terminate(self):
        if not self._done:
            self._done = True
            self._tracker["active"] -= 1
        self.returncode = -15


class _ListBuffer:
    def __init__(self, items):
        self._items = list(items)
        self.done_calls = 0

    async def get(self):
        await asyncio.sleep(0)
        return self._items.pop(0) if self._items else None

    def task_done(self):
        self.done_calls += 1

    async def signal_refill(self):
        pass


async def test_legacy_player_never_overlaps_ffplay(monkeypatch, tmp_path):
    """Pacing + serialisation: at most one ffplay alive at any moment."""
    tracker = {"active": 0, "max": 0, "spawned": 0}

    async def fake_create_subprocess_exec(*args, **kwargs):
        tracker["spawned"] += 1
        tracker["active"] += 1
        tracker["max"] = max(tracker["max"], tracker["active"])
        return _FakeFfplay(tracker)

    monkeypatch.setattr(audio.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    timing = {"word_timings": [], "speech_duration": 0.05, "total_duration": 0.05}
    items = []
    for i in range(3):
        path = tmp_path / f"seg_{i}.wav"
        path.write_bytes(b"RIFF")
        # duration shorter than the fake process lifetime: without
        # serialisation the next ffplay is spawned while this one runs.
        items.append((str(path), 0, 0, i, 0.05, timing))

    reader = SimpleNamespace(
        running=True,
        lookahead_buffer=_ListBuffer(items),
        audio_queue=asyncio.Queue(),
        playback_finished_event=asyncio.Event(),
        active_playback_tasks=[],
        playback_processes=[],
        playback_speed=1.0,
        tts_model=None,
        loop=asyncio.get_running_loop(),
        _post_command_sync=lambda *args: None,
    )

    await asyncio.wait_for(audio._player_loop(reader), timeout=10.0)

    assert tracker["spawned"] == 3
    assert tracker["max"] == 1, "legacy path spawned overlapping ffplay processes"
    assert tracker["active"] == 0
    assert reader.playback_finished_event.is_set()
    # 3 real items + the None sentinel each call mark_done().
    assert reader.lookahead_buffer.done_calls == 4


async def test_legacy_player_terminates_inflight_ffplay_on_cancel(
    monkeypatch, tmp_path
):
    """Cancelling the player while it waits on ffplay stops that sentence."""
    tracker = {"active": 0, "max": 0, "spawned": 0}
    procs = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        tracker["spawned"] += 1
        tracker["active"] += 1
        tracker["max"] = max(tracker["max"], tracker["active"])
        proc = _FakeFfplay(tracker, lifetime=5.0)
        procs.append(proc)
        return proc

    monkeypatch.setattr(audio.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    timing = {"word_timings": [], "speech_duration": 0.05, "total_duration": 0.05}
    items = []
    for i in range(2):
        path = tmp_path / f"cancel_{i}.wav"
        path.write_bytes(b"RIFF")
        items.append((str(path), 0, 0, i, 0.05, timing))

    reader = SimpleNamespace(
        running=True,
        lookahead_buffer=_ListBuffer(items),
        audio_queue=asyncio.Queue(),
        playback_finished_event=asyncio.Event(),
        active_playback_tasks=[],
        playback_processes=[],
        playback_speed=1.0,
        tts_model=None,
        loop=asyncio.get_running_loop(),
        _post_command_sync=lambda *args: None,
    )

    player = asyncio.create_task(audio._player_loop(reader))
    # First ffplay is spawned; the second iteration blocks on its completion.
    await asyncio.sleep(0.2)
    assert tracker["spawned"] == 1

    player.cancel()
    await asyncio.wait_for(player, timeout=2.0)  # loop swallows the cancel

    assert procs[0].returncode == -15, "in-flight ffplay must be terminated"
    assert tracker["active"] == 0


# ── C2: EOF sentinel must never be dropped on a full decode queue ─────────


class _FakeStdin:
    def __init__(self):
        self.closed = False
        self.writes: list[bytes] = []

    def write(self, data):
        self.writes.append(bytes(data))

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


class _FakeSink:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.proc = SimpleNamespace(returncode=None)
        self.terminated = False

    def close_stdin(self):
        self.stdin.closed = True

    async def terminate(self):
        self.terminated = True


async def test_producer_retries_eof_sentinel_until_writer_has_room():
    """A full bounded queue must not swallow the EOF sentinel."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait("occupying-slot")

    reader = SimpleNamespace(
        tts_model=SimpleNamespace(initialized=True),
        chapters=[],  # loop exits immediately -> finally still sends EOF
        chapter_idx=0,
        paragraph_idx=0,
        sentence_idx=0,
        running=True,
        audio_generation=7,
        decode_queue=queue,
        _resume=None,
        playback_speed=1.0,
    )

    producer = asyncio.create_task(audio._producer_loop_sink(reader, 7))

    # Old behaviour (bounded 1.0 s put) gave up here.
    await asyncio.sleep(1.2)
    assert not producer.done(), "producer should be retrying, not dropping EOF"
    assert queue.qsize() == 1

    # Slow consumer frees the slot only after the old timeout would have fired.
    assert queue.get_nowait() == "occupying-slot"
    sentinel = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert sentinel is audio_sink.EOF_SENTINEL
    await asyncio.wait_for(producer, timeout=1.0)


async def test_writer_closes_stdin_on_eof_without_padding():
    """Receiving the sentinel closes stdin and writes no silence padding."""
    sink = _FakeSink()
    reader = SimpleNamespace(
        audio_generation=1,
        sink=sink,
        sink_clock=audio_sink.SinkClock(),
        segment_table=audio_sink.SegmentTable(),
        decode_queue=asyncio.Queue(),
        _sink_eof_intended=False,
    )

    writer = asyncio.create_task(audio_sink.writer_task(reader, 1))
    await asyncio.sleep(0.05)  # let the writer reach its idle timeout path
    await reader.decode_queue.put(audio_sink.EOF_SENTINEL)

    await asyncio.wait_for(writer, timeout=2.0)

    assert sink.stdin.closed, "writer must close sink stdin on EOF"
    assert sink.stdin.writes == [], "writer must not pad silence after EOF"
    assert reader._sink_eof_intended is True


# ── C3: resume snapshot lifecycle across pause / resume / navigation ──────


def _pause_reader() -> SimpleNamespace:
    return SimpleNamespace(
        pause_toggle_lock=asyncio.Lock(),
        audio_restart_lock=asyncio.Lock(),
        current_pause_toggle_task=None,
        pending_restart_task=None,
        is_paused=False,
        running=True,
        tts_model=object(),
        _resume=None,
        playback_speed=1.0,
    )


class _StaticClock:
    def __init__(self, position: float):
        self._position = position

    def position(self, now: float) -> float:  # noqa: ARG002
        return self._position


class _FakeTimingTTS:
    initialized = True
    output_format = "wav"

    def __init__(self):
        self.calls = 0

    async def generate_audio_with_timing(self, text, path):
        self.calls += 1
        return {"word_timings": [], "speech_duration": 0.5, "total_duration": 0.5}


async def test_pause_resume_keeps_mid_sentence_snapshot(monkeypatch):
    """A pause snapshot survives teardown and is still set when resuming."""
    stop_calls: list[bool] = []
    played: list[bool] = []

    async def fake_stop(reader, *, preserve_resume=False):
        stop_calls.append(preserve_resume)
        if not preserve_resume:
            reader._resume = None

    async def fake_play(reader):
        played.append(True)

    monkeypatch.setattr(audio, "stop_and_clear_audio", fake_stop)
    monkeypatch.setattr(audio, "play_from_current_position", fake_play)

    reader = _pause_reader()
    table = audio_sink.SegmentTable()
    table.open_segment(
        audio_sink.Segment(
            generation=1,
            pos=(0, 0, 1),
            out_start=0.0,
            out_duration=10.0,
            src_duration=10.0,
        )
    )
    reader.sink_clock = _StaticClock(3.27)
    reader.segment_table = table
    reader._snapshot_resume_position = lambda: Lue._snapshot_resume_position(reader)

    # Pause: snapshot the audible offset (about one word into the sentence).
    reader.is_paused = True
    await Lue._handle_pause_toggle(reader)
    assert reader._resume == ((0, 0, 1), 3.27)

    # Resume: the snapshot must not be wiped before the producer consumes it.
    reader.is_paused = False
    await Lue._handle_pause_toggle(reader)
    assert reader._resume == ((0, 0, 1), 3.27)
    assert played == [True]
    assert stop_calls == [True, True]


async def test_resume_snapshot_seeds_first_decode_job(monkeypatch, tmp_path):
    """The producer turns ``_resume`` into the first job's ``src_offset``."""
    tts = _FakeTimingTTS()

    async def fake_duration(path):  # noqa: ARG001
        return 0.5

    monkeypatch.setattr(audio, "get_audio_duration", fake_duration)

    reader = SimpleNamespace(
        tts_model=tts,
        audio_generation=1,
        running=True,
        chapters=[["Alpha beta gamma. Delta epsilon zeta."]],
        chapter_idx=0,
        paragraph_idx=0,
        sentence_idx=1,
        decode_queue=asyncio.Queue(maxsize=4),
        _resume=((0, 0, 1), 3.27),
        _session_dir=str(tmp_path),
        playback_speed=1.0,
    )
    reader._advance_position = MethodType(Lue._advance_position, reader)

    await asyncio.wait_for(audio._producer_loop_sink(reader, 1), timeout=5.0)

    job = reader.decode_queue.get_nowait()
    assert isinstance(job, audio_sink.DecodeJob)
    assert job.pos == (0, 0, 1)
    assert job.src_offset == 3.27, "resume offset lost before decoding"
    assert reader._resume is None, "producer must clear the snapshot it consumed"
    assert reader.decode_queue.get_nowait() is audio_sink.EOF_SENTINEL


def _stoppable_reader() -> SimpleNamespace:
    return SimpleNamespace(
        running=True,
        audio_generation=1,
        producer_task=None,
        writer_task=None,
        decode_queue=asyncio.Queue(),
        audio_queue=asyncio.Queue(),
        playback_processes=[],
        active_playback_tasks=[],
        playback_finished_event=asyncio.Event(),
        _resume=None,
        playback_speed=1.0,
    )


async def test_teardown_preserves_or_drops_resume_according_to_ownership():
    reader = _stoppable_reader()
    reader._resume = ((3, 0, 0), 2.0)

    await audio.stop_and_clear_audio(reader, preserve_resume=True)
    assert reader._resume == ((3, 0, 0), 2.0)

    await audio.stop_and_clear_audio(reader)
    assert reader._resume is None


async def test_switch_book_while_paused_does_not_inherit_resume(monkeypatch, tmp_path):
    """Book switch drops the old snapshot; the new book starts clean."""
    async def fake_duration(path):  # noqa: ARG001
        return 0.5

    monkeypatch.setattr(audio, "get_audio_duration", fake_duration)

    reader = _stoppable_reader()
    reader._resume = ((9, 0, 0), 4.0)  # position in the *old* book

    # ``_switch_book`` performs exactly this teardown before loading content.
    await audio.stop_and_clear_audio(reader)
    assert reader._resume is None

    # New, much shorter book: a stale resume would raise IndexError.
    reader.chapters = [["New book sentence."]]
    reader.chapter_idx = reader.paragraph_idx = reader.sentence_idx = 0
    reader.tts_model = _FakeTimingTTS()
    reader.audio_generation = 2
    reader.decode_queue = asyncio.Queue(maxsize=4)
    reader._session_dir = str(tmp_path)
    reader._advance_position = MethodType(Lue._advance_position, reader)

    await asyncio.wait_for(audio._producer_loop_sink(reader, 2), timeout=5.0)

    job = reader.decode_queue.get_nowait()
    assert job.pos == (0, 0, 0)
    assert job.src_offset == 0.0


async def test_navigation_restart_does_not_inherit_resume(monkeypatch):
    calls: list[bool] = []

    async def fake_stop(reader, *, preserve_resume=False):
        calls.append(preserve_resume)
        if not preserve_resume:
            reader._resume = None

    async def fake_play(reader):
        pass

    monkeypatch.setattr(audio, "stop_and_clear_audio", fake_stop)
    monkeypatch.setattr(audio, "play_from_current_position", fake_play)

    reader = _pause_reader()
    reader._resume = ((5, 0, 0), 1.0)

    await Lue._restart_audio_after_navigation(reader)

    assert calls == [False]
    assert reader._resume is None


# ── W2: sub-word mapping must clamp before the first timing ────────────────


def test_word_index_clamps_before_first_timing():
    """A local_src before timings[0] must map to the first sub-word (0)."""
    reader = SimpleNamespace(current_sentence_words=["Alpha", "beta"])
    seg = audio_sink.Segment(
        generation=1,
        pos=(0, 0, 0),
        out_start=0.0,
        src_duration=3.0,
        word_timings=[("alpha", 1.0, 2.0)],
        word_mapping=[0, 0],  # one TTS word split across two output words
    )

    # Without the max(0, ..) clamp this returns mapped[-1] (1).
    assert Lue._word_index_for_segment(reader, seg, 0.5, 0.5) == 0


# ── W3: pause in silence / at the very end resumes the next sentence ───────


def _resume_reader(chapters):
    reader = SimpleNamespace(
        chapters=chapters, _sink_seg=None, audio_generation=1
    )
    reader._advance_position = MethodType(Lue._advance_position, reader)
    return reader


async def test_pause_in_silence_resumes_next_sentence():
    table = audio_sink.SegmentTable()
    table.open_segment(
        audio_sink.Segment(
            generation=1, pos=(0, 0, 0), out_start=0.0,
            out_duration=2.0, src_duration=2.0,
        )
    )
    table.close_segment()
    table.open_segment(
        audio_sink.Segment(
            generation=1, pos=None, out_start=2.0,
            out_duration=0.5, is_silence=True,
        )
    )

    reader = _resume_reader([["First. Second."]])
    reader.sink_clock = _StaticClock(2.1)
    reader.segment_table = table

    assert Lue._snapshot_resume_position(reader) == ((0, 0, 1), 0.0)


async def test_pause_in_final_50ms_resumes_next_sentence():
    table = audio_sink.SegmentTable()
    table.open_segment(
        audio_sink.Segment(
            generation=1, pos=(0, 0, 0), out_start=0.0,
            out_duration=10.0, src_duration=10.0,
        )
    )

    reader = _resume_reader([["First sentence. Second sentence."]])
    reader.sink_clock = _StaticClock(9.96)  # 40 ms before src_duration
    reader.segment_table = table

    assert Lue._snapshot_resume_position(reader) == ((0, 0, 1), 0.0)


async def test_pause_at_end_of_book_returns_none():
    table = audio_sink.SegmentTable()
    table.open_segment(
        audio_sink.Segment(
            generation=1, pos=(0, 0, 0), out_start=0.0,
            out_duration=10.0, src_duration=10.0,
        )
    )

    reader = _resume_reader([["Only sentence."]])
    reader.sink_clock = _StaticClock(9.96)
    reader.segment_table = table

    assert Lue._snapshot_resume_position(reader) is None


# ── W4: CLI exit codes are propagated (no swallow in cli()) ────────────────


def test_cli_propagates_system_exit(monkeypatch):
    def fake_run(coro):
        coro.close()
        raise SystemExit(2)

    monkeypatch.setattr(cli_main.asyncio, "run", fake_run)

    with pytest.raises(SystemExit) as excinfo:
        cli_main.cli()
    assert excinfo.value.code == 2


def test_cli_returns_cleanly_on_valid_run(monkeypatch):
    def fake_run(coro):
        coro.close()
        return None

    monkeypatch.setattr(cli_main.asyncio, "run", fake_run)

    assert cli_main.cli() is None


@pytest.mark.parametrize("speed", ["0.2", "5", "abc"])
def test_cli_invalid_speed_exits_2(speed):
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "lue", "--speed", speed],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)


# ── W5: a decoder-spawn crash must fail closed, not wedge the sink ─────────


async def test_writer_fails_closed_when_decoder_spawn_fails(monkeypatch, tmp_path):
    async def boom(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(audio_sink.asyncio, "create_subprocess_exec", boom)

    sink = _FakeSink()
    reader = SimpleNamespace(
        audio_generation=1,
        sink=sink,
        sink_clock=audio_sink.SinkClock(),
        segment_table=audio_sink.SegmentTable(),
        decode_queue=asyncio.Queue(),
        _sink_eof_intended=False,
    )
    writer = asyncio.create_task(audio_sink.writer_task(reader, 1))
    await asyncio.sleep(0.05)  # let the writer reach its queue wait
    await reader.decode_queue.put(
        audio_sink.DecodeJob(
            generation=1,
            file_path=str(tmp_path / "seg.wav"),
            pos=(0, 0, 0),
            timing_info={"word_timings": []},
            requested_speed=1.0,
            duration=0.5,
        )
    )

    await asyncio.wait_for(writer, timeout=2.0)

    assert sink.stdin.closed, "writer must close sink stdin when it crashes"
    assert sink.terminated, "writer must terminate the sink so the supervisor restarts"


# ── W2b: restart budget, backoff and fail-open routing ─────────────────────


def _budget_reader() -> SimpleNamespace:
    return SimpleNamespace(_sink_restart_times=[])


def test_restart_budget_allows_max_attempts_then_fails_open():
    """A permanently broken sink may be restarted ``SINK_RESTART_MAX`` times."""
    reader = _budget_reader()
    results = [
        audio._sink_restart_budget(reader)
        for _ in range(config.SINK_RESTART_MAX)
    ]

    assert all(allowed for allowed, _ in results)
    expected = [
        min(config.SINK_RESTART_BACKOFF_S * (2 ** attempt),
            config.SINK_RESTART_BACKOFF_MAX_S)
        for attempt in range(config.SINK_RESTART_MAX)
    ]
    assert [delay for _, delay in results] == pytest.approx(expected)

    allowed, delay = audio._sink_restart_budget(reader)
    assert allowed is False, "4th restart inside the window must fail open"
    assert delay == 0.0


def test_restart_budget_window_expiry_frees_attempts(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        audio, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    reader = _budget_reader()

    for _ in range(config.SINK_RESTART_MAX):
        assert audio._sink_restart_budget(reader)[0] is True
    assert audio._sink_restart_budget(reader)[0] is False

    # Past the sliding window the old attempts no longer count.
    clock["now"] += config.SINK_RESTART_WINDOW_S + 1.0
    assert audio._sink_restart_budget(reader)[0] is True


def test_restart_budget_backoff_is_capped(monkeypatch):
    monkeypatch.setattr(config, "SINK_RESTART_MAX", 6)
    monkeypatch.setattr(config, "SINK_RESTART_BACKOFF_MAX_S", 1.0)
    reader = _budget_reader()

    delays = [audio._sink_restart_budget(reader)[1] for _ in range(6)]
    assert delays[-1] == 1.0
    assert all(delay <= 1.0 for delay in delays)


async def test_fail_open_disables_sink_and_routes_later_starts_to_legacy(monkeypatch):
    """``_fail_open_to_legacy`` must actually change the routing decision."""
    starts: list[str] = []

    async def fake_persistent(reader):  # noqa: ARG001
        starts.append("persistent")

    async def fake_legacy(reader):  # noqa: ARG001
        starts.append("legacy")

    async def fake_restart():
        starts.append("restart")

    monkeypatch.setattr(audio, "_start_persistent", fake_persistent)
    monkeypatch.setattr(audio, "_start_legacy", fake_legacy)
    monkeypatch.setattr(config, "USE_PERSISTENT_SINK", True)

    reader = SimpleNamespace(
        is_paused=False,
        running=True,
        tts_model=object(),
        sink_disabled=False,
        sink=None,
        sink_clock=object(),
        segment_table=audio_sink.SegmentTable(),
        _sink_seg=object(),
        pending_restart_task=None,
        console=None,
        _restart_audio_after_navigation=fake_restart,
    )

    await audio.play_from_current_position(reader)
    assert starts == ["persistent"]
    assert reader.sink_disabled is False

    audio._fail_open_to_legacy(reader)

    assert reader.sink_disabled is True, "fail-open must latch the legacy path"
    assert reader.sink_clock is None and reader.sink is None
    assert reader._sink_seg is None

    await asyncio.wait_for(reader.pending_restart_task, timeout=1.0)
    assert starts == ["persistent", "restart"]

    # Even with USE_PERSISTENT_SINK still true, the next start is legacy.
    await audio.play_from_current_position(reader)
    assert starts == ["persistent", "restart", "legacy"]


def test_sink_disabled_reader_defaults_to_legacy_route(monkeypatch):
    """A reader with no ``sink_disabled`` attribute must still not crash."""
    calls: list[str] = []

    async def fake_persistent(reader):  # noqa: ARG001
        calls.append("persistent")

    async def fake_legacy(reader):  # noqa: ARG001
        calls.append("legacy")

    monkeypatch.setattr(audio, "_start_persistent", fake_persistent)
    monkeypatch.setattr(audio, "_start_legacy", fake_legacy)
    monkeypatch.setattr(config, "USE_PERSISTENT_SINK", True)

    reader = SimpleNamespace(is_paused=False, running=True, tts_model=object())
    asyncio.run(audio.play_from_current_position(reader))
    assert calls == ["persistent"]


# ── W3b: a stuck sink is terminated after EOF (no supervisor hang) ─────────


class _HangingStdin:
    closed = False

    def write(self, data):  # noqa: ARG002
        pass

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        await asyncio.Event().wait()  # never drains


class _StuckSink:
    """ffplay that stops reading stdin: drain and exit both hang."""

    def __init__(self):
        self.stdin = _HangingStdin()
        self.terminated = False

    def close_stdin(self):
        self.stdin.closed = True

    async def wait(self):
        await asyncio.Event().wait()

    async def terminate(self):
        self.terminated = True


async def test_writer_terminates_sink_when_eof_drain_and_exit_hang(monkeypatch):
    monkeypatch.setattr(config, "SINK_EOF_TIMEOUT_S", 0.05)
    monkeypatch.setattr(config, "SINK_TERM_TIMEOUT_S", 0.05)

    sink = _StuckSink()
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(audio_sink.EOF_SENTINEL)
    reader = SimpleNamespace(
        audio_generation=1,
        sink=sink,
        sink_clock=audio_sink.SinkClock(),
        segment_table=audio_sink.SegmentTable(),
        decode_queue=queue,
        _sink_eof_intended=False,
    )

    await asyncio.wait_for(audio_sink.writer_task(reader, 1), timeout=2.0)

    assert sink.stdin.closed, "writer must close stdin on EOF"
    assert sink.terminated, "a sink that never exits must be terminated"
    assert reader._sink_eof_intended is True


# ── W1: sink highlight must survive hyphen-split word mapping ─────────────


def test_sink_word_index_stays_on_hyphenated_token():
    """``well-known`` is one displayed token but two spoken words."""
    reader = SimpleNamespace(
        current_sentence_words=["Alpha", "well-known", "beta"]
    )
    seg = audio_sink.Segment(
        generation=1,
        pos=(0, 0, 0),
        out_start=0.0,
        src_duration=4.0,
        word_timings=[
            ("alpha", 0.0, 1.0),
            ("well", 1.0, 2.0),
            ("known", 2.0, 3.0),
            ("beta", 3.0, 4.0),
        ],
        # Only the first spoken word of each token carries its entry.
        word_mapping=[0, 1, 3],
    )

    # 2.5 s is inside "known" (spoken idx 2), which no token owns directly.
    assert Lue._word_index_for_segment(reader, seg, 2.5, 2.5) == 1

    # The legacy selector agrees, so both paths highlight the same token.
    assert select_highlight_word(
        word_timings=seg.word_timings,
        word_mapping=seg.word_mapping,
        elapsed=2.5,
        total_words=3,
    ) == 1


# ── S4: the cached highlight segment is generation-scoped ─────────────────


async def test_snapshot_ignores_sink_seg_from_previous_generation():
    table = audio_sink.SegmentTable()
    table.open_segment(
        audio_sink.Segment(
            generation=2, pos=None, out_start=0.0,
            out_duration=0.5, is_silence=True,
        )
    )

    reader = _resume_reader([["Only sentence."]])
    reader.audio_generation = 2
    # Cached while the previous generation played: position (9, 0, 0) does
    # not even exist in this book, so using it would raise IndexError.
    reader._sink_seg = audio_sink.Segment(
        generation=1, pos=(9, 0, 0), out_start=0.0, out_duration=1.0,
    )
    reader.sink_clock = _StaticClock(0.1)
    reader.segment_table = table

    assert Lue._snapshot_resume_position(reader) is None


async def test_snapshot_uses_current_generation_sink_seg():
    table = audio_sink.SegmentTable()
    table.open_segment(
        audio_sink.Segment(
            generation=2, pos=(0, 0, 0), out_start=0.0,
            out_duration=2.0, src_duration=2.0,
        )
    )
    table.close_segment()
    table.open_segment(
        audio_sink.Segment(
            generation=2, pos=None, out_start=2.0,
            out_duration=0.5, is_silence=True,
        )
    )

    reader = _resume_reader([["First. Second."]])
    reader.audio_generation = 2
    reader._sink_seg = audio_sink.Segment(
        generation=2, pos=(0, 0, 0), out_start=0.0, out_duration=2.0,
    )
    reader.sink_clock = _StaticClock(2.1)
    reader.segment_table = table

    assert Lue._snapshot_resume_position(reader) == ((0, 0, 1), 0.0)


async def test_teardown_and_consume_reset_cached_sink_seg(monkeypatch, tmp_path):
    seg = audio_sink.Segment(
        generation=1, pos=(0, 0, 0), out_start=0.0, out_duration=1.0,
    )

    # Book switch / navigation / pause teardown.
    reader = _stoppable_reader()
    reader._sink_seg = seg
    await audio.stop_and_clear_audio(reader)
    assert reader._sink_seg is None, "teardown must drop the cached segment"

    # Producer consuming a resume snapshot.
    async def fake_duration(path):  # noqa: ARG001
        return 0.5

    monkeypatch.setattr(audio, "get_audio_duration", fake_duration)
    reader = SimpleNamespace(
        tts_model=_FakeTimingTTS(),
        audio_generation=1,
        running=True,
        chapters=[["Alpha. Beta."]],
        chapter_idx=0,
        paragraph_idx=0,
        sentence_idx=0,
        decode_queue=asyncio.Queue(maxsize=4),
        _resume=((0, 0, 1), 0.5),
        _sink_seg=seg,
        _session_dir=str(tmp_path),
        playback_speed=1.0,
    )
    reader._advance_position = MethodType(Lue._advance_position, reader)

    await asyncio.wait_for(audio._producer_loop_sink(reader, 1), timeout=5.0)
    assert reader._sink_seg is None, "consuming the snapshot starts a new generation"


# ── S5: a second pause while buffering keeps the valid snapshot ────────────


async def test_second_pause_while_buffering_keeps_valid_snapshot(monkeypatch):
    async def fake_stop(reader, *, preserve_resume=False):
        if not preserve_resume:
            reader._resume = None

    async def fake_play(reader):  # noqa: ARG001
        pass

    monkeypatch.setattr(audio, "stop_and_clear_audio", fake_stop)
    monkeypatch.setattr(audio, "play_from_current_position", fake_play)

    reader = _pause_reader()
    valid = ((0, 0, 1), 3.27)
    reader._resume = valid
    reader.sink_clock = None
    reader.segment_table = None
    # Buffering: the resumed generation has no segment on the clock yet, so
    # the snapshot is legitimately None.
    reader._snapshot_resume_position = lambda: None

    reader.is_paused = True
    await Lue._handle_pause_toggle(reader)

    assert reader._resume == valid, "a buffering pause must not wipe the snapshot"


# ── S6: Ctrl-C is a clean exit, not a fatal startup error ─────────────────


def test_cli_converts_keyboard_interrupt_to_exit_130(monkeypatch, caplog):
    def fake_run(coro):
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_main.asyncio, "run", fake_run)

    with caplog.at_level("DEBUG"):
        with pytest.raises(SystemExit) as excinfo:
            cli_main.cli()

    assert excinfo.value.code == 130
    assert "Fatal error" not in caplog.text, "Ctrl-C must not log a fatal error"
