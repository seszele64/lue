"""Regression tests for the persistent-sink / decode command builders.

These guard against the ffmpeg n9.0.2 regression where the removed ``-ac``
option made both the ``ffplay`` sink and the ``ffmpeg`` decoder fail at
startup (silent TTS).  See diagnosis: ``lue/audio_sink.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from lue import audio, audio_sink, config


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
    return SimpleNamespace(
        console=Mock(),
        audio_generation=1,
        producer_task=None,
        writer_task=None,
        player_task=None,
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
        def __init__(self, gen):
            self.generation = gen
            self.proc = SimpleNamespace(returncode=1)

        async def start(self):
            return self

    monkeypatch.setattr(audio_sink, "SinkProcess", DeadSink)
    monkeypatch.setattr(audio, "_start_legacy", fake_start_legacy)
    monkeypatch.setattr(config, "SINK_STARTUP_GRACE_S", 0.0)
    monkeypatch.setattr(audio_sink, "make_session_dir", lambda gen: str(tmp_path))

    reader = _fake_reader()
    await audio._start_persistent(reader)

    assert fallback["called"] is True
    assert reader.sink is None
    reader.console.print.assert_called()


async def test_start_persistent_wires_tasks_when_sink_alive(monkeypatch, tmp_path):
    fallback = {"called": False}

    async def fake_start_legacy(reader):
        fallback["called"] = True

    class LiveSink:
        def __init__(self, gen):
            self.generation = gen
            self.proc = SimpleNamespace(returncode=None)
            self.stdin = None

        async def start(self):
            return self

        def close_stdin(self):
            pass

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

    for attr in ("writer_task", "producer_task", "player_task"):
        task = getattr(reader, attr)
        assert task is not None
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass