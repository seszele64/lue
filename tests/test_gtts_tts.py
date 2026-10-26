"""Minimal tests for the free Google Translate (``gTTS``) TTS provider.

``gTTS`` is an optional dependency, so a fake module is injected into
``sys.modules`` to exercise the generation path without network access.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest
from rich.console import Console

from lue import config
from lue.tts.gtts_tts import GTTTTS


def _install_fake_gtts(monkeypatch, payload: bytes = b"\xff\xfbGTTSPAYLOAD") -> dict:
    """Inject a minimal stand-in for the ``gTTS`` package and return its calls."""
    module = types.ModuleType("gtts")
    calls: dict = {}

    class gTTS:
        def __init__(self, text=None, lang=None, slow=False, tld=None, **kwargs):
            calls.update(text=text, lang=lang, slow=slow, tld=tld)

        def write_to_fp(self, fp):
            fp.write(payload)

    module.gTTS = gTTS
    monkeypatch.setitem(sys.modules, "gtts", module)
    return calls


@pytest.fixture
def tts(mock_console) -> GTTTTS:
    return GTTTTS(mock_console)


def test_class_contract(tts):
    assert tts.name == "gtts"
    assert tts.output_format == "mp3"
    assert tts.supports_word_timing is False
    assert tts.lang == "en"
    assert tts.initialized is False


def test_lang_arg_overrides(mock_console):
    assert GTTTTS(mock_console, lang="fr").lang == "fr"


def test_voice_arg_used_as_lang(mock_console):
    assert GTTTTS(mock_console, voice="de").lang == "de"


def test_lang_falls_back_to_config(monkeypatch, mock_console):
    monkeypatch.setattr(config, "GTTS_LANG", "ja")
    assert GTTTTS(mock_console).lang == "ja"


def test_config_gtts_lang_precedence():
    """LUE_GTTS_LANG > TTS_VOICES["gtts"] > TTS_LANGUAGE_CODES["gtts"] > 'en'."""
    from lue.config import _resolve_gtts_lang

    voices = {"gtts": "fr"}
    codes = {"gtts": "it"}
    assert _resolve_gtts_lang("de", voices, codes) == "de"
    assert _resolve_gtts_lang("  de  ", voices, codes) == "de"
    assert _resolve_gtts_lang("", voices, codes) == "fr"
    assert _resolve_gtts_lang("   ", voices, codes) == "fr"
    assert _resolve_gtts_lang("", {}, codes) == "it"
    assert _resolve_gtts_lang("", {}, {}) == "en"


def test_config_gtts_lang_is_wired_from_voices():
    """The module-level default honours TTS_VOICES (PR8 finding 4)."""
    assert config.GTTS_LANG == (config.TTS_VOICES.get("gtts") or "en")


def test_get_overlap_seconds(tts):
    # Crossfade/overlap was removed (spec-v3 §9.1): the accessor returns None.
    assert tts.get_overlap_seconds() is None


def test_get_overlap_seconds_logs_deprecation_warning(caplog):
    from lue.tts import base

    base._overlap_warned = False  # reset the one-time flag for this test
    tts = GTTTTS.__new__(GTTTTS)  # no console needed for the accessor
    with caplog.at_level("WARNING", logger="lue.tts.base"):
        assert tts.get_overlap_seconds() is None
    assert any(
        "overlap" in record.message.lower() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_initialize_without_gtts(monkeypatch, tts):
    monkeypatch.setitem(sys.modules, "gtts", None)
    assert await tts.initialize() is False
    assert tts.initialized is False


@pytest.mark.asyncio
async def test_initialize_missing_dep_hint_renders_brackets(monkeypatch):
    """The install hint must render the literal ``lue-reader[free]`` extra."""
    console = Console(record=True, width=200)
    tts = GTTTTS(console)
    monkeypatch.setitem(sys.modules, "gtts", None)

    assert await tts.initialize() is False
    assert "lue-reader[free]" in console.export_text()


@pytest.mark.asyncio
async def test_initialize_with_gtts(monkeypatch, tts):
    _install_fake_gtts(monkeypatch)
    assert await tts.initialize() is True
    assert tts.initialized is True


@pytest.mark.asyncio
async def test_initialize_rejects_malformed_lang(monkeypatch):
    """A bad --voice must fail fast instead of retrying every sentence."""
    console = Console(record=True, width=200)
    tts = GTTTTS(console, voice="en-US-JennyNeural")
    _install_fake_gtts(monkeypatch)

    assert await tts.initialize() is False
    assert tts.initialized is False
    assert "not a valid gTTS language code" in console.export_text()


@pytest.mark.asyncio
async def test_initialize_rejects_empty_lang(monkeypatch, mock_console):
    """Defensive guard: a blank language code must never reach the network."""
    tts = GTTTTS(mock_console)
    tts.lang = ""  # e.g. config.GTTS_LANG blanked at runtime
    _install_fake_gtts(monkeypatch)

    assert await tts.initialize() is False
    assert tts.initialized is False


def test_blank_lang_arg_falls_back_to_config(monkeypatch, mock_console):
    monkeypatch.setattr(config, "GTTS_LANG", "ja")
    assert GTTTTS(mock_console, lang="   ").lang == "ja"
    assert GTTTTS(mock_console, voice="").lang == "ja"


@pytest.mark.asyncio
async def test_initialize_rejects_unsupported_lang(monkeypatch):
    """gTTS itself rejects unknown codes (ValueError) — surface it at startup."""
    console = Console(record=True, width=200)
    tts = GTTTTS(console, lang="zz")
    module = types.ModuleType("gtts")

    class gTTS:
        def __init__(self, text=None, lang=None, slow=False, tld=None, **kwargs):
            raise ValueError(f"Language not supported: {lang}")

    module.gTTS = gTTS
    monkeypatch.setitem(sys.modules, "gtts", module)

    assert await tts.initialize() is False
    assert tts.initialized is False
    assert "does not support language" in console.export_text()


@pytest.mark.asyncio
async def test_initialize_normalizes_deprecated_lang(monkeypatch, mock_console):
    """'en-uk' is deprecated; gTTS normalises it to 'en' — we adopt that."""
    tts = GTTTTS(mock_console, lang="en-uk")
    module = types.ModuleType("gtts")

    class gTTS:
        def __init__(self, text=None, lang=None, slow=False, tld=None, **kwargs):
            self.lang = "en"

        def write_to_fp(self, fp):
            fp.write(b"x")

    module.gTTS = gTTS
    monkeypatch.setitem(sys.modules, "gtts", module)

    assert await tts.initialize() is True
    assert tts.lang == "en"
    assert tts.voice == "en"


@pytest.mark.asyncio
async def test_generate_audio_timeout(monkeypatch, tmp_output_dir):
    """A hung network call must time out instead of stalling the producer."""
    console = Console(record=True, width=200)
    tts = GTTTTS(console)
    module = types.ModuleType("gtts")

    class gTTS:
        def __init__(self, text=None, lang=None, slow=False, tld=None, **kwargs):
            pass

        def write_to_fp(self, fp):
            time.sleep(1.0)  # simulate a hung request
            fp.write(b"late")

    module.gTTS = gTTS
    monkeypatch.setitem(sys.modules, "gtts", module)
    monkeypatch.setattr(config, "GTTS_TIMEOUT", 0.05)

    assert await tts.initialize() is True
    out = tmp_output_dir / "timeout.mp3"

    with pytest.raises(asyncio.TimeoutError):
        await tts.generate_audio("Hello.", str(out))

    # A timed-out attempt must not leave a partial/stale file behind.
    assert not out.exists()


@pytest.mark.asyncio
async def test_generate_audio(monkeypatch, tts, tmp_output_dir):
    calls = _install_fake_gtts(monkeypatch, payload=b"GTTS-AUDIO")
    assert await tts.initialize() is True

    out = tmp_output_dir / "out.mp3"
    await tts.generate_audio("Hello there.", str(out))

    assert out.read_bytes() == b"GTTS-AUDIO"
    assert calls["text"] == "Hello there."
    assert calls["lang"] == "en"


@pytest.mark.asyncio
async def test_generate_audio_requires_initialization(tts, tmp_output_dir):
    with pytest.raises(RuntimeError):
        await tts.generate_audio("hi", str(tmp_output_dir / "x.mp3"))


def test_discovered_by_manager():
    from lue.tts_manager import TTSManager

    assert "gtts" in TTSManager().get_available_tts_names()
