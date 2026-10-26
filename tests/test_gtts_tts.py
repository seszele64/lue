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
            calls.update(text=text, lang=lang, slow=slow, tld=tld, **kwargs)

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


def test_tld_pattern_accepts_google_domains():
    """The accepted shape is ``com``, ``co.uk``, ``co.in``, ``de`` — not 'comm'."""
    from lue.tts.gtts_tts import _TLD_PATTERN

    for ok in ("com", "co.uk", "co.in", "de", "fr", "us"):
        assert _TLD_PATTERN.match(ok), ok

    for bad in ("", "c", "comm", "c0m", "google.com", "co.uk.", ".com"):
        assert not _TLD_PATTERN.match(bad), bad


@pytest.mark.asyncio
@pytest.mark.parametrize("tld", ["com", "co.uk", "co.in", "de"])
async def test_initialize_accepts_valid_tld(monkeypatch, mock_console, tld):
    monkeypatch.setattr(config, "GTTS_TLD", tld)
    tts = GTTTTS(mock_console)
    _install_fake_gtts(monkeypatch)

    assert await tts.initialize() is True
    assert tts.tld == tld


def test_tld_is_normalized(monkeypatch, mock_console):
    """Whitespace/case in LUE_GTTS_TLD must not reach the request URL verbatim."""
    monkeypatch.setattr(config, "GTTS_TLD", "  CO.UK  ")
    assert GTTTTS(mock_console).tld == "co.uk"


@pytest.mark.asyncio
async def test_initialize_rejects_malformed_tld(monkeypatch):
    """A TLD typo must fail once at startup instead of on every sentence."""
    console = Console(record=True, width=200)
    monkeypatch.setattr(config, "GTTS_TLD", "comm")
    tts = GTTTTS(console)
    _install_fake_gtts(monkeypatch)

    assert await tts.initialize() is False
    assert tts.initialized is False
    text = console.export_text()
    assert "not a valid gTTS top-level domain" in text
    assert "co.uk" in text


def test_chunk_count_matches_gtts_chunking():
    """gTTS requests one chunk per ~100 characters (minimum one)."""
    from lue.tts.gtts_tts import _chunk_count

    assert _chunk_count("") == 1
    assert _chunk_count("Hello.") == 1
    assert _chunk_count("a" * 100) == 1
    assert _chunk_count("a" * 101) == 2
    assert _chunk_count("a" * 1000) == 10


def test_total_timeout_scales_per_chunk(monkeypatch, tts):
    """The outer wait_for budget is per-request timeout × chunks + margin."""
    from lue.tts import gtts_tts

    monkeypatch.setattr(config, "GTTS_TIMEOUT", 2.0)
    monkeypatch.setattr(gtts_tts, "_GTTS_TOTAL_TIMEOUT_MARGIN", 5.0)

    assert tts._total_timeout("Hello.") == pytest.approx(7.0)
    # 10 chunks × 2s + 5s margin: a long sentence is no longer capped at 2s.
    assert tts._total_timeout("a" * 1000) == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_long_text_is_not_aborted_by_per_request_timeout(
    monkeypatch, mock_console, tmp_output_dir
):
    """A sentence needing several chunk requests may exceed one timeout."""
    tts = GTTTTS(mock_console)
    module = types.ModuleType("gtts")

    class gTTS:
        def __init__(self, text=None, lang=None, slow=False, tld=None, **kwargs):
            pass

        def write_to_fp(self, fp):
            time.sleep(0.4)  # > GTTS_TIMEOUT, < total budget for 10 chunks
            fp.write(b"long")

    module.gTTS = gTTS
    monkeypatch.setitem(sys.modules, "gtts", module)
    monkeypatch.setattr(config, "GTTS_TIMEOUT", 0.1)

    assert await tts.initialize() is True
    out = tmp_output_dir / "long.mp3"
    await tts.generate_audio("a" * 1000, str(out))

    assert out.read_bytes() == b"long"


@pytest.mark.asyncio
async def test_generate_audio_timeout(monkeypatch, tmp_output_dir):
    """A hung chunk must time out instead of stalling the producer."""
    from lue.tts import gtts_tts

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
    # Drop the fixed headroom so the test does not wait 10s for the budget.
    monkeypatch.setattr(gtts_tts, "_GTTS_TOTAL_TIMEOUT_MARGIN", 0.0)

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
    # The per-request (per-chunk) timeout is handed to gTTS itself.
    assert calls["timeout"] == config.GTTS_TIMEOUT


@pytest.mark.asyncio
async def test_generate_audio_requires_initialization(tts, tmp_output_dir):
    with pytest.raises(RuntimeError):
        await tts.generate_audio("hi", str(tmp_output_dir / "x.mp3"))


def test_discovered_by_manager():
    from lue.tts_manager import TTSManager

    assert "gtts" in TTSManager().get_available_tts_names()
