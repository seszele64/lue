"""Offline unit tests for the CapCut TTS provider.

``capcut-tts-api`` is an optional, git-only dependency, so a fake module is
injected into ``sys.modules`` for the initialization tests, and a fake
``requests`` module stands in for the download step.  No test touches the
network.
"""

from __future__ import annotations

import json
import random
import sys
import types
from unittest.mock import Mock

import pytest

from lue.tts.capcut_tts import (
    CapCutTaskError,
    CapCutTTS,
    SignError,
)
from lue.tts_manager import TTSManager


# ── Fakes ──────────────────────────────────────────────────────────────────

class FakeCapCutClient:
    """Configurable stand-in for ``capcut_tts_api.CapCutClient``."""

    def __init__(
        self,
        *,
        session=True,
        voices=None,
        create=None,
        query=None,
        fail_create_times=0,
        create_exc=None,
        query_exc=None,
    ):
        self.session = object() if session else None
        self._voices = voices if voices is not None else [{"id": "BV074_streaming"}]
        self._create = create if create is not None else {
            "ret": "0",
            "data": {"id": "task-1", "token": "tok-1"},
        }
        self._query = query if query is not None else {
            "ret": "0",
            "data": {
                "status": "succeed",
                "payload": json.dumps(
                    {"audio_subtitles": [{"speech_url": "https://cdn.test/a.mp3"}]}
                ),
            },
        }
        self.fail_create_times = fail_create_times
        self.create_exc = create_exc
        self.query_exc = query_exc
        self.create_calls = 0
        self.query_calls = 0
        # Every positional/keyword arg of each ``create_tts_task`` call, so the
        # tests can assert the resource_id/rate plumbing is correct.
        self.create_args: list[dict] = []

    def list_voices(self):
        return self._voices

    def create_tts_task(self, text, voice="BV074_streaming", resource_id=None, rate="1.0"):
        # Mirrors capcut_tts_api.CapCutClient.create_tts_task's signature.
        self.create_calls += 1
        self.create_args.append(
            {"text": text, "voice": voice, "resource_id": resource_id, "rate": rate}
        )
        if self.fail_create_times and self.create_calls <= self.fail_create_times:
            raise self.create_exc or ConnectionError("transient")
        return self._create

    def query_tts_task(self, task_id, token):
        self.query_calls += 1
        if self.query_exc is not None:
            raise self.query_exc
        return self._query


def install_fake_sdk(monkeypatch, client) -> None:
    """Inject a minimal ``capcut_tts_api`` module returning ``client``."""
    module = types.ModuleType("capcut_tts_api")
    module.CapCutClient = lambda *args, **kwargs: client
    monkeypatch.setitem(sys.modules, "capcut_tts_api", module)


#: Minimal bytes that satisfy the provider's MP3 magic check (ID3 tag).
MP3_MAGIC = b"ID3\x03\x00\x00\x00\x00\x00\x00"


def install_fake_requests(
    monkeypatch, payload: bytes = MP3_MAGIC, status: int = 200, headers: dict | None = None
) -> dict:
    """Inject a minimal synchronous ``requests`` module and return its calls."""
    module = types.ModuleType("requests")
    calls: dict = {"url": None}

    class RequestException(Exception):
        pass

    class ConnectionError(RequestException):  # noqa: A001 - mirrors requests
        pass

    class Timeout(RequestException):
        pass

    class HTTPError(RequestException):
        pass

    module.exceptions = types.SimpleNamespace(
        RequestException=RequestException,
        ConnectionError=ConnectionError,
        Timeout=Timeout,
        HTTPError=HTTPError,
    )

    class FakeResponse:
        def __init__(self):
            self.status_code = status
            self.headers = dict(headers or {})

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            if status >= 400:
                raise HTTPError(f"HTTP {status}")

        def iter_content(self, chunk_size=8192):
            yield payload

    def get(url, stream=True, timeout=None):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    module.get = get
    monkeypatch.setitem(sys.modules, "requests", module)
    return calls


@pytest.fixture
def tts(mock_console) -> CapCutTTS:
    return CapCutTTS(mock_console)


# ── Class contract ─────────────────────────────────────────────────────────

def test_class_contract(tts):
    assert tts.name == "capcut"
    assert tts.output_format == "mp3"
    assert tts.supports_word_timing is False
    assert tts.initialized is False
    assert tts.voice == "BV074_streaming"


def test_voice_arg_overrides(mock_console):
    assert CapCutTTS(mock_console, voice="custom_voice").voice == "custom_voice"


@pytest.mark.asyncio
async def test_generate_audio_requires_initialization(tts, tmp_output_dir):
    with pytest.raises(RuntimeError):
        await tts.generate_audio("hi", str(tmp_output_dir / "x.mp3"))


@pytest.mark.asyncio
async def test_get_raw_timing_data_empty(tts):
    assert await tts.get_raw_timing_data("hello world", "/tmp/x.mp3") == []


# ── Initialization ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_initialize_missing_package(monkeypatch, tts):
    monkeypatch.setitem(sys.modules, "capcut_tts_api", None)
    assert await tts.initialize() is False
    assert tts.initialized is False


@pytest.mark.asyncio
async def test_initialize_session_none(monkeypatch, tts):
    install_fake_sdk(monkeypatch, FakeCapCutClient(session=False))
    assert await tts.initialize() is False
    assert tts.initialized is False


@pytest.mark.asyncio
async def test_initialize_success(monkeypatch, tts):
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    assert await tts.initialize() is True
    assert tts.initialized is True
    assert tts._client is client
    assert tts.voice == "BV074_streaming"


@pytest.mark.asyncio
async def test_initialize_warns_on_empty_voice_list(monkeypatch, mock_console):
    client = FakeCapCutClient(voices=[])
    install_fake_sdk(monkeypatch, client)
    instance = CapCutTTS(mock_console, voice="custom_voice")
    assert await instance.initialize() is True
    assert instance.initialized is True
    # A warning must have been printed for the empty catalogue.
    assert mock_console.print.called


# ── Generation ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_writes_bytes(monkeypatch, mock_console, tmp_output_dir):
    payload = b"ID3" + b"CAPCUT-BYTES"
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    req_calls = install_fake_requests(monkeypatch, payload=payload)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    await instance.generate_audio("Hello there.", str(out))

    assert out.read_bytes() == payload
    assert req_calls["url"] == "https://cdn.test/a.mp3"
    assert client.create_calls == 1
    assert client.query_calls == 1
    # No leftover per-attempt temp files.
    assert not list(tmp_output_dir.glob("*.part"))


@pytest.mark.asyncio
async def test_create_tts_task_plumbs_voice_resource_rate(monkeypatch, mock_console, tmp_output_dir):
    """C1: resource_id/rate must be passed by keyword, not positionally."""
    monkeypatch.setattr("lue.config.CAPCUT_RESOURCE_ID", "res-42")
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RATE", "1.25")

    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch)

    instance = CapCutTTS(mock_console, voice="BV999_voice")
    assert await instance.initialize() is True
    assert instance.resource_id == "res-42"

    await instance.generate_audio("Hello.", str(tmp_output_dir / "out.mp3"))

    assert client.create_args == [
        {
            "text": "Hello.",
            "voice": "BV999_voice",
            "resource_id": "res-42",
            "rate": "1.25",
        }
    ]


@pytest.mark.asyncio
async def test_sdk_shaped_tasks_payload(monkeypatch, mock_console, tmp_output_dir):
    """C2: id/token/status live in data["tasks"][0] per the SDK's generate_speech."""
    create = {"ret": 0, "data": {"tasks": [{"id": "sdk-task", "token": "sdk-tok"}]}}
    query = {
        "ret": 0,
        "data": {
            "tasks": [
                {
                    "status": "succeed",
                    "audio_subtitles": [{"speech_url": "https://cdn.test/sdk.mp3"}],
                }
            ]
        },
    }
    client = FakeCapCutClient(create=create, query=query)
    install_fake_sdk(monkeypatch, client)
    req_calls = install_fake_requests(monkeypatch, payload=b"ID3SDK")

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "sdk.mp3"
    await instance.generate_audio("Hi.", str(out))

    assert out.read_bytes() == b"ID3SDK"
    assert req_calls["url"] == "https://cdn.test/sdk.mp3"
    assert client.query_calls == 1


@pytest.mark.asyncio
async def test_success_spelling_accepted(monkeypatch, mock_console, tmp_output_dir):
    """C2: the backend says "succeed", but "success" must also count."""
    query = {
        "ret": 0,
        "data": {
            "tasks": [
                {
                    "status": "success",
                    "audio_subtitles": [{"speech_url": "https://cdn.test/s2.mp3"}],
                }
            ]
        },
    }
    client = FakeCapCutClient(query=query)
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=b"ID3S2")

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "s2.mp3"
    await instance.generate_audio("Hi.", str(out))
    assert out.read_bytes() == b"ID3S2"


@pytest.mark.asyncio
async def test_download_rejects_non_https(monkeypatch, mock_console, tmp_output_dir):
    query = {
        "ret": 0,
        "data": {"status": "succeed", "payload": json.dumps(
            {"audio_subtitles": [{"speech_url": "http://cdn.test/a.mp3"}]}
        )},
    }
    client = FakeCapCutClient(query=query)
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    with pytest.raises(ValueError):
        await instance.generate_audio("Hi.", str(out))
    # Non-HTTPS is terminal: one submit, nothing published.
    assert client.create_calls == 1
    assert not out.exists()


@pytest.mark.asyncio
async def test_download_rejects_non_mp3(monkeypatch, mock_console, tmp_output_dir):
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=b"<html>not audio</html>")

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    with pytest.raises(ValueError):
        await instance.generate_audio("Hi.", str(out))
    assert not out.exists()
    assert not list(tmp_output_dir.glob("*.part"))


@pytest.mark.asyncio
async def test_download_enforces_size_cap(monkeypatch, mock_console, tmp_output_dir):
    monkeypatch.setattr("lue.tts.capcut_tts._MAX_DOWNLOAD_BYTES", 8)
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=b"X" * 100)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    with pytest.raises(ValueError):
        await instance.generate_audio("Hi.", str(out))
    assert not out.exists()


class TestSdkParsers:
    def test_extract_task_ids_from_tasks(self):
        resp = {"ret": 0, "data": {"tasks": [{"id": "abc", "token": "def"}]}}
        assert CapCutTTS._extract_task_ids(resp) == ("abc", "def")

    def test_extract_status_from_tasks(self):
        resp = {"ret": 0, "data": {"tasks": [{"status": "Succeed"}]}}
        assert CapCutTTS._extract_status(resp) == "succeed"

    def test_extract_payload_from_tasks(self):
        resp = {
            "ret": 0,
            "data": {"tasks": [{"audio_subtitles": [{"speech_url": "https://x/y.mp3"}]}]},
        }
        payload = CapCutTTS._extract_payload(resp)
        assert CapCutTTS._extract_audio_url(payload) == "https://x/y.mp3"


@pytest.mark.asyncio
async def test_generate_no_url_is_terminal(monkeypatch, mock_console, tmp_output_dir):
    query = {
        "ret": "0",
        "data": {"status": "succeed", "payload": json.dumps({"audio_subtitles": []})},
    }
    client = FakeCapCutClient(query=query)
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    with pytest.raises(RuntimeError):
        await instance.generate_audio("Hello.", str(tmp_output_dir / "out.mp3"))

    # Terminal: exactly one attempt, no retries.
    assert client.create_calls == 1


@pytest.mark.asyncio
async def test_retry_then_success(monkeypatch, mock_console, tmp_output_dir):
    monkeypatch.setattr("lue.config.CAPCUT_TTS_MAX_RETRIES", 2)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RETRY_BASE_DELAY", 0.1)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_TIMEOUT", 5)
    monkeypatch.setattr(random, "random", Mock(return_value=0.5))

    payload = b"ID3" + b"RETRIED"
    client = FakeCapCutClient(fail_create_times=2)
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=payload)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    await instance.generate_audio("Hello.", str(out))

    assert out.read_bytes() == payload
    assert client.create_calls == 3  # two transient failures, then success
    assert not list(tmp_output_dir.glob("*.part"))


@pytest.mark.asyncio
async def test_retries_exhausted(monkeypatch, mock_console, tmp_output_dir):
    monkeypatch.setattr("lue.config.CAPCUT_TTS_MAX_RETRIES", 2)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RETRY_BASE_DELAY", 0.1)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_TIMEOUT", 5)
    monkeypatch.setattr(random, "random", Mock(return_value=0.5))

    client = FakeCapCutClient(fail_create_times=999)
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    with pytest.raises(ConnectionError):
        await instance.generate_audio("Hello.", str(tmp_output_dir / "out.mp3"))

    assert client.create_calls == 3  # attempts 0, 1, 2


@pytest.mark.asyncio
async def test_task_error_retried_then_success(monkeypatch, mock_console, tmp_output_dir):
    monkeypatch.setattr("lue.config.CAPCUT_TTS_MAX_RETRIES", 2)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RETRY_BASE_DELAY", 0.1)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_TIMEOUT", 5)
    monkeypatch.setattr(random, "random", Mock(return_value=0.5))

    payload = b"ID3" + b"OK"
    client = FakeCapCutClient(fail_create_times=1, create_exc=CapCutTaskError("boom"))
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=payload)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    await instance.generate_audio("Hello.", str(out))
    assert out.read_bytes() == payload
    assert client.create_calls == 2


# ── URL extraction ─────────────────────────────────────────────────────────

class TestExtractAudioUrl:
    def test_direct_speech_url(self):
        assert (
            CapCutTTS._extract_audio_url({"speech_url": "https://a.test/x.mp3"})
            == "https://a.test/x.mp3"
        )

    def test_camel_case_and_alt_keys(self):
        assert CapCutTTS._extract_audio_url({"audioUrl": "https://a.test/y.mp3"}) == (
            "https://a.test/y.mp3"
        )
        assert CapCutTTS._extract_audio_url({"url": "https://a.test/z.mp3"}) == (
            "https://a.test/z.mp3"
        )

    def test_nested_list_dict(self):
        payload = {"audio_subtitles": [{"speech_url": "https://a.test/n.mp3"}]}
        assert CapCutTTS._extract_audio_url(payload) == "https://a.test/n.mp3"

    def test_deeply_nested(self):
        payload = {"data": {"result": {"items": [{"speech_url": "https://a.test/d.mp3"}]}}}
        assert CapCutTTS._extract_audio_url(payload) == "https://a.test/d.mp3"

    def test_json_encoded_string(self):
        encoded = json.dumps({"audio_subtitles": [{"speech_url": "https://a.test/j.mp3"}]})
        assert CapCutTTS._extract_audio_url(encoded) == "https://a.test/j.mp3"

    def test_direct_url_string(self):
        assert CapCutTTS._extract_audio_url("https://a.test/direct.mp3") == (
            "https://a.test/direct.mp3"
        )

    def test_no_url_returns_none(self):
        assert CapCutTTS._extract_audio_url({"audio_subtitles": []}) is None
        assert CapCutTTS._extract_audio_url({"foo": "bar"}) is None
        assert CapCutTTS._extract_audio_url("not-a-url") is None
        assert CapCutTTS._extract_audio_url(123) is None


# ── Retry classification ───────────────────────────────────────────────────

class TestShouldRetry:
    def test_retryable(self):
        assert CapCutTTS._should_retry(ConnectionError()) is True
        assert CapCutTTS._should_retry(TimeoutError()) is True
        assert CapCutTTS._should_retry(CapCutTaskError("failed")) is True

        exc_429 = RuntimeError("rate limited")
        exc_429.status_code = 429
        assert CapCutTTS._should_retry(exc_429) is True

        exc_503 = RuntimeError("server")
        exc_503.status_code = 503
        assert CapCutTTS._should_retry(exc_503) is True

    def test_terminal(self):
        assert CapCutTTS._should_retry(SignError("bad sign")) is False
        assert CapCutTTS._should_retry(ValueError("bad value")) is False
        assert CapCutTTS._should_retry(RuntimeError("no audio URL")) is False

        exc_400 = RuntimeError("bad request")
        exc_400.status_code = 400
        assert CapCutTTS._should_retry(exc_400) is False

    def test_name_based_fallback(self):
        FakeSign = type("SignError", (Exception,), {})
        assert CapCutTTS._should_retry(FakeSign()) is False

        FakeTimeout = type("ReadTimeout", (Exception,), {})
        assert CapCutTTS._should_retry(FakeTimeout()) is True


# ── Discovery ──────────────────────────────────────────────────────────────

def test_discovered_by_manager():
    # The SDK is not installed in CI; discovery must still register "capcut".
    assert "capcut" in TTSManager().get_available_tts_names()


# ── Config validation ──────────────────────────────────────────────────────

def test_config_rate_is_clamped_string():
    from lue import config

    assert isinstance(config.CAPCUT_TTS_RATE, str)
    assert 0.5 <= float(config.CAPCUT_TTS_RATE) <= 2.0


def test_config_rate_clamps_out_of_range(monkeypatch):
    import importlib

    from lue import config

    monkeypatch.setenv("LUE_CAPCUT_TTS_RATE", "9.9")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.CAPCUT_TTS_RATE == "2.0"
    finally:
        monkeypatch.delenv("LUE_CAPCUT_TTS_RATE", raising=False)
        importlib.reload(config)
