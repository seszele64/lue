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

    def _record_create(self, text, voice, resource_id, rate, rate_passed):
        self.create_calls += 1
        entry = {"text": text, "voice": voice, "resource_id": resource_id}
        if rate_passed:
            entry["rate"] = rate
        self.create_args.append(entry)
        if self.fail_create_times and self.create_calls <= self.fail_create_times:
            raise self.create_exc or ConnectionError("transient")
        return self._create

    def create_tts_task(self, text, voice="BV074_streaming", resource_id=None, rate="1.0"):
        # Mirrors capcut_tts_api.CapCutClient.create_tts_task's signature.
        return self._record_create(text, voice, resource_id, rate, True)

    def query_tts_task(self, task_id, token):
        self.query_calls += 1
        if self.query_exc is not None:
            raise self.query_exc
        return self._query


class FakeCapCutClientNoRate(FakeCapCutClient):
    """Mirrors the pinned SDK (capcut-tts-api 1.5.0): no ``rate`` parameter."""

    def create_tts_task(self, text, voice="BV074_streaming", resource_id=None):
        return self._record_create(text, voice, resource_id, None, False)


class FakeStatusError(Exception):
    """Duck-typed stand-in for ``capcut_tts_api.CapCutAPIError``."""

    def __init__(self, status, message="boom"):
        super().__init__(message)
        self.status_code = status
        self.response_data = {}


def install_fake_sdk(monkeypatch, client) -> None:
    """Inject a minimal ``capcut_tts_api`` module returning ``client``."""
    module = types.ModuleType("capcut_tts_api")
    module.CapCutClient = lambda *args, **kwargs: client
    monkeypatch.setitem(sys.modules, "capcut_tts_api", module)


def printed_text(console) -> str:
    """Return everything a ``Mock`` console printed, as one string."""
    parts = []
    for call in console.print.call_args_list:
        parts.extend(str(arg) for arg in call.args)
        parts.extend(str(value) for value in call.kwargs.values())
    return "\n".join(parts)


#: Minimal bytes that satisfy the provider's MP3 magic check (ID3 tag).
MP3_MAGIC = b"ID3\x03\x00\x00\x00\x00\x00\x00"


def install_fake_requests(
    monkeypatch,
    payload: bytes = MP3_MAGIC,
    status: int = 200,
    headers: dict | None = None,
    redirects: dict[str, str] | None = None,
) -> dict:
    """Inject a minimal synchronous ``requests`` module and record its calls.

    ``redirects`` maps a request URL to the ``Location`` it answers with
    (HTTP 302, empty body).  Every other URL answers with ``status`` and
    ``payload``.  The returned dict records the requested URLs in order, the
    ``allow_redirects`` flag of every call and the last ``timeout``.
    """
    module = types.ModuleType("requests")
    calls: dict = {"url": None, "urls": [], "allow_redirects": [], "timeout": None}
    redirect_map = dict(redirects or {})

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
        def __init__(self, status_code, response_headers, body):
            self.status_code = status_code
            self.headers = dict(response_headers)
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HTTPError(f"HTTP {self.status_code}")

        def iter_content(self, chunk_size=8192):
            if self._body:
                yield self._body

    def get(url, stream=True, timeout=None, allow_redirects=True):
        calls["url"] = url
        calls["urls"].append(url)
        calls["allow_redirects"].append(allow_redirects)
        calls["timeout"] = timeout
        if url in redirect_map:
            return FakeResponse(
                302, {"Location": redirect_map[url]}, b""
            )
        return FakeResponse(status, headers or {}, payload)

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
    # A warning must name the empty catalogue and the voice in use.  The old
    # indentation bug skipped this block entirely, so asserting the text (not
    # just that print was called) is what guards the regression.
    text = printed_text(mock_console)
    assert "empty voice list" in text
    assert "custom_voice" in text


@pytest.mark.asyncio
async def test_initialize_no_empty_voice_warning_for_fallback(monkeypatch, mock_console):
    """The empty-catalogue warning must not fire for the built-in fallback."""
    client = FakeCapCutClient(voices=[])
    install_fake_sdk(monkeypatch, client)
    instance = CapCutTTS(mock_console)
    assert instance.voice == "BV074_streaming"
    assert await instance.initialize() is True
    assert "empty voice list" not in printed_text(mock_console)


@pytest.mark.asyncio
async def test_rate_unsupported_warns_and_is_omitted(monkeypatch, mock_console, tmp_output_dir):
    """The pinned SDK has no ``rate`` param: warn once, never raise TypeError."""
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RATE", "1.25")
    client = FakeCapCutClientNoRate()
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=b"ID3RATE")

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True
    assert instance._rate_supported is False

    text = printed_text(mock_console)
    assert "does not support a speaking rate" in text
    assert "1.25" in text

    out = tmp_output_dir / "rate.mp3"
    await instance.generate_audio("Hi.", str(out))
    assert out.read_bytes() == b"ID3RATE"
    assert client.create_args == [
        {"text": "Hi.", "voice": "BV074_streaming", "resource_id": None}
    ]


@pytest.mark.asyncio
async def test_no_rate_warning_for_default_rate(monkeypatch, mock_console):
    """An unsupported rate is only worth a warning when it was customised."""
    client = FakeCapCutClientNoRate()
    install_fake_sdk(monkeypatch, client)
    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True
    assert "speaking rate" not in printed_text(mock_console)


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
    assert req_calls["allow_redirects"] == [False]
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
    # Valid MP3 magic plus an oversized body: the cap must trip during the
    # stream, so the payload is never mistaken for a corrupt file.
    install_fake_requests(monkeypatch, payload=b"ID3" + b"X" * 100)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "out.mp3"
    with pytest.raises(ValueError, match="limit"):
        await instance.generate_audio("Hi.", str(out))
    assert not out.exists()
    assert not list(tmp_output_dir.glob("*.part"))


@pytest.mark.asyncio
async def test_download_follows_https_redirect(monkeypatch, mock_console, tmp_output_dir):
    """A redirect is followed hop by hop with automatic redirects disabled."""
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    calls = install_fake_requests(
        monkeypatch,
        payload=b"ID3REDIR",
        redirects={"https://cdn.test/a.mp3": "https://cdn2.test/b.mp3"},
    )

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "redir.mp3"
    await instance.generate_audio("Hi.", str(out))

    assert out.read_bytes() == b"ID3REDIR"
    assert calls["urls"] == ["https://cdn.test/a.mp3", "https://cdn2.test/b.mp3"]
    assert calls["allow_redirects"] == [False, False]


@pytest.mark.asyncio
async def test_download_relative_redirect_is_resolved(monkeypatch, mock_console, tmp_output_dir):
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    calls = install_fake_requests(
        monkeypatch,
        payload=b"ID3REL",
        redirects={"https://cdn.test/a.mp3": "/moved/b.mp3"},
    )

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "rel.mp3"
    await instance.generate_audio("Hi.", str(out))

    assert out.read_bytes() == b"ID3REL"
    assert calls["urls"] == ["https://cdn.test/a.mp3", "https://cdn.test/moved/b.mp3"]


@pytest.mark.asyncio
async def test_download_rejects_redirect_downgrade(monkeypatch, mock_console, tmp_output_dir):
    """A redirect to plain HTTP must be refused before it is requested."""
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    calls = install_fake_requests(
        monkeypatch,
        redirects={"https://cdn.test/a.mp3": "http://evil.test/a.mp3"},
    )

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    out = tmp_output_dir / "downgrade.mp3"
    with pytest.raises(ValueError, match="non-HTTPS"):
        await instance.generate_audio("Hi.", str(out))

    # The downgrade was never requested, and it is terminal (no retry).
    assert calls["urls"] == ["https://cdn.test/a.mp3"]
    assert client.create_calls == 1
    assert not out.exists()


@pytest.mark.asyncio
async def test_download_rejects_redirect_without_location(monkeypatch, mock_console, tmp_output_dir):
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, redirects={"https://cdn.test/a.mp3": ""})

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    with pytest.raises(ValueError, match="Location"):
        await instance.generate_audio("Hi.", str(tmp_output_dir / "out.mp3"))


@pytest.mark.asyncio
async def test_download_redirect_chain_is_bounded(monkeypatch, mock_console, tmp_output_dir):
    """More than ``_MAX_REDIRECTS`` hops is an error, not an infinite loop."""
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    redirects = {
        "https://cdn.test/a.mp3": "https://cdn.test/r1.mp3",
        "https://cdn.test/r1.mp3": "https://cdn.test/r2.mp3",
        "https://cdn.test/r2.mp3": "https://cdn.test/r3.mp3",
        "https://cdn.test/r3.mp3": "https://cdn.test/r4.mp3",
    }
    calls = install_fake_requests(monkeypatch, redirects=redirects)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    with pytest.raises(ValueError, match="redirects"):
        await instance.generate_audio("Hi.", str(tmp_output_dir / "loop.mp3"))

    assert len(calls["urls"]) == 4  # initial request + 3 hops
    assert client.create_calls == 1


@pytest.mark.asyncio
async def test_stale_part_files_are_swept(monkeypatch, mock_console, tmp_output_dir):
    """Orphans from a timed-out attempt must not survive a later success."""
    client = FakeCapCutClient()
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch, payload=b"ID3SWEEP")

    out = tmp_output_dir / "out.mp3"
    orphan = tmp_output_dir / f"{out.name}.4242-deadbeef.part"
    orphan.write_bytes(b"junk from an orphaned worker")

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    await instance.generate_audio("Hi.", str(out))

    assert out.read_bytes() == b"ID3SWEEP"
    assert not orphan.exists()
    assert not list(tmp_output_dir.glob("*.part"))


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


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_error_status_is_terminal(
    status, monkeypatch, mock_console, tmp_output_dir
):
    """4xx (other than 408/409/429) must not be retried."""
    monkeypatch.setattr("lue.config.CAPCUT_TTS_MAX_RETRIES", 2)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RETRY_BASE_DELAY", 0.1)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_TIMEOUT", 5)

    client = FakeCapCutClient(fail_create_times=999, create_exc=FakeStatusError(status))
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    with pytest.raises(FakeStatusError):
        await instance.generate_audio("Hello.", str(tmp_output_dir / "out.mp3"))

    assert client.create_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_server_error_status_is_retried(
    status, monkeypatch, mock_console, tmp_output_dir
):
    """429 and 5xx are transient: exhaust every configured attempt."""
    monkeypatch.setattr("lue.config.CAPCUT_TTS_MAX_RETRIES", 2)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_RETRY_BASE_DELAY", 0.1)
    monkeypatch.setattr("lue.config.CAPCUT_TTS_TIMEOUT", 5)
    monkeypatch.setattr(random, "random", Mock(return_value=0.5))

    client = FakeCapCutClient(fail_create_times=999, create_exc=FakeStatusError(status))
    install_fake_sdk(monkeypatch, client)
    install_fake_requests(monkeypatch)

    instance = CapCutTTS(mock_console)
    assert await instance.initialize() is True

    with pytest.raises(FakeStatusError):
        await instance.generate_audio("Hello.", str(tmp_output_dir / "out.mp3"))

    assert client.create_calls == 3  # attempts 0, 1, 2


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

        exc_404 = RuntimeError("not found")
        exc_404.status_code = 404
        assert CapCutTTS._should_retry(exc_404) is False

    def test_status_policy_before_name_fallback(self):
        """A named retryable error that carries a 4xx status is terminal."""
        FakeReadTimeout = type("ReadTimeout", (Exception,), {})
        exc = FakeReadTimeout("timeout, but the server said 400")
        exc.status_code = 400
        assert CapCutTTS._should_retry(exc) is False

        exc_429 = FakeReadTimeout("rate limited")
        exc_429.status_code = 429
        assert CapCutTTS._should_retry(exc_429) is True

    def test_http_error_without_status_is_terminal(self, monkeypatch):
        install_fake_requests(monkeypatch)
        import requests

        assert CapCutTTS._should_retry(requests.exceptions.HTTPError("no status")) is False

    def test_requests_transport_error_is_retryable(self, monkeypatch):
        install_fake_requests(monkeypatch)
        import requests

        assert CapCutTTS._should_retry(requests.exceptions.ConnectionError("x")) is True
        assert CapCutTTS._should_retry(requests.exceptions.Timeout("x")) is True

    def test_capcut_api_error_without_status_is_retryable(self):
        """An SDK error carrying status_code=0 comes from a 200/ret!=0 body."""
        FakeAPIError = type("CapCutAPIError", (Exception,), {})
        exc = FakeAPIError("application-level failure")
        exc.status_code = 0
        assert CapCutTTS._should_retry(exc) is True

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


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "not-a-number"])
def test_config_rate_non_finite_falls_back(monkeypatch, raw):
    """``float('nan')`` passes float() and defeats the clamp, so it is caught."""
    import importlib

    from lue import config

    monkeypatch.setenv("LUE_CAPCUT_TTS_RATE", raw)
    try:
        reloaded = importlib.reload(config)
        assert reloaded.CAPCUT_TTS_RATE == "1.0"
    finally:
        monkeypatch.delenv("LUE_CAPCUT_TTS_RATE", raising=False)
        importlib.reload(config)
