"""Unit tests for the NanoGPT TTS provider.

Covers initialization gating (API key / dependency), request payload shape,
response handling (binary, JSON ``audioUrl``, async ticket), text chunking,
and the timeout / retry behaviour.
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from lue.tts.nanogpt_tts import NanoGPTAPIError, NanoGPTTTS


# ── Helpers ────────────────────────────────────────────────────────────────

def make_response(
    status: int = 200,
    content: bytes = b"",
    content_type: str = "audio/mpeg",
    json_data: dict | None = None,
):
    """Build a mock httpx-like response."""
    resp = Mock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    resp.content = content
    resp.text = "" if json_data is None else str(json_data)
    resp.json = Mock(return_value=json_data if json_data is not None else {})
    resp.request = Mock()
    return resp


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tts(mock_console: Mock) -> NanoGPTTTS:
    """Return an initialised NanoGPTTTS instance with a mock console."""
    instance = NanoGPTTTS(mock_console)
    instance.initialized = True
    instance.client = AsyncMock()
    return instance


# ── Identity / capability ──────────────────────────────────────────────────

class TestIdentity:
    def test_name_matches_filename(self, mock_console):
        assert NanoGPTTTS(mock_console).name == "nanogpt"

    def test_output_format(self, mock_console):
        assert NanoGPTTTS(mock_console).output_format == "mp3"

    def test_no_word_timing(self, mock_console):
        instance = NanoGPTTTS(mock_console)
        assert instance.supports_word_timing is False

    def test_default_voice_is_rex(self, mock_console):
        assert NanoGPTTTS(mock_console).voice == "Rex"

    def test_explicit_voice_overrides_default(self, mock_console):
        assert NanoGPTTTS(mock_console, voice="Eve").voice == "Eve"

    @pytest.mark.asyncio
    async def test_raw_timing_data_empty(self, tts):
        assert await tts.get_raw_timing_data("hello", "/tmp/x.mp3") == []


# ── _should_retry ──────────────────────────────────────────────────────────

class TestShouldRetry:
    def test_timeouts_retry(self):
        assert NanoGPTTTS._should_retry(asyncio.TimeoutError()) is True
        assert NanoGPTTTS._should_retry(TimeoutError()) is True

    def test_httpx_errors_retry(self):
        for name in (
            "TimeoutException", "ConnectError", "ConnectTimeout",
            "ReadTimeout", "WriteTimeout", "PoolTimeout",
            "RemoteProtocolError", "NetworkError",
        ):
            exc_class = type(name, (Exception,), {})
            assert NanoGPTTTS._should_retry(exc_class()) is True, name

    def test_connection_errors_retry(self):
        assert NanoGPTTTS._should_retry(ConnectionError()) is True
        assert NanoGPTTTS._should_retry(BrokenPipeError()) is True

    @pytest.mark.parametrize(
        "status,expected",
        [(429, True), (500, True), (502, True), (503, True),
         (400, False), (401, False), (402, False), (404, False)],
    )
    def test_status_code_branch(self, status, expected):
        assert NanoGPTTTS._should_retry(NanoGPTAPIError("x", status_code=status)) is expected

    @pytest.mark.parametrize("status,expected", [(429, True), (503, True), (400, False)])
    def test_http_status_error_branch(self, status, expected):
        HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        exc = HTTPStatusError()
        exc.response = MagicMock()
        exc.response.status_code = status
        assert NanoGPTTTS._should_retry(exc) is expected

    def test_generic_errors_do_not_retry(self):
        assert NanoGPTTTS._should_retry(ValueError()) is False
        assert NanoGPTTTS._should_retry(RuntimeError()) is False


# ── Chunking ───────────────────────────────────────────────────────────────

class TestChunking:
    def test_short_text_single_chunk(self):
        assert NanoGPTTTS._chunk_text("hello", 5000) == ["hello"]

    def test_long_hard_split_concatenates(self):
        text = "A" * 12000
        chunks = NanoGPTTTS._chunk_text(text, 5000)
        assert all(len(c) <= 5000 for c in chunks)
        assert "".join(chunks) == text

    def test_prefers_sentence_boundaries(self):
        text = ("This is sentence number %d and it is long enough to matter. " % 0) * 200
        chunks = NanoGPTTTS._chunk_text(text, 5000)
        assert all(len(c) <= 5000 for c in chunks)
        assert len(chunks) > 1
        # All non-final chunks should end at a sentence boundary
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith((".", "!", "?"))


# ── generate_audio ─────────────────────────────────────────────────────────

class TestGenerateAudio:
    @pytest.mark.asyncio
    async def test_binary_response_written(self, tts, tmp_path):
        """A 200 with audio bytes is written verbatim to output_path."""
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(return_value=make_response(content=b"ID3AUDIO"))

        await tts.generate_audio("Hello world", out)

        with open(out, "rb") as f:
            assert f.read() == b"ID3AUDIO"

    @pytest.mark.asyncio
    async def test_request_payload_shape(self, tts, tmp_path):
        """Payload carries text+input, model, voice and speed."""
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(return_value=make_response(content=b"ID3"))

        await tts.generate_audio("Hello", out)

        _, kwargs = tts.client.post.call_args
        payload = kwargs["json"]
        assert payload["text"] == "Hello"
        assert payload["input"] == "Hello"          # media-integration-spec/v2
        assert payload["model"] == "xai-tts"
        assert payload["voice"] == "Rex"
        assert "speed" in payload

    @pytest.mark.asyncio
    async def test_endpoint_url(self, tts, tmp_path):
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(return_value=make_response(content=b"ID3"))

        await tts.generate_audio("Hello", out)

        args, _ = tts.client.post.call_args
        assert args[0] == "https://nano-gpt.com/api/tts"

    @pytest.mark.asyncio
    async def test_json_audio_url_downloaded(self, tts, tmp_path):
        """A JSON response with audioUrl triggers a download."""
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(
                content_type="application/json",
                json_data={"audioUrl": "https://cdn.example/a.mp3"},
            )
        )
        tts.client.get = AsyncMock(return_value=make_response(content=b"DOWNLOADED"))

        await tts.generate_audio("Hello", out)

        tts.client.get.assert_awaited_once()
        with open(out, "rb") as f:
            assert f.read() == b"DOWNLOADED"

    @pytest.mark.asyncio
    async def test_async_ticket_polled(self, tts, tmp_path, monkeypatch):
        """A 202 ticket is polled until completed, then downloaded."""
        monkeypatch.setattr("lue.config.NANOGPT_TTS_POLL_INTERVAL", 0)
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(
                status=202,
                content_type="application/json",
                json_data={"runId": "abc", "model": "xai-tts", "cost": 0.01},
            )
        )
        tts.client.get = AsyncMock(
            side_effect=[
                make_response(
                    content_type="application/json",
                    json_data={"status": "pending"},
                ),
                make_response(
                    content_type="application/json",
                    json_data={"status": "completed", "audioUrl": "https://cdn/a.mp3"},
                ),
                make_response(content=b"POLLED"),
            ]
        )

        await tts.generate_audio("Hello", out)

        with open(out, "rb") as f:
            assert f.read() == b"POLLED"
        # 2 status polls (pending, completed) + 1 audio download
        assert tts.client.get.await_count == 3

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self, tts, tmp_path):
        with pytest.raises(ValueError):
            await tts.generate_audio("   ", str(tmp_path / "out.mp3"))

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self, mock_console, tmp_path):
        instance = NanoGPTTTS(mock_console)
        with pytest.raises(RuntimeError):
            await instance.generate_audio("Hello", str(tmp_path / "out.mp3"))

    @pytest.mark.asyncio
    async def test_long_text_chunked_and_concatenated(self, tts, tmp_path):
        """Text over 5000 chars is split into multiple requests."""
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(content=b"SEGMENT")
        )

        await tts.generate_audio("A" * 12000, out)

        assert tts.client.post.await_count == 3  # 5000 + 5000 + 2000
        with open(out, "rb") as f:
            assert f.read() == b"SEGMENT" * 3


# ── Retry behaviour ────────────────────────────────────────────────────────

class TestRetryBehaviour:
    @pytest.mark.asyncio
    async def test_retry_then_success(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.NANOGPT_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_RETRY_BASE_DELAY", 0.01)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        calls = {"n": 0}

        async def side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise NanoGPTAPIError("server error", status_code=503)
            return make_response(content=b"OK")

        tts.client.post = AsyncMock(side_effect=side_effect)

        await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_retries_exhausted(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.NANOGPT_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_RETRY_BASE_DELAY", 0.01)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        tts.client.post = AsyncMock(
            side_effect=NanoGPTAPIError("server error", status_code=503)
        )

        with pytest.raises(NanoGPTAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert tts.client.post.await_count == 3  # attempts 0, 1, 2

    @pytest.mark.asyncio
    async def test_non_retryable_400_raises_immediately(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.NANOGPT_TTS_MAX_RETRIES", 3)

        tts.client.post = AsyncMock(
            side_effect=NanoGPTAPIError(
                "Text must be a string between 1 and 5000 characters.",
                status_code=400,
            )
        )

        with pytest.raises(NanoGPTAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert tts.client.post.await_count == 1  # no retries on 400

    @pytest.mark.asyncio
    async def test_timeout_wraps_request(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.NANOGPT_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_MAX_RETRIES", 0)

        async def slow_post(*args, **kwargs):
            await asyncio.sleep(10)
            return make_response()

        tts.client.post = AsyncMock(side_effect=slow_post)

        start = asyncio.get_event_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 3, f"timeout not enforced (took {elapsed:.1f}s)"

    @pytest.mark.asyncio
    async def test_backoff_schedule(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.NANOGPT_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_MAX_RETRIES", 3)
        monkeypatch.setattr("lue.config.NANOGPT_TTS_RETRY_BASE_DELAY", 0.1)

        tts.client.post = AsyncMock(
            side_effect=NanoGPTAPIError("server error", status_code=500)
        )

        sleeps: list[float] = []

        async def mock_sleep(duration):
            sleeps.append(duration)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(random, "random", Mock(return_value=0.5))  # jitter = 0

        with pytest.raises(NanoGPTAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))

        assert len(sleeps) == 3
        assert sleeps[0] == pytest.approx(0.1, abs=0.001)
        assert sleeps[1] == pytest.approx(0.2, abs=0.001)
        assert sleeps[2] == pytest.approx(0.4, abs=0.001)


# ── initialize ─────────────────────────────────────────────────────────────

class TestInitialize:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_console, monkeypatch):
        monkeypatch.delenv("NANOGPT_API_KEY", raising=False)
        instance = NanoGPTTTS(mock_console)
        assert await instance.initialize() is False
        assert instance.initialized is False

    @pytest.mark.asyncio
    async def test_success(self, mock_console, monkeypatch):
        import httpx

        monkeypatch.setenv("NANOGPT_API_KEY", "sk-nano-test")
        instance = NanoGPTTTS(mock_console)

        mock_client = MagicMock(spec=httpx.AsyncClient)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

        assert await instance.initialize() is True
        assert instance.initialized is True
        assert instance.client is mock_client

    @pytest.mark.asyncio
    async def test_api_key_sent_as_header(self, mock_console, monkeypatch):
        import httpx

        monkeypatch.setenv("NANOGPT_API_KEY", "sk-nano-test")
        instance = NanoGPTTTS(mock_console)

        captured: dict = {}

        def fake_client(**kwargs):
            captured.update(kwargs)
            return MagicMock(spec=httpx.AsyncClient)

        monkeypatch.setattr(httpx, "AsyncClient", fake_client)
        assert await instance.initialize() is True
        assert captured["headers"]["x-api-key"] == "sk-nano-test"


# ── warm-up ────────────────────────────────────────────────────────────────

class TestWarmUp:
    @pytest.mark.asyncio
    async def test_warm_up_generates_and_cleans_up(self, tts, tmp_path, monkeypatch):
        from lue import config

        monkeypatch.setattr(config, "AUDIO_DATA_DIR", str(tmp_path))

        tts.generate_audio = AsyncMock()
        await tts.warm_up()

        tts.generate_audio.assert_awaited_once()
        args, _ = tts.generate_audio.call_args
        assert args[0] == "Ready."
        # Warm-up file is removed afterwards (generate_audio is mocked, so
        # explicitly create the file to verify cleanup).
        warmup_file = tmp_path / f".warmup_nanogpt.{tts.output_format}"
        warmup_file.write_bytes(b"stub")
        await tts.warm_up()
        assert not warmup_file.exists()
