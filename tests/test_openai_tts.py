"""Unit tests for OpenAI TTS timeout and retry behaviour.

These tests exercise the per-request timeout wrapper, exponential-backoff retry
logic, and the ``_should_retry`` static-method heuristics.
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from lue.tts.openai_tts import OpenAITTS


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tts(mock_console: Mock) -> OpenAITTS:
    """Return an initialised OpenAITTS instance with a mock console."""
    tts_instance = OpenAITTS(mock_console)
    tts_instance.initialized = True
    return tts_instance


@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client and its sub-mocks.

    Returns a tuple of ``(mock_client, mock_response, mock_cm)`` where
    ``mock_cm`` is the async-context-manager returned (synchronously) by
    ``client.audio.speech.with_streaming_response.create(...).
    """
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    # In the real SDK ``.create(...)`` returns the context manager synchronously,
    # not via a coroutine.  We use a ``MagicMock`` here to prevent ``async with``
    # from receiving a coroutine object.
    mock_streaming = MagicMock()
    mock_streaming.create.return_value = mock_cm
    mock_client.audio.speech.with_streaming_response = mock_streaming

    return mock_client, mock_response, mock_cm


# ── _should_retry tests ─────────────────────────────────────────────────────

class TestShouldRetry:
    """Tests for ``OpenAITTS._should_retry`` static method."""

    def test_should_retry_timeout_error(self):
        """asyncio.TimeoutError should be retryable."""
        assert OpenAITTS._should_retry(asyncio.TimeoutError()) is True

    def test_should_retry_httpx_errors(self):
        """Common httpx exception names should trigger a retry."""
        names = [
            "TimeoutException",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "NetworkError",
        ]
        for name in names:
            exc_class = type(name, (Exception,), {})
            exc = exc_class()
            assert OpenAITTS._should_retry(exc) is True, f"{name} should be retryable"

    def test_should_retry_connection_errors(self):
        """ConnectionError and BrokenPipeError should be retryable."""
        assert OpenAITTS._should_retry(ConnectionError()) is True
        assert OpenAITTS._should_retry(BrokenPipeError()) is True

    def test_should_retry_http_status(self):
        """429 and 5xx HTTP status codes should be retryable; other 4xx should not."""
        # --- Branch 1: exception has a top-level ``status_code`` attribute ---
        for status, expected in [
            (429, True),
            (500, True),
            (502, True),
            (503, True),
            (599, True),
            (400, False),
            (404, False),
            (422, False),
        ]:
            exc = MagicMock()
            exc.status_code = status
            result = OpenAITTS._should_retry(exc)
            assert result is expected, (
                f"MagicMock with status_code={status}: expected {expected}, got {result}"
            )

        # --- Branch 2: exception class name is "HTTPStatusError" ---
        HTTPStatusError = type("HTTPStatusError", (Exception,), {})

        for status, expected in [
            (429, True),
            (500, True),
            (503, True),
            (400, False),
            (404, False),
        ]:
            exc = HTTPStatusError()
            exc.response = MagicMock()
            exc.response.status_code = status
            result = OpenAITTS._should_retry(exc)
            assert result is expected, (
                f"HTTPStatusError with response.status_code={status}: "
                f"expected {expected}, got {result}"
            )

    def test_non_retryable_errors(self):
        """Generic exceptions should not be retryable."""
        assert OpenAITTS._should_retry(ValueError()) is False
        assert OpenAITTS._should_retry(RuntimeError()) is False
        assert OpenAITTS._should_retry(TypeError()) is False


# ── generate_audio success / failure paths ───────────────────────────────────

class TestGenerateAudio:
    """Tests for ``OpenAITTS.generate_audio``."""

    @pytest.mark.asyncio
    async def test_generate_audio_success_first_attempt(self, tts, mock_openai_client):
        """When ``stream_to_file`` succeeds immediately, ``generate_audio`` returns cleanly."""
        mock_client, mock_response, _ = mock_openai_client
        mock_response.stream_to_file = AsyncMock(return_value=None)
        tts.client = mock_client

        # Should not raise.
        await tts.generate_audio("Hello world", "/tmp/test.mp3")
        mock_response.stream_to_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_audio_timeout_no_retries(self, tts, mock_openai_client, monkeypatch):
        """With max_retries=0, a timeout on the first attempt should be raised immediately."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 0)

        mock_client, mock_response, _ = mock_openai_client
        mock_response.stream_to_file = AsyncMock(side_effect=asyncio.TimeoutError)
        tts.client = mock_client

        with pytest.raises(asyncio.TimeoutError):
            await tts.generate_audio("Hello world", "/tmp/test.mp3")

    @pytest.mark.asyncio
    async def test_retry_then_success(self, tts, mock_openai_client, monkeypatch):
        """Transient failures on attempts 0 and 1 should be retried; success on attempt 2."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.OPENAI_TTS_RETRY_BASE_DELAY", 0.1)

        mock_client, mock_response, _ = mock_openai_client
        tts.client = mock_client

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            if call_count < 2:
                call_count += 1
                raise asyncio.TimeoutError()
            call_count += 1
            return None

        mock_response.stream_to_file = AsyncMock(side_effect=side_effect)

        # Should eventually succeed.
        await tts.generate_audio("Hello world", "/tmp/test.mp3")
        assert call_count == 3  # failures on calls 0 and 1, success on call 2

    @pytest.mark.asyncio
    async def test_retries_exhausted(self, tts, mock_openai_client, monkeypatch):
        """When all retries are exhausted, the last exception should be raised."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.OPENAI_TTS_RETRY_BASE_DELAY", 0.1)

        mock_client, mock_response, _ = mock_openai_client
        mock_response.stream_to_file = AsyncMock(side_effect=asyncio.TimeoutError)
        tts.client = mock_client

        with pytest.raises(asyncio.TimeoutError):
            await tts.generate_audio("Hello world", "/tmp/test.mp3")

        assert mock_response.stream_to_file.call_count == 3  # attempts 0, 1, 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, tts, mock_openai_client, monkeypatch):
        """Retry delays should follow the exponential-backoff formula."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 3)
        monkeypatch.setattr("lue.config.OPENAI_TTS_RETRY_BASE_DELAY", 0.1)

        mock_client, mock_response, _ = mock_openai_client
        mock_response.stream_to_file = AsyncMock(side_effect=asyncio.TimeoutError)
        tts.client = mock_client

        sleeps: list[float] = []

        async def mock_sleep(duration):
            sleeps.append(duration)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        # Fix random so jitter is eliminated:
        #   jitter = delay * 0.25 * (2*0.5 - 1)  == 0
        monkeypatch.setattr(random, "random", Mock(return_value=0.5))

        with pytest.raises(asyncio.TimeoutError):
            await tts.generate_audio("Hello world", "/tmp/test.mp3")

        # 3 retry sleeps after attempts 0, 1, 2 (attempt 3 exhausts retries)
        assert len(sleeps) == 3
        assert sleeps[0] == pytest.approx(0.1, abs=0.001)   # 0.1 * 2**0
        assert sleeps[1] == pytest.approx(0.2, abs=0.001)   # 0.1 * 2**1
        assert sleeps[2] == pytest.approx(0.4, abs=0.001)   # 0.1 * 2**2

    @pytest.mark.asyncio
    async def test_connection_setup_is_wrapped_by_timeout(self, tts, monkeypatch):
        """Verify the fix: the ``async with`` entry IS now wrapped by the timeout.

        After the fix: the entire request lifecycle (connection setup + streaming)
        is guarded by ``asyncio.wait_for``. A hung connection should now raise
        ``asyncio.TimeoutError`` within the configured timeout (1s), not after 5s.
        """
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 0)

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.stream_to_file = AsyncMock(return_value=None)

        mock_cm = AsyncMock()

        async def slow_aenter(self=None):
            # Simulate a hung connection setup (no HTTP response).
            await asyncio.sleep(10)
            return mock_response

        mock_cm.__aenter__ = slow_aenter
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        # Same as the fixture: ``.create(...)`` must return the CM synchronously.
        mock_streaming = MagicMock()
        mock_streaming.create.return_value = mock_cm
        mock_client.audio.speech.with_streaming_response = mock_streaming

        tts.client = mock_client

        start = asyncio.get_event_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                tts.generate_audio("Hello world", "/tmp/test.mp3"),
                timeout=5,
            )
        elapsed = asyncio.get_event_loop().time() - start

        # The inner timeout now fires at ~1 s, proving the fix works.
        assert elapsed < 2, (
            f"BUG: connection setup is not wrapped by timeout "
            f"(took {elapsed:.1f}s, expected < 2s)"
        )
