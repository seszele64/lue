"""Unit tests for OpenAI TTS timeout and retry behaviour.

These tests exercise the per-request timeout wrapper, exponential-backoff retry
logic, and the ``_should_retry`` static-method heuristics.

The retry classifier is verified against *real* ``openai`` and ``httpx``
exception instances (not duck-typed stand-ins), because the production bug was
precisely that the SDK wraps every network/HTTP failure in its own exception
types, which the original name-based heuristic never matched.
"""

from __future__ import annotations

import asyncio
import random
import ssl
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import openai
import pytest

from lue.tts.openai_tts import (
    _RETRYABLE_STATUSES,
    _RETRYABLE_TRANSPORT_NAMES,
    _status_is_retryable,
    _status_of,
    OpenAITTS,
)


# ── Helpers for building genuine SDK exceptions ────────────────────────────

def make_request() -> httpx.Request:
    """A realistic ``httpx.Request``, as the SDK attaches to every error."""
    return httpx.Request("POST", "https://api.openai.com/v1/audio/speech")


def make_response(status: int) -> httpx.Response:
    """A realistic ``httpx.Response`` carrying ``status``."""
    return httpx.Response(status, request=make_request())


def make_status_error(cls, status: int) -> Exception:
    """Instantiate an ``openai`` status-error subclass bound to ``status``."""
    return cls(f"HTTP {status}", response=make_response(status), body=None)


def chained(outer: BaseException, cause: BaseException) -> BaseException:
    """Return ``outer`` linked to ``cause`` exactly as ``raise ... from ...``."""
    try:
        try:
            raise cause
        except Exception as inner:
            raise outer from inner
    except Exception as exc:
        return exc


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


# ── Real OpenAI SDK exceptions ──────────────────────────────────────────────
#
# Regression coverage for the production bug: the real SDK never raises the
# bare httpx exceptions the original heuristic inspected, it wraps them in
# ``openai.APIConnectionError`` / ``openai.APITimeoutError`` /
# ``openai.APIStatusError``, so no real transient failure was ever retried.

class TestShouldRetryOpenAISDKErrors:
    """``_should_retry`` against genuine ``openai`` exception instances."""

    def test_api_status_error_retryable_statuses(self):
        """408, 409, 429 and 5xx are transient and must be retried."""
        for status in (408, 409, 429, 500, 502, 503, 504, 529):
            exc = openai.APIStatusError(
                f"HTTP {status}", response=make_response(status), body=None
            )
            assert OpenAITTS._should_retry(exc) is True, f"status {status} should retry"

    def test_api_status_error_non_retryable_statuses(self):
        """Other 4xx client errors are terminal and must not be retried."""
        for status in (400, 401, 403, 404, 422):
            exc = openai.APIStatusError(
                f"HTTP {status}", response=make_response(status), body=None
            )
            assert OpenAITTS._should_retry(exc) is False, f"status {status} must not retry"

    def test_rate_limit_error_retryable(self):
        """``openai.RateLimitError`` (429) is retryable."""
        assert OpenAITTS._should_retry(make_status_error(openai.RateLimitError, 429)) is True

    def test_internal_server_error_retryable(self):
        """``openai.InternalServerError`` (5xx) is retryable."""
        for status in (500, 502, 503):
            exc = make_status_error(openai.InternalServerError, status)
            assert OpenAITTS._should_retry(exc) is True

    def test_bad_request_error_not_retryable(self):
        """``openai.BadRequestError`` (400) is terminal."""
        assert OpenAITTS._should_retry(make_status_error(openai.BadRequestError, 400)) is False

    def test_authentication_error_not_retryable(self):
        """``openai.AuthenticationError`` (401) is terminal."""
        assert OpenAITTS._should_retry(make_status_error(openai.AuthenticationError, 401)) is False

    def test_not_found_error_not_retryable(self):
        """``openai.NotFoundError`` (404) is terminal."""
        assert OpenAITTS._should_retry(make_status_error(openai.NotFoundError, 404)) is False

    def test_conflict_error_retryable(self):
        """``openai.ConflictError`` (409) is treated as transient."""
        assert OpenAITTS._should_retry(make_status_error(openai.ConflictError, 409)) is True

    def test_permission_denied_error_not_retryable(self):
        """``openai.PermissionDeniedError`` (403) is terminal."""
        assert OpenAITTS._should_retry(make_status_error(openai.PermissionDeniedError, 403)) is False

    def test_unprocessable_entity_error_not_retryable(self):
        """``openai.UnprocessableEntityError`` (422) is terminal."""
        assert OpenAITTS._should_retry(make_status_error(openai.UnprocessableEntityError, 422)) is False

    def test_api_connection_error_retryable(self):
        """``openai.APIConnectionError`` wraps transport failures — retryable."""
        exc = openai.APIConnectionError(request=make_request())
        assert OpenAITTS._should_retry(exc) is True

    def test_api_timeout_error_retryable(self):
        """``openai.APITimeoutError`` (subclass of APIConnectionError) is retryable."""
        exc = openai.APITimeoutError(request=make_request())
        assert OpenAITTS._should_retry(exc) is True

    def test_api_error_base_not_retryable(self):
        """A plain ``openai.APIError`` carries no status and is not transient."""
        exc = openai.APIError("boom", request=make_request(), body=None)
        assert OpenAITTS._should_retry(exc) is False

    def test_api_status_error_non_error_status_falls_through(self):
        """A non-error status (e.g. 302) yields no verdict, so the walk ends False."""
        exc = openai.APIStatusError("redirect", response=make_response(302), body=None)
        assert OpenAITTS._should_retry(exc) is False


class TestShouldRetryHttpxErrors:
    """``_should_retry`` against genuine ``httpx`` exception instances."""

    def test_httpx_timeout_family_retryable(self):
        """Every ``httpx`` timeout variant is transient."""
        for cls in (
            httpx.TimeoutException,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
        ):
            assert OpenAITTS._should_retry(cls("t", request=make_request())) is True, cls

    def test_httpx_connect_error_retryable(self):
        assert OpenAITTS._should_retry(httpx.ConnectError("c", request=make_request())) is True

    def test_httpx_remote_protocol_error_retryable(self):
        exc = httpx.RemoteProtocolError("peer went away", request=make_request())
        assert OpenAITTS._should_retry(exc) is True

    def test_httpx_protocol_error_retryable(self):
        exc = httpx.ProtocolError("protocol", request=make_request())
        assert OpenAITTS._should_retry(exc) is True

    def test_httpx_read_error_retryable(self):
        assert OpenAITTS._should_retry(httpx.ReadError("r", request=make_request())) is True

    def test_httpx_write_error_retryable(self):
        assert OpenAITTS._should_retry(httpx.WriteError("w", request=make_request())) is True

    def test_httpx_close_error_retryable(self):
        assert OpenAITTS._should_retry(httpx.CloseError("c", request=make_request())) is True

    def test_httpx_proxy_error_retryable(self):
        assert OpenAITTS._should_retry(httpx.ProxyError("p", request=make_request())) is True

    def test_httpx_unsupported_protocol_retryable(self):
        exc = httpx.UnsupportedProtocol("bad scheme", request=make_request())
        assert OpenAITTS._should_retry(exc) is True

    def test_httpx_transport_error_base_retryable(self):
        assert OpenAITTS._should_retry(httpx.TransportError("t", request=make_request())) is True

    def test_httpx_http_status_error_retryable(self):
        """``httpx.HTTPStatusError`` follows the HTTP status policy."""
        for status in (408, 409, 429, 500, 503):
            exc = httpx.HTTPStatusError(
                f"HTTP {status}", request=make_request(), response=make_response(status)
            )
            assert OpenAITTS._should_retry(exc) is True, f"status {status} should retry"

    def test_httpx_http_status_error_not_retryable(self):
        for status in (400, 401, 404, 422):
            exc = httpx.HTTPStatusError(
                f"HTTP {status}", request=make_request(), response=make_response(status)
            )
            assert OpenAITTS._should_retry(exc) is False, f"status {status} must not retry"

    def test_httpx_decoding_error_not_retryable(self):
        """``DecodingError`` is a RequestError but not a TransportError."""
        exc = httpx.DecodingError("bad gzip", request=make_request())
        assert OpenAITTS._should_retry(exc) is False


# ── __cause__ chain walking ─────────────────────────────────────────────────

class TestShouldRetryCauseChain:
    """The classifier unwraps ``__cause__`` to find the real failure."""

    def test_cause_chain_sdk_timeout_retryable(self):
        """A generic wrapper around ``APITimeoutError`` is still retryable."""
        exc = chained(RuntimeError("wrapped"), openai.APITimeoutError(request=make_request()))
        assert OpenAITTS._should_retry(exc) is True

    def test_cause_chain_sdk_status_error_consulted(self):
        """Status policy is applied to a wrapped ``APIStatusError``."""
        retryable = chained(
            RuntimeError("wrapped"),
            make_status_error(openai.RateLimitError, 429),
        )
        terminal = chained(
            RuntimeError("wrapped"),
            make_status_error(openai.BadRequestError, 400),
        )
        assert OpenAITTS._should_retry(retryable) is True
        assert OpenAITTS._should_retry(terminal) is False

    def test_cause_chain_sdk_connection_error_retryable(self):
        """``APIConnectionError`` two levels deep is still found."""
        exc = chained(
            RuntimeError("outer"),
            chained(
                ValueError("inner"),
                openai.APIConnectionError(request=make_request()),
            ),
        )
        assert OpenAITTS._should_retry(exc) is True

    def test_cause_chain_two_levels_deep_retryable(self):
        """Three-deep chain ending in a 500 is retryable."""
        exc = chained(
            RuntimeError("a"),
            chained(ValueError("b"), make_status_error(openai.InternalServerError, 500)),
        )
        assert OpenAITTS._should_retry(exc) is True

    def test_cause_chain_builtin_socket_error_retryable(self):
        """``SSLError`` / ``ConnectionResetError`` behind a wrapper are retryable."""
        assert OpenAITTS._should_retry(
            chained(RuntimeError("w"), ssl.SSLError("handshake failure"))
        ) is True
        assert OpenAITTS._should_retry(
            chained(RuntimeError("w"), ConnectionResetError("reset by peer"))
        ) is True

    def test_cause_cycles_terminate(self):
        """Self-referential and mutual cause cycles must not loop forever."""
        self_ref = ValueError("self")
        self_ref.__cause__ = self_ref
        assert OpenAITTS._should_retry(self_ref) is False

        left, right = ValueError("left"), ValueError("right")
        left.__cause__ = right
        right.__cause__ = left
        assert OpenAITTS._should_retry(left) is False


class TestShouldRetryRobustness:
    """The classifier must not blow up on odd, mocky or absent inputs."""

    def test_magicmock_exception_not_retryable(self):
        """A bare ``MagicMock`` has no real status and must not be retried."""
        assert OpenAITTS._should_retry(MagicMock()) is False

    def test_non_int_status_code_ignored(self):
        """Placeholder status values are ignored rather than misread as ints."""
        for bogus in ("429", 429.0, None, True, MagicMock()):
            exc = MagicMock()
            exc.status_code = bogus
            assert OpenAITTS._should_retry(exc) is False, f"status {bogus!r} must not retry"

    def test_status_helpers_directly(self):
        """``_status_of`` / ``_status_is_retryable`` behave as tri-state helpers."""
        assert _status_of(RuntimeError("x")) is None
        assert _status_of(make_status_error(openai.RateLimitError, 429)) == 429

        for status in (408, 409, 429, 500, 503):
            assert _status_is_retryable(status) is True, status
        for status in (400, 401, 404, 422):
            assert _status_is_retryable(status) is False, status
        for status in (None, 200, 302):
            assert _status_is_retryable(status) is None, status

    def test_missing_openai_module_still_classifies(self):
        """With ``openai`` uninstalled the name/status fallbacks still work."""
        monkey = pytest.MonkeyPatch()
        monkey.setitem(sys.modules, "openai", None)
        try:
            # httpx is importable, so isinstance checks still work.
            exc = httpx.HTTPStatusError("429", request=make_request(), response=make_response(429))
            assert OpenAITTS._should_retry(exc) is True

            # Duck-typed stand-ins fall back to name + status matching.
            Fake = type("HTTPStatusError", (Exception,), {})
            fake = Fake()
            fake.response = make_response(503)
            assert OpenAITTS._should_retry(fake) is True

            Timeoutish = type("ReadTimeout", (Exception,), {})
            assert OpenAITTS._should_retry(Timeoutish()) is True
        finally:
            monkey.undo()

    def test_missing_httpx_and_openai_still_uses_builtin_checks(self):
        """With both optional deps absent only builtin checks remain."""
        monkey = pytest.MonkeyPatch()
        monkey.setitem(sys.modules, "openai", None)
        monkey.setitem(sys.modules, "httpx", None)
        try:
            assert OpenAITTS._should_retry(asyncio.TimeoutError()) is True
            assert OpenAITTS._should_retry(ConnectionResetError("reset")) is True
            assert OpenAITTS._should_retry(ValueError("nope")) is False

            # Name fallback for a duck-typed transport error.
            Fake = type("ConnectError", (Exception,), {})
            assert OpenAITTS._should_retry(Fake()) is True

            # Status fallback for a duck-typed SDK error.
            FakeStatus = type("APIStatusError", (Exception,), {})
            fake = FakeStatus()
            fake.status_code = 503
            assert OpenAITTS._should_retry(fake) is True
            fake.status_code = 400
            assert OpenAITTS._should_retry(fake) is False
        finally:
            monkey.undo()

    def test_retryable_transport_names_cover_httpx_families(self):
        """The name fallback must list every transient httpx family."""
        for name in (
            "TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
            "PoolTimeout", "ConnectError", "NetworkError", "ProxyError",
            "ReadError", "WriteError", "CloseError", "RemoteProtocolError",
            "ProtocolError", "UnsupportedProtocol",
        ):
            assert name in _RETRYABLE_TRANSPORT_NAMES, name

    def test_retryable_statuses_contents(self):
        """Only 408/409/429 are opted in beyond the 5xx range."""
        assert _RETRYABLE_STATUSES == frozenset({408, 409, 429})


# ── End-to-end: real SDK errors flowing through generate_audio ─────────────

class TestGenerateAudioWithRealSdkErrors:
    """``generate_audio`` must retry real SDK errors, not swallow the bug."""

    @pytest.mark.asyncio
    async def test_transient_sdk_errors_are_retried_then_succeed(self, tts, mock_openai_client, monkeypatch):
        """Two transient SDK failures then success => 3 requests total."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.OPENAI_TTS_RETRY_BASE_DELAY", 0.1)

        mock_client, mock_response, _ = mock_openai_client
        tts.client = mock_client

        transient = [
            openai.APITimeoutError(request=make_request()),
            openai.APIConnectionError(request=make_request()),
        ]
        calls = 0

        async def side_effect(*args, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            if index < len(transient):
                raise transient[index]
            return None

        mock_response.stream_to_file = AsyncMock(side_effect=side_effect)

        await tts.generate_audio("Hello world", "/tmp/test.mp3")

        assert mock_response.stream_to_file.call_count == 3
        assert calls == 3

    @pytest.mark.asyncio
    async def test_raw_httpx_transport_error_is_retried(self, tts, mock_openai_client, monkeypatch):
        """A bare ``httpx.ReadError`` (e.g. raised by a proxy) must also be retried."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.OPENAI_TTS_RETRY_BASE_DELAY", 0.1)

        mock_client, mock_response, _ = mock_openai_client
        tts.client = mock_client

        calls = 0

        async def side_effect(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise httpx.ReadError("peer reset", request=make_request())
            return None

        mock_response.stream_to_file = AsyncMock(side_effect=side_effect)

        await tts.generate_audio("Hello world", "/tmp/test.mp3")

        assert mock_response.stream_to_file.call_count == 3

    @pytest.mark.asyncio
    async def test_bad_request_error_is_raised_without_retry(self, tts, mock_openai_client, monkeypatch):
        """A terminal 400 must surface immediately without a second request."""
        monkeypatch.setattr("lue.config.OPENAI_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.OPENAI_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.OPENAI_TTS_RETRY_BASE_DELAY", 0.1)

        mock_client, mock_response, _ = mock_openai_client
        tts.client = mock_client

        mock_response.stream_to_file = AsyncMock(
            side_effect=make_status_error(openai.BadRequestError, 400)
        )

        with pytest.raises(openai.BadRequestError):
            await tts.generate_audio("Hello world", "/tmp/test.mp3")

        assert mock_response.stream_to_file.call_count == 1
