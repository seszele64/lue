import asyncio
import os
import random
import logging
from ssl import SSLError

from rich.console import Console

from .base import TTSBase
from .. import config


# ── Retry classification helpers ───────────────────────────────────────────

#: Exception *class names* of transient transport failures.  Matching on the
#: name (rather than only on the type) keeps the heuristic working for
#: duck-typed stand-ins and for SDKs that cannot be imported at all.
_RETRYABLE_TRANSPORT_NAMES = frozenset(
    {
        # timeouts
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        # connection / network
        "ConnectError",
        "NetworkError",
        "ProxyError",
        "ReadError",
        "WriteError",
        "CloseError",
        # protocol
        "RemoteProtocolError",
        "ProtocolError",
        "UnsupportedProtocol",
    }
)

#: Non-5xx HTTP statuses that are still worth retrying: request timeout,
#: version-conflict (optimistic concurrency) and rate limiting.
_RETRYABLE_STATUSES = frozenset({408, 409, 429})


def _status_of(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from ``exc``.

    Checks the SDK-style top-level ``status_code`` attribute first, then the
    ``response.status_code`` pair used by ``httpx``.  Returns ``None`` when no
    usable integer status can be found (missing, ``None``, or a mock/str
    placeholder).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status

    return None


def _status_is_retryable(status: int | None) -> bool | None:
    """Tri-state verdict for an HTTP status code.

    * ``True``  — 408, 409, 429 or any 5xx server error.
    * ``False`` — any other 4xx client error (terminal: bad request, auth,
      not found, ...).
    * ``None``  — unknown / no error status, so no verdict can be given and
      the caller should fall through to the next check.
    """
    if status is None:
        return None
    if status in _RETRYABLE_STATUSES:
        return True
    if 500 <= status < 600:
        return True
    if 400 <= status < 500:
        return False
    return None


class OpenAITTS(TTSBase):
    """TTS implementation for OpenAI's Speech API (and compatible endpoints)."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def output_format(self) -> str:
        return "mp3"

    @property
    def supports_word_timing(self) -> bool:
        """OpenAI TTS does not provide word-level timing data."""
        return False

    def __init__(self, console: Console, voice: str = None, lang: str = None):
        super().__init__(console, voice, lang)
        self.client = None
        self._model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
        if self.voice is None:
            self.voice = config.TTS_VOICES.get(self.name)

    async def initialize(self) -> bool:
        """Check for openai package and OPENAI_API_KEY environment variable."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            self.console.print(
                "[bold red]Error: 'openai' package not found.[/bold red]"
            )
            self.console.print(
                "[yellow]Please run 'pip install openai' to use this TTS model.[/yellow]"
            )
            logging.error("'openai' is not installed.")
            return False

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.console.print(
                "[bold red]Error: OPENAI_API_KEY environment variable is not set.[/bold red]"
            )
            self.console.print(
                "[yellow]Please set your OpenAI API key: export OPENAI_API_KEY='your-key'[/yellow]"
            )
            logging.error("OPENAI_API_KEY is not set.")
            return False

        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)

        self.initialized = True
        self.console.print("[green]OpenAI TTS model is available.[/green]")
        return True

    async def generate_audio(self, text: str, output_path: str):
        """Generate audio from text using OpenAI Speech API and save to file.

        Applies per-request timeout and exponential-backoff retry for
        transient failures (timeouts, server errors, rate limits).
        """
        if not self.initialized:
            raise RuntimeError("OpenAI TTS has not been initialized.")

        from ..config import OPENAI_TTS_TIMEOUT, OPENAI_TTS_MAX_RETRIES, OPENAI_TTS_RETRY_BASE_DELAY

        last_exception = None

        for attempt in range(OPENAI_TTS_MAX_RETRIES + 1):
            try:
                async def _do_request():
                    """Execute the full TTS request inside the timeout scope."""
                    async with self.client.audio.speech.with_streaming_response.create(
                        model=self._model,
                        voice=self.voice,
                        input=text,
                        response_format="mp3",
                    ) as response:
                        await response.stream_to_file(output_path)

                await asyncio.wait_for(
                    _do_request(),
                    timeout=OPENAI_TTS_TIMEOUT,
                )
                # Success — return immediately
                if attempt > 0:
                    logging.info(
                        "OpenAI TTS recovered after %d retries for '%s...'",
                        attempt,
                        text[:50],
                    )
                return

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"OpenAI TTS timed out after {OPENAI_TTS_TIMEOUT}s "
                    f"for text: '{text[:50]}...'"
                )
            except Exception as e:
                last_exception = e

            # --- Should we retry? -----------------------------------------
            if not self._should_retry(last_exception):
                # Non-retryable error — re-raise immediately
                logging.error(
                    "OpenAI TTS non-retryable error (attempt %d/%d) "
                    "for '%s...': %s",
                    attempt + 1,
                    OPENAI_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            # --- Exhausted retries? ---------------------------------------
            if attempt >= OPENAI_TTS_MAX_RETRIES:
                logging.error(
                    "OpenAI TTS exhausted %d retries for '%s...': %s",
                    OPENAI_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            # --- Backoff and retry ----------------------------------------
            delay = OPENAI_TTS_RETRY_BASE_DELAY * (2 ** attempt)
            jitter = delay * 0.25 * (2 * random.random() - 1)  # ±25%
            wait_time = max(0.1, delay + jitter)

            logging.warning(
                "OpenAI TTS retry %d/%d in %.1fs for '%s...': %s",
                attempt + 1,
                OPENAI_TTS_MAX_RETRIES,
                wait_time,
                text[:50],
                type(last_exception).__name__,
            )
            await asyncio.sleep(wait_time)

    @staticmethod
    def _should_retry(exc: BaseException) -> bool:
        """Return True if ``exc`` represents a transient failure worth retrying.

        The real OpenAI SDK does **not** raise the httpx exceptions the
        original heuristic looked for: it wraps them in
        ``openai.APIConnectionError`` / ``openai.APITimeoutError`` (network
        problems) and ``openai.APIStatusError`` subclasses (HTTP errors), each
        linked to the underlying httpx exception through ``__cause__``.  So we
        walk the ``__cause__`` chain and classify at every level.

        Ordered checks (the first one that yields a verdict wins):

        1. ``asyncio.TimeoutError`` / builtin ``TimeoutError`` — our
           per-request timeout plus any socket timeout surfaced as builtin.
        2. ``openai.APIConnectionError`` — SDK connection failure (this also
           covers ``openai.APITimeoutError``).
        3. ``openai.APIStatusError`` — decided by the HTTP status policy.
        4. ``httpx.HTTPStatusError`` — decided by the HTTP status policy.
        5. ``httpx.TransportError`` — timeout/network/protocol/stream faults.
        6. Name-based fallback for the same httpx families.  Kept
           deliberately so duck-typed stand-ins still classify correctly even
           when ``httpx``/``openai`` cannot be imported.
        7. Builtin socket errors (``ConnectionError``, ``BrokenPipeError``,
           ``SSLError``).
        8. Any status code carried on the exception (``status_code`` or
           ``response.status_code``), decided by the HTTP status policy.
        9. Otherwise unwrap ``__cause__`` and repeat from step 1.

        The HTTP status policy is ``_status_is_retryable``: 408, 409, 429 and
        5xx retry; every other 4xx is terminal; anything else is inconclusive
        and falls through to the next check.  A ``seen`` id-set makes
        self-referential / cyclic ``__cause__`` chains terminate safely.
        """
        try:
            from openai import (
                APIConnectionError as _OAIConnectionError,
                APIStatusError as _OAIStatusError,
            )
        except ImportError:  # openai is an optional dependency
            _OAIConnectionError = _OAIStatusError = ()  # type: ignore[assignment]

        try:
            import httpx as _httpx
        except ImportError:  # httpx is an optional dependency
            _httpx = None  # type: ignore[assignment]

        seen: set[int] = set()
        current: BaseException | None = exc

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            # (1) timeouts — ours, and any builtin socket timeout
            if isinstance(current, (asyncio.TimeoutError, TimeoutError)):
                return True

            # (2) OpenAI SDK connection failure (incl. APITimeoutError)
            if _OAIConnectionError and isinstance(current, _OAIConnectionError):
                return True

            # (3) OpenAI SDK HTTP error — status policy
            if _OAIStatusError and isinstance(current, _OAIStatusError):
                verdict = _status_is_retryable(_status_of(current))
                if verdict is not None:
                    return verdict

            if _httpx is not None:
                # (4) httpx HTTP error — status policy
                if isinstance(current, _httpx.HTTPStatusError):
                    verdict = _status_is_retryable(_status_of(current))
                    if verdict is not None:
                        return verdict
                # (5) httpx transport-level fault
                elif isinstance(current, _httpx.TransportError):
                    return True

            # (6) name-based fallback for the same families
            exc_name = type(current).__name__
            if exc_name in _RETRYABLE_TRANSPORT_NAMES:
                return True
            if exc_name == "HTTPStatusError":
                verdict = _status_is_retryable(_status_of(current))
                if verdict is not None:
                    return verdict

            # (7) builtin socket errors
            if isinstance(current, (ConnectionError, BrokenPipeError, SSLError)):
                return True

            # (8) any status code carried on the exception
            verdict = _status_is_retryable(_status_of(current))
            if verdict is not None:
                return verdict

            # (9) unwrap and keep walking the cause chain
            current = getattr(current, "__cause__", None)

        return False

    async def get_raw_timing_data(self, text: str, output_path: str):
        """OpenAI TTS does not provide word-level timing data. Returns empty list."""
        return []

    async def warm_up(self):
        """Warm up the model by making a short request to reduce first-call latency."""
        if not self.initialized:
            return

        self.console.print("[bold cyan]Warming up the OpenAI TTS model...[/bold cyan]")
        warmup_file = os.path.join(
            config.AUDIO_DATA_DIR, f".warmup_openai.{self.output_format}"
        )
        try:
            await self.generate_audio("Ready.", warmup_file)
            self.console.print("[green]OpenAI TTS model is ready.[/green]")
        except Exception as e:
            self.console.print(
                f"[bold yellow]Warning: OpenAI model warm-up failed.[/bold yellow]"
            )
            self.console.print(
                f"[yellow]This may indicate an API issue or invalid voice name: {self.voice}[/yellow]"
            )
            logging.warning(f"OpenAI TTS model warm-up failed: {e}", exc_info=True)
        finally:
            if os.path.exists(warmup_file):
                try:
                    os.remove(warmup_file)
                except OSError:
                    pass
