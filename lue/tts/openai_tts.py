import asyncio
import os
import random
import logging
from rich.console import Console

from .base import TTSBase
from .. import config


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
    def _should_retry(exc: Exception) -> bool:
        """Return True if the exception should trigger a retry.

        Retries on:
        - asyncio.TimeoutError (our own per-request timeout)
        - httpx.TimeoutException (connection-level timeout)
        - httpx.ConnectError, httpx.RemoteProtocolError (network issues)
        - HTTP 429 rate limit (from httpx.HTTPStatusError)
        - HTTP 5xx server errors (from httpx.HTTPStatusError)
        - SSLError, ConnectionResetError, BrokenPipeError (network issues)
        """
        exc_name = type(exc).__name__

        # Our per-request timeout
        if isinstance(exc, asyncio.TimeoutError):
            return True

        # httpx-specific network errors
        if exc_name in (
            "TimeoutException",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "NetworkError",
        ):
            return True

        # Transport-level errors
        if isinstance(exc, (ConnectionError, BrokenPipeError)):
            return True

        # SSL errors
        try:
            from ssl import SSLError
            if isinstance(exc, SSLError):
                return True
        except ImportError:
            pass

        # HTTP status code errors (429 rate limit, 5xx server errors)
        if hasattr(exc, "status_code"):
            status = getattr(exc, "status_code", None)
            if status is not None:
                if status == 429 or (500 <= status < 600):
                    return True
                return False  # 4xx (except 429) are non-retryable

        # Check if it wraps an httpx.HTTPStatusError
        if exc_name in ("HTTPStatusError",):
            if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                status = exc.response.status_code
                if status == 429 or (500 <= status < 600):
                    return True
            return False

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
