"""NanoGPT TTS provider (``POST https://nano-gpt.com/api/tts``) for Lue.

Talks JSON to the NanoGPT TTS route authenticated with the ``x-api-key``
header and writes the returned MP3 bytes to the requested output path.

The default model is ``xai-tts`` (SpaceXAI TTS, Runware-hosted) with the
``Rex`` voice. The route accepts the text under either the ``text`` key
(official docs) or the ``input`` key (media-integration-spec/v2); both are
sent so either contract is satisfied.

Notes:
- ``xai-tts`` is synchronous: it returns ``audio/mpeg`` bytes directly.
  JSON (``audioUrl``) and async ticket (HTTP 202) responses are also
  handled for provider compatibility.
- No word-level timing data is available, so the reader falls back to
  sentence-level highlighting (``supports_word_timing`` is ``False``).
- Input text is limited to 5000 characters per request; longer text is
  chunked at sentence/whitespace boundaries and the MP3s are concatenated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re

from rich.console import Console

from .base import TTSBase
from .. import config


class NanoGPTAPIError(Exception):
    """Raised for non-success HTTP responses from the NanoGPT TTS API.

    Carries ``status_code`` so the retry heuristic can distinguish
    transient failures (429/5xx) from permanent ones (400/401/402).
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class NanoGPTTTS(TTSBase):
    """TTS implementation for the NanoGPT ``/api/tts`` endpoint (xai-tts)."""

    @property
    def name(self) -> str:
        return "nanogpt"

    @property
    def output_format(self) -> str:
        return "mp3"

    @property
    def supports_word_timing(self) -> bool:
        """NanoGPT TTS does not provide word-level timing data."""
        return False

    def __init__(self, console: Console, voice: str = None, lang: str = None):
        super().__init__(console, voice, lang)
        self.client = None
        self._base_url = os.environ.get(
            "NANOGPT_BASE_URL", "https://nano-gpt.com/api"
        ).rstrip("/")
        self._model = os.environ.get("NANOGPT_TTS_MODEL", "xai-tts")
        # ``lang`` is intentionally ignored: xai-tts auto-detects the language.
        if self.voice is None:
            self.voice = config.TTS_VOICES.get(self.name)

    @property
    def _endpoint(self) -> str:
        """Absolute URL of the TTS route."""
        return f"{self._base_url}/tts"

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Check for the httpx package and the NANOGPT_API_KEY variable."""
        try:
            import httpx
        except ImportError:
            self.console.print("[bold red]Error: 'httpx' package not found.[/bold red]")
            self.console.print(
                "[yellow]Please run 'pip install httpx' (or 'pip install lue[nanogpt]') "
                "to use this TTS model.[/yellow]"
            )
            logging.error("'httpx' is not installed.")
            return False

        api_key = os.environ.get("NANOGPT_API_KEY")
        if not api_key:
            self.console.print(
                "[bold red]Error: NANOGPT_API_KEY environment variable is not set.[/bold red]"
            )
            self.console.print(
                "[yellow]Please set your NanoGPT API key: export NANOGPT_API_KEY='your-key'[/yellow]"
            )
            logging.error("NANOGPT_API_KEY is not set.")
            return False

        self.client = httpx.AsyncClient(
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=config.NANOGPT_TTS_TIMEOUT,
        )

        self.initialized = True
        self.console.print("[green]NanoGPT TTS model is available.[/green]")
        return True

    # ── Generation ─────────────────────────────────────────────────────────

    async def generate_audio(self, text: str, output_path: str):
        """Generate audio from text using the NanoGPT TTS route.

        Long input is chunked to the API's 5000-character limit; each chunk
        gets per-request timeout and exponential-backoff retry, and the
        resulting MP3 segments are concatenated into ``output_path``.
        """
        if not self.initialized:
            raise RuntimeError("NanoGPT TTS has not been initialized.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        chunks = self._chunk_text(text, config.NANOGPT_TTS_MAX_CHARS)
        segments = [await self._synthesize_chunk(chunk) for chunk in chunks]

        with open(output_path, "wb") as f:
            for segment in segments:
                f.write(segment)

    async def _synthesize_chunk(self, text: str) -> bytes:
        """POST one chunk with timeout + exponential-backoff retry; return MP3 bytes."""
        from ..config import (
            NANOGPT_TTS_TIMEOUT,
            NANOGPT_TTS_MAX_RETRIES,
            NANOGPT_TTS_RETRY_BASE_DELAY,
        )

        last_exception = None

        for attempt in range(NANOGPT_TTS_MAX_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    self._post_tts(text),
                    timeout=NANOGPT_TTS_TIMEOUT,
                )

                # Async ticket: poll outside the request timeout so a slow
                # job is not mistaken for a hung connection (which would
                # re-POST and double-bill).
                if resp.status_code == 202:
                    audio = await self._poll_ticket(resp.json())
                else:
                    audio = await self._extract_audio(resp)

                if attempt > 0:
                    logging.info(
                        "NanoGPT TTS recovered after %d retries for '%s...'",
                        attempt,
                        text[:50],
                    )
                return audio

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"NanoGPT TTS timed out after {NANOGPT_TTS_TIMEOUT}s "
                    f"for text: '{text[:50]}...'"
                )
            except Exception as e:
                last_exception = e

            # --- Should we retry? -----------------------------------------
            if not self._should_retry(last_exception):
                logging.error(
                    "NanoGPT TTS non-retryable error (attempt %d/%d) "
                    "for '%s...': %s",
                    attempt + 1,
                    NANOGPT_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            # --- Exhausted retries? ---------------------------------------
            if attempt >= NANOGPT_TTS_MAX_RETRIES:
                logging.error(
                    "NanoGPT TTS exhausted %d retries for '%s...': %s",
                    NANOGPT_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            # --- Backoff and retry ----------------------------------------
            delay = NANOGPT_TTS_RETRY_BASE_DELAY * (2 ** attempt)
            jitter = delay * 0.25 * (2 * random.random() - 1)  # ±25%
            wait_time = max(0.1, delay + jitter)

            logging.warning(
                "NanoGPT TTS retry %d/%d in %.1fs for '%s...': %s",
                attempt + 1,
                NANOGPT_TTS_MAX_RETRIES,
                wait_time,
                text[:50],
                type(last_exception).__name__,
            )
            await asyncio.sleep(wait_time)

    async def _post_tts(self, text: str):
        """Send one JSON request to the TTS route."""
        payload = {
            # Official docs use ``text``; media-integration-spec/v2 uses
            # ``input``. The route accepts both (verified live), so send
            # both for contract compatibility.
            "text": text,
            "input": text,
            "model": self._model,
            "voice": self.voice,
            "speed": config.NANOGPT_TTS_SPEED,
        }
        return await self.client.post(self._endpoint, json=payload)

    async def _extract_audio(self, resp) -> bytes:
        """Return audio bytes from a synchronous (non-202) response.

        Handles both binary audio (``audio/mpeg``) and JSON with an
        ``audioUrl`` that must be downloaded.
        """
        if resp.status_code != 200:
            raise NanoGPTAPIError(
                self._error_message(resp),
                status_code=resp.status_code,
            )

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            audio_url = data.get("audioUrl")
            if not audio_url:
                raise NanoGPTAPIError(
                    "NanoGPT TTS response contained no audioUrl.",
                    status_code=200,
                )
            return await self._download(audio_url)

        return resp.content

    async def _download(self, url: str) -> bytes:
        """Download generated audio from a URL."""
        resp = await asyncio.wait_for(
            self.client.get(url, follow_redirects=True),
            timeout=config.NANOGPT_TTS_TIMEOUT,
        )
        if resp.status_code != 200:
            raise NanoGPTAPIError(
                f"Failed to download audio from {url}: "
                f"HTTP {resp.status_code} {resp.text[:200]}",
                status_code=resp.status_code,
            )
        return resp.content

    async def _poll_ticket(self, ticket: dict) -> bytes:
        """Poll an async (HTTP 202) ticket until the audio URL is ready."""
        run_id = ticket.get("runId")
        if not run_id:
            raise NanoGPTAPIError(
                "NanoGPT TTS 202 response contained no runId.",
                status_code=202,
            )

        params = {"runId": run_id, "model": self._model}
        if isinstance(ticket.get("cost"), (int, float)):
            params["cost"] = str(ticket["cost"])
        if ticket.get("paymentSource"):
            params["paymentSource"] = ticket["paymentSource"]
        params["isApiRequest"] = "true"

        status_url = f"{self._base_url}/tts/status"
        for _ in range(config.NANOGPT_TTS_POLL_MAX_ATTEMPTS):
            await asyncio.sleep(config.NANOGPT_TTS_POLL_INTERVAL)

            resp = await asyncio.wait_for(
                self.client.get(status_url, params=params),
                timeout=config.NANOGPT_TTS_TIMEOUT,
            )
            if resp.status_code != 200:
                raise NanoGPTAPIError(
                    self._error_message(resp),
                    status_code=resp.status_code,
                )

            data = resp.json()
            status = data.get("status")
            if status == "completed":
                audio_url = data.get("audioUrl")
                if not audio_url:
                    raise NanoGPTAPIError(
                        "NanoGPT TTS job completed without an audioUrl.",
                        status_code=200,
                    )
                return await self._download(audio_url)
            if status == "error":
                raise RuntimeError(
                    f"NanoGPT TTS job failed: {data.get('error', 'unknown error')}"
                )

        raise asyncio.TimeoutError(
            f"NanoGPT TTS job {run_id} did not complete within "
            f"{config.NANOGPT_TTS_POLL_MAX_ATTEMPTS} polls."
        )

    @staticmethod
    def _error_message(resp) -> str:
        """Extract a readable error message from an error response."""
        try:
            data = resp.json()
        except Exception:
            return f"NanoGPT TTS request failed: HTTP {resp.status_code}"

        error = data.get("error")
        if isinstance(error, dict):
            error = error.get("message") or error.get("code") or str(error)
        message = data.get("message") or error
        return (
            f"NanoGPT TTS request failed: HTTP {resp.status_code} "
            f"{message if message else resp.text[:200]}"
        )

    # ── Text chunking ──────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        """Split ``text`` into pieces of at most ``max_chars`` characters.

        Prefers sentence boundaries (``.``, ``!``, ``?``), then any
        whitespace, and only hard-splits when a single token exceeds the
        limit.
        """
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break

            window = remaining[:max_chars]

            # Prefer the last sentence boundary so cuts don't break prosody;
            # fall back to whitespace, then to a hard split.
            sentence_ends = list(re.finditer(r"[.!?]\s", window))
            if sentence_ends:
                cut = sentence_ends[-1].end()
            else:
                whitespace = list(re.finditer(r"\s", window))
                cut = whitespace[-1].end() if whitespace else max_chars

            if cut <= 0:
                cut = max_chars

            chunk = remaining[:cut]
            rest = remaining[cut:]
            # Keep exact concatenation: only strip a separator when the cut
            # already landed on/after it.
            if rest.startswith((" ", "\t", "\n")):
                rest = rest[1:]
            chunks.append(chunk)
            remaining = rest or chunk[-1:]  # guarantee progress on empty rest

        return chunks

    # ── Retry heuristic ────────────────────────────────────────────────────

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        """Return True if the exception should trigger a retry.

        Retries on:
        - asyncio.TimeoutError (our own per-request timeout)
        - httpx network/timeout errors
        - Connection-level and SSL errors
        - HTTP 429 rate limit and 5xx server errors
        Non-retryable: other 4xx (400 bad input, 401 auth, 402 balance).
        """
        exc_name = type(exc).__name__

        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True

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

        if isinstance(exc, (ConnectionError, BrokenPipeError)):
            return True

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

        if exc_name == "HTTPStatusError":
            if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                status = exc.response.status_code
                if status == 429 or (500 <= status < 600):
                    return True
            return False

        return False

    # ── Timing / warm-up ───────────────────────────────────────────────────

    async def get_raw_timing_data(self, text: str, output_path: str):
        """NanoGPT TTS does not provide word-level timing data. Returns empty list."""
        return []

    async def warm_up(self):
        """Warm up the model by making a short request to reduce first-call latency."""
        if not self.initialized:
            return

        self.console.print("[bold cyan]Warming up the NanoGPT TTS model...[/bold cyan]")
        warmup_file = os.path.join(
            config.AUDIO_DATA_DIR, f".warmup_nanogpt.{self.output_format}"
        )
        try:
            await self.generate_audio("Ready.", warmup_file)
            self.console.print("[green]NanoGPT TTS model is ready.[/green]")
        except Exception as e:
            self.console.print(
                "[bold yellow]Warning: NanoGPT model warm-up failed.[/bold yellow]"
            )
            self.console.print(
                f"[yellow]This may indicate an API issue or invalid voice name: {self.voice}[/yellow]"
            )
            logging.warning(f"NanoGPT TTS model warm-up failed: {e}", exc_info=True)
        finally:
            if os.path.exists(warmup_file):
                try:
                    os.remove(warmup_file)
                except OSError:
                    pass
