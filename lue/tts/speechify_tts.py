"""Speechify TTS provider (``POST https://api.speechify.ai/v1/audio/speech``) for Lue.

Talks JSON to the Speechify synthesis route authenticated with a Bearer
token and decodes the base64 ``audio_data`` returned in the JSON response
into MP3 bytes at the requested output path.

The default model is ``simba-3.2`` (English, lowest TTFB) with the ``wren``
voice.

Notes:
- Unlike OpenAI, this endpoint returns **word-level timings** in the
  ``speech_marks`` field, so ``supports_word_timing`` is ``True`` and the
  reader keeps word-level highlighting.
- ``speech_marks`` timings are milliseconds; lue's timing calculator expects
  seconds, so they are divided by 1000 on the way out.
- Speechify's chunk segmentation does **not** match lue's tokenisation: it
  merges neighbouring source words into one chunk (``"October 2001"``,
  ``"5,000 copies,"``) and occasionally splits one (``"C"`` + ``"ROSSING"``).
  Every chunk does carry exact character offsets into the input, so timings
  are re-anchored onto lue's own word boundaries by interpolating between
  those offsets. That produces a 1:1 word list, which ``create_word_mapping``
  matches perfectly — otherwise its greedy fallback shifts the whole
  alignment by one after the first merge and the highlight drifts.
- Input is limited to 2000 characters per request (HTTP 400 otherwise), so
  longer text is chunked at sentence/whitespace boundaries, the MP3 segments
  are concatenated, and later segments' timings are offset by the preceding
  audio duration.
- The endpoint is subject to a **plan concurrency limit** (1 simultaneous
  request on the free plan). Requests are serialised through a semaphore and
  429 responses are retried with backoff honouring ``Retry-After``.
- Exactly one HTTP request is made per generation: ``generate_audio_with_timing``
  is overridden so timing data and audio come from the same response.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import re
import string

from rich.console import Console

from .base import TTSBase
from .. import config


class SpeechifyAPIError(Exception):
    """Raised for non-success HTTP responses from the Speechify TTS API.

    Carries ``status_code`` so the retry heuristic can distinguish transient
    failures (429/5xx) from permanent ones (400/401/404), and ``retry_after``
    so rate-limit waits can follow the server's own guidance.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class SpeechifyTTS(TTSBase):
    """TTS implementation for the Speechify ``/v1/audio/speech`` endpoint."""

    @property
    def name(self) -> str:
        return "speechify"

    @property
    def output_format(self) -> str:
        return "mp3"

    @property
    def supports_word_timing(self) -> bool:
        """Speechify returns word-level ``speech_marks`` with every response."""
        return True

    def __init__(self, console: Console, voice: str = None, lang: str = None):
        super().__init__(console, voice, lang)
        self.client = None
        self._base_url = os.environ.get(
            "SPEECHIFY_BASE_URL", "https://api.speechify.ai"
        ).rstrip("/")
        self._model = os.environ.get("LUE_SPEECHIFY_TTS_MODEL", "simba-3.2")
        # ``lang`` is intentionally unused: simba-3.2 is English-only and
        # other models route off the voice's own locale.
        if self.voice is None:
            self.voice = config.TTS_VOICES.get(self.name)
        # Serialise requests to respect the API's per-plan concurrency limit.
        # Created here (not in ``initialize``) so the guard exists for every
        # code path, including tests that drive the client directly.
        self._semaphore = asyncio.Semaphore(config.SPEECHIFY_TTS_MAX_CONCURRENT)

    @property
    def _endpoint(self) -> str:
        """Absolute URL of the TTS route."""
        return f"{self._base_url}/v1/audio/speech"

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Check for the httpx package and the SPEECHIFY_API_KEY variable."""
        try:
            import httpx
        except ImportError:
            self.console.print("[bold red]Error: 'httpx' package not found.[/bold red]")
            self.console.print(
                "[yellow]Please run 'pip install httpx' (or 'pip install lue[speechify]') "
                "to use this TTS model.[/yellow]"
            )
            logging.error("'httpx' is not installed.")
            return False

        api_key = os.environ.get("SPEECHIFY_API_KEY")
        if not api_key:
            self.console.print(
                "[bold red]Error: SPEECHIFY_API_KEY environment variable is not set.[/bold red]"
            )
            self.console.print(
                "[yellow]Please set your Speechify API key: "
                "export SPEECHIFY_API_KEY='your-key'[/yellow]"
            )
            logging.error("SPEECHIFY_API_KEY is not set.")
            return False

        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.SPEECHIFY_TTS_TIMEOUT,
        )

        self.initialized = True
        self.console.print("[green]Speechify TTS model is available.[/green]")
        return True

    # ── Generation ─────────────────────────────────────────────────────────

    async def generate_audio(self, text: str, output_path: str):
        """Generate audio from text using the Speechify TTS route.

        Long input is chunked to the API's 2000-character limit; each chunk
        gets per-request timeout and exponential-backoff retry, and the
        resulting MP3 segments are concatenated into ``output_path``.
        """
        if not self.initialized:
            raise RuntimeError("Speechify TTS has not been initialized.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        await self._synthesize_to_file(text, output_path)

    async def get_raw_timing_data(self, text: str, output_path: str):
        """Generate audio *and* return ``(word, start, end)`` timings in seconds.

        Mirrors the edge/kokoro contract: this method writes the audio file as
        a side effect so callers get both artefacts from a single billed
        request.
        """
        if not self.initialized:
            raise RuntimeError("Speechify TTS has not been initialized.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        return await self._synthesize_to_file(text, output_path)

    async def generate_audio_with_timing(self, text: str, output_path: str):
        """Generate audio and timing data from **one** request.

        The base class would call ``generate_audio`` and then
        ``get_raw_timing_data`` separately, which for a billed HTTP API means
        two charges for the same sentence. Both artefacts come from the same
        response here instead.
        """
        raw_timings = await self.get_raw_timing_data(text, output_path)

        try:
            from .. import audio
        except ImportError:
            import sys

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            import audio

        # Real duration from the file keeps total_duration accurate even when
        # the MP3 carries trailing silence past the final word. ffprobe
        # returns None (rather than raising) when the file is unreadable, so
        # treat that as a failure too.
        duration = None
        try:
            duration = await audio.get_audio_duration(output_path)
        except Exception:
            logging.warning(
                "Speechify could not read audio duration for '%s...'",
                text[:50],
                exc_info=True,
            )
        if not duration or duration <= 0:
            duration = raw_timings[-1][2] if raw_timings else 3.0

        try:
            from ..timing_calculator import process_tts_timing_data
        except ImportError:
            import sys

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            import timing_calculator

            process_tts_timing_data = timing_calculator.process_tts_timing_data

        return process_tts_timing_data(text, raw_timings, duration)

    async def _synthesize_to_file(self, text: str, output_path: str) -> list:
        """POST the text (chunked as needed), write MP3 bytes, return timings.

        Returns a list of ``(word, start_seconds, end_seconds)`` tuples.
        """
        chunks = self._chunk_text(text, config.SPEECHIFY_TTS_MAX_CHARS)

        segments: list[tuple[bytes, list]] = []
        for chunk in chunks:
            segments.append(await self._request(chunk))

        with open(output_path, "wb") as f:
            for audio_bytes, _ in segments:
                f.write(audio_bytes)

        # Offset later segments by the preceding audio's estimated duration so
        # word timings stay aligned across the concatenated file. The running
        # offset must accumulate from its own previous value, since ``timings``
        # holds raw (unshifted) values for this segment.
        merged: list[tuple[str, float, float]] = []
        offset = 0.0
        for _, timings in segments:
            for word, start, end in timings:
                merged.append((word, start + offset, end + offset))
            if timings:
                offset += timings[-1][2] + 0.15
        return merged

    async def _request(self, text: str) -> tuple[bytes, list]:
        """POST one chunk under the concurrency semaphore; retry transient errors."""
        from ..config import (
            SPEECHIFY_TTS_TIMEOUT,
            SPEECHIFY_TTS_MAX_RETRIES,
            SPEECHIFY_TTS_RETRY_BASE_DELAY,
        )

        last_exception = None

        for attempt in range(SPEECHIFY_TTS_MAX_RETRIES + 1):
            try:
                async with self._semaphore:
                    resp = await asyncio.wait_for(
                        self._post_tts(text),
                        timeout=SPEECHIFY_TTS_TIMEOUT,
                    )
                audio_bytes, timings = self._parse_response(resp, text)

                if attempt > 0:
                    logging.info(
                        "Speechify TTS recovered after %d retries for '%s...'",
                        attempt,
                        text[:50],
                    )
                return audio_bytes, timings

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"Speechify TTS timed out after {SPEECHIFY_TTS_TIMEOUT}s "
                    f"for text: '{text[:50]}...'"
                )
            except Exception as e:
                last_exception = e

            # --- Should we retry? -----------------------------------------
            if not self._should_retry(last_exception):
                logging.error(
                    "Speechify TTS non-retryable error (attempt %d/%d) "
                    "for '%s...': %s",
                    attempt + 1,
                    SPEECHIFY_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            # --- Exhausted retries? ---------------------------------------
            if attempt >= SPEECHIFY_TTS_MAX_RETRIES:
                logging.error(
                    "Speechify TTS exhausted %d retries for '%s...': %s",
                    SPEECHIFY_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            # --- Backoff and retry ----------------------------------------
            delay = SPEECHIFY_TTS_RETRY_BASE_DELAY * (2 ** attempt)
            jitter = delay * 0.25 * (2 * random.random() - 1)  # ±25%
            wait_time = max(0.1, delay + jitter)

            # Honour the server's Retry-After on 429 when it is longer.
            retry_after = getattr(last_exception, "retry_after", None)
            if retry_after:
                wait_time = max(wait_time, float(retry_after))

            logging.warning(
                "Speechify TTS retry %d/%d in %.1fs for '%s...': %s",
                attempt + 1,
                SPEECHIFY_TTS_MAX_RETRIES,
                wait_time,
                text[:50],
                type(last_exception).__name__,
            )
            await asyncio.sleep(wait_time)

    async def _post_tts(self, text: str):
        """Send one JSON request to the TTS route."""
        payload = {
            "input": text,
            "voice_id": self.voice,
            "audio_format": self.output_format,
            "model": self._model,
            "options": {
                "text_normalization": config.SPEECHIFY_TTS_TEXT_NORMALIZATION,
            },
        }
        return await self.client.post(self._endpoint, json=payload)

    def _parse_response(self, resp, text: str) -> tuple[bytes, list]:
        """Decode one response into ``(mp3_bytes, word_timings_seconds)``."""
        if resp.status_code != 200:
            raise SpeechifyAPIError(
                self._error_message(resp),
                status_code=resp.status_code,
                retry_after=self._retry_after(resp),
            )

        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type:
            # Some deployments may return raw audio; accept it as-is.
            return resp.content, []

        try:
            data = resp.json()
        except Exception as e:
            raise SpeechifyAPIError(
                f"Speechify TTS returned invalid JSON for text: '{text[:50]}...'",
                status_code=200,
            ) from e

        audio_b64 = data.get("audio_data")
        if not audio_b64:
            raise SpeechifyAPIError(
                "Speechify TTS response contained no audio_data.",
                status_code=200,
            )

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            raise SpeechifyAPIError(
                "Speechify TTS response contained undecodable audio_data.",
                status_code=200,
            ) from e

        return audio_bytes, self._extract_timings(data.get("speech_marks"), text)

    @staticmethod
    def _source_words_with_offsets(text: str) -> list[tuple[str, int, int]]:
        """Tokenise ``text`` exactly like lue's timing calculator, keeping spans.

        Mirrors ``timing_calculator._get_highlightable_words`` (split on
        whitespace, drop tokens with no ASCII alphanumeric, strip leading and
        trailing punctuation) but also returns each word's ``[start, end)``
        character span so timings can be anchored to the source rather than to
        Speechify's own segmentation.

        Matching the tokenizer exactly is what makes the resulting list line up
        1:1 with ``reader.current_sentence_words``.
        """
        words: list[tuple[str, int, int]] = []
        pos = 0
        for token in text.split():
            start = text.find(token, pos)
            if start < 0:
                start = pos  # defensive: tokens come from text.split()
            pos = start + len(token)

            if not re.search(r"[a-zA-Z0-9]", token):
                continue
            core = token.strip(string.punctuation)
            if not core:
                continue
            lead = len(token) - len(token.lstrip(string.punctuation))
            words.append((core, start + lead, start + lead + len(core)))
        return words

    @staticmethod
    def _time_at(position: int, anchors: list[tuple[int, int, float, float]]) -> float:
        """Interpolate the audio time (seconds) at ``position``.

        ``anchors`` are ``(char_start, char_end, time_start, time_end)``. Inside
        a chunk time is interpolated linearly; inside the gap between two chunks
        it is interpolated between them; outside the extremes it clamps. This
        lets a source word that spans a merge or a split resolve correctly even
        though Speechify never reported it as a unit.
        """
        first_c0, _first_c1, first_ts, _first_te = anchors[0]
        if position <= first_c0:
            return first_ts

        prev: tuple[int, int, float, float] | None = None
        for c0, c1, ts, te in anchors:
            if c0 <= position <= c1:
                span = c1 - c0
                if span <= 0:
                    return ts
                return ts + (te - ts) * (position - c0) / span
            if c0 > position:
                if prev is None:
                    return ts
                _p0, p1, _pts, pte = prev
                if c0 <= p1:
                    return pte
                return pte + (ts - pte) * (position - p1) / (c0 - p1)
            prev = (c0, c1, ts, te)

        return anchors[-1][3]

    @staticmethod
    def _extract_timings(speech_marks, text: str = "") -> list:
        """Convert ``speech_marks`` into ``(word, start_seconds, end_seconds)``.

        ``speech_marks`` is a single sentence object carrying ``chunks`` of word
        entries (occasionally a list of such objects). Values are milliseconds
        and are converted to the seconds lue expects.

        When the chunks carry character offsets (the normal case) the timings
        are re-anchored onto **lue's own word boundaries** via
        :meth:`_source_words_with_offsets`, ignoring Speechify's segmentation
        entirely. This matters because Speechify merges neighbours into one
        chunk (``"October 2001"``, ``"5,000 copies,"``); feeding those merged
        chunks to ``create_word_mapping`` makes its greedy fallback shift the
        alignment permanently after the first merge, so the highlight walks off
        the sentence. Anchored output is always 1:1 with the source, which the
        mapping short-circuits as a perfect match.

        Falls back to the raw chunk values when offsets are absent, so a
        response shape change degrades rather than breaks.
        """
        if not speech_marks:
            return []

        entries = (
            speech_marks if isinstance(speech_marks, list) else [speech_marks]
        )

        chunks: list[dict] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for chunk in entry.get("chunks") or []:
                if not isinstance(chunk, dict):
                    continue
                chunk_type = chunk.get("type")
                if chunk_type is not None and chunk_type != "word":
                    continue
                word = chunk.get("value")
                start = chunk.get("start_time")
                end = chunk.get("end_time")
                if word is None or start is None or end is None:
                    continue
                chunks.append({
                    "word": word,
                    "start": start / 1000.0,
                    "end": end / 1000.0,
                    "c0": chunk.get("start"),
                    "c1": chunk.get("end"),
                })

        if not chunks:
            return []

        have_offsets = all(
            isinstance(c["c0"], int) and isinstance(c["c1"], int)
            for c in chunks
        )
        if not have_offsets or not text:
            return [(c["word"], c["start"], c["end"]) for c in chunks]

        anchors = sorted(
            ((c["c0"], c["c1"], c["start"], c["end"]) for c in chunks),
            key=lambda a: (a[0], a[1]),
        )
        source_words = SpeechifyTTS._source_words_with_offsets(text)
        if not source_words:
            return [(c["word"], c["start"], c["end"]) for c in chunks]

        timings: list[tuple[str, float, float]] = []
        for word, c0, c1 in source_words:
            start = SpeechifyTTS._time_at(c0, anchors)
            end = SpeechifyTTS._time_at(c1, anchors)
            if end < start:
                end = start
            timings.append((word, start, end))
        return timings

    @staticmethod
    def _retry_after(resp) -> float | None:
        """Parse a ``Retry-After`` header into seconds, if present."""
        raw = resp.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_message(resp) -> str:
        """Extract a readable error message from an error response."""
        try:
            data = resp.json()
        except Exception:
            return f"Speechify TTS request failed: HTTP {resp.status_code}"

        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
        else:
            message = error
        message = message or data.get("message")
        return (
            f"Speechify TTS request failed: HTTP {resp.status_code} "
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
        - HTTP 429 (rate limit / plan concurrency limit) and 5xx server errors
        Non-retryable: other 4xx (400 bad input, 401 auth, 404 unknown voice).
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

        # HTTP status code errors (429 concurrency/rate limit, 5xx errors)
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

    # ── Warm-up ────────────────────────────────────────────────────────────

    async def warm_up(self):
        """Warm up the model by making a short request to reduce first-call latency."""
        if not self.initialized:
            return

        self.console.print("[bold cyan]Warming up the Speechify TTS model...[/bold cyan]")
        warmup_file = os.path.join(
            config.AUDIO_DATA_DIR, f".warmup_speechify.{self.output_format}"
        )
        try:
            await self.generate_audio("Ready.", warmup_file)
            self.console.print("[green]Speechify TTS model is ready.[/green]")
        except Exception as e:
            self.console.print(
                "[bold yellow]Warning: Speechify model warm-up failed.[/bold yellow]"
            )
            self.console.print(
                f"[yellow]This may indicate an API issue or invalid voice name: {self.voice}[/yellow]"
            )
            logging.warning(f"Speechify TTS model warm-up failed: {e}", exc_info=True)
        finally:
            if os.path.exists(warmup_file):
                try:
                    os.remove(warmup_file)
                except OSError:
                    pass
