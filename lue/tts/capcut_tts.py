"""TTS implementation backed by CapCut's online text-to-speech service.

This provider wraps the optional ``capcut-tts-api`` package
(K07VN/capcut-tts-api).  The dependency is imported **lazily inside**
:meth:`CapCutTTS.initialize` so that
:class:`lue.tts_manager.TTSManager` can still *discover* the ``capcut``
provider when the SDK is not installed.

Generation deliberately avoids the SDK's high-level ``generate_speech``
helper: with ``wait=True`` it checks for a ``"success"`` status while the
CapCut backend actually reports ``"succeed"``, so the helper never returns a
usable result.  Instead this module performs the two-step flow itself:

1. ``create_tts_task(text, voice=..., resource_id=..., rate=...)`` -> task id + token
2. poll ``query_tts_task(id, token)`` until the task reports ``"succeed"``
3. decode the returned payload and download ``audio_subtitles[0].speech_url``

Everything network-facing is synchronous (the SDK and ``requests`` are
synchronous); :meth:`generate_audio` offloads the blocking work to a thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

from rich.console import Console

from .base import TTSBase
from .. import config


# ── Constants ──────────────────────────────────────────────────────────────

#: The SDK's built-in fallback voice, used when the voice catalogue cannot be
#: loaded.  Kept in sync with ``config.TTS_VOICES["capcut"]``.
_FALLBACK_VOICE = "BV074_streaming"

#: Upper bound on how long to poll a submitted task before giving up (seconds).
#: The effective deadline is ``min(_POLL_DEADLINE, CAPCUT_TTS_TIMEOUT)``.
_POLL_DEADLINE = 60.0

#: Delay between polling attempts (seconds).
_POLL_INTERVAL = 1.0

#: Task statuses that mean a task produced usable audio.  The CapCut backend
#: reports ``"succeed"``; some SDK/backend revisions spell it ``"success"``.
_SUCCESS_STATUSES = frozenset({"succeed", "success"})

#: Task statuses that mean the task will never produce audio.
_TERMINAL_STATUSES = frozenset(
    {"failed", "fail", "error", "cancelled", "canceled", "expired"}
)

#: Hard ceiling for a single audio download (bytes).  A misbehaving server
#: must not be able to stream unbounded data onto disk.
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

#: Keys that may hold the task id in a ``create_tts_task`` response.
_ID_KEYS = ("id", "task_id", "taskId", "taskID")
#: Keys that may hold the task token in a ``create_tts_task`` response.
_TOKEN_KEYS = ("token", "task_token", "taskToken")
#: Keys that may hold the task status in a ``query_tts_task`` response.
_STATUS_KEYS = ("status", "state", "task_status", "taskStatus")
#: Dict keys that may directly hold an audio URL (checked before deep search).
_URL_KEYS = (
    "speech_url",
    "speechUrl",
    "audio_url",
    "audioUrl",
    "mp3_url",
    "mp3Url",
    "play_url",
    "playUrl",
    "url",
)

#: ``__cause__``-free, name-based retry hints (duck-typed SDK stand-ins).
_RETRYABLE_EXC_NAMES = frozenset(
    {
        "RequestException",
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "SSLError",
        "ChunkedEncodingError",
        "ProxyError",
        "HTTPError",
    }
)

#: Exception names that must never be retried.
_TERMINAL_EXC_NAMES = frozenset(
    {"SignError", "CapCutSignError", "AuthError", "AuthenticationError"}
)


# ── Errors ─────────────────────────────────────────────────────────────────

class CapCutTTSError(Exception):
    """Base class for errors raised by the CapCut TTS integration."""


class CapCutTaskError(CapCutTTSError):
    """A CapCut task failed, was cancelled, or returned an unusable response.

    Treated as transient by :meth:`CapCutTTS._should_retry`: the retry loop
    re-submits the task.
    """


class SignError(CapCutTTSError):
    """CapCut request signing failed.  Terminal — never retried."""


class CapCutTTS(TTSBase):
    """TTS implementation for CapCut's online text-to-speech service."""

    @property
    def name(self) -> str:
        return "capcut"

    @property
    def output_format(self) -> str:
        return "mp3"

    @property
    def supports_word_timing(self) -> bool:
        """CapCut does not expose word-level timing metadata."""
        return False

    def __init__(self, console: Console, voice: str = None, lang: str = None):
        super().__init__(console, voice, lang)
        self._client = None
        self.resource_id = config.CAPCUT_RESOURCE_ID
        # ``self.voice`` (from TTSBase) is the single source of truth for the
        # selected voice; fall back to the configured default when unset.
        if self.voice is None:
            self.voice = config.TTS_VOICES.get(self.name)

    async def initialize(self) -> bool:
        """Import the SDK, build a client, and verify it has an HTTP session."""
        try:
            # CRITICAL: the optional dependency is imported here (not at module
            # scope) so TTSManager can discover this provider without it.
            from capcut_tts_api import CapCutClient
        except ImportError:
            self.console.print(
                "[bold red]Error: 'capcut-tts-api' package not found.[/bold red]"
            )
            self.console.print(
                '[yellow]Install it with: pip install "capcut-tts-api @ '
                'git+https://github.com/K07VN/capcut-tts-api"[/yellow]'
            )
            logging.error("'capcut-tts-api' is not installed.")
            return False

        try:
            # ``CapCutClient.__init__`` only accepts ``device``/``session``;
            # ``resource_id`` is *not* a constructor argument.  It is plumbed
            # through ``create_tts_task(resource_id=...)`` instead.
            client = CapCutClient()
        except Exception as exc:  # noqa: BLE001 - surface any SDK construction error
            self.console.print(
                f"[bold red]Error: failed to create the CapCut client: {exc}[/bold red]"
            )
            logging.error("Failed to create CapCutClient: %s", exc, exc_info=True)
            return False

        # The client only works when it holds a live ``requests.Session``;
        # a ``None`` session means ``requests`` is unavailable or init failed.
        session = getattr(client, "session", None)
        if session is None:
            self.console.print(
                "[bold red]Error: CapCut client has no HTTP session.[/bold red]"
            )
            self.console.print(
                "[yellow]The 'requests' package may be missing. "
                "Install it with 'pip install requests'.[/yellow]"
            )
            logging.error("CapCutClient.session is None (requests not installed?).")
            return False

        # Warn when the voice catalogue is empty but we are not using the
        # known-good built-in fallback voice.
        if self.voice and self.voice != _FALLBACK_VOICE:
            try:
                voices = client.list_voices()
            except Exception as exc:  # noqa: BLE001 - best-effort catalogue check
                voices = None
                logging.warning("CapCut list_voices() failed: %s", exc)
            if voices is not None and not voices:
                self.console.print(
                    "[bold yellow]Warning: CapCut returned an empty voice list; "
                    f"using '{self.voice}' anyway.[/bold yellow]"
                )

        self._client = client
        self.voice = self.voice or config.TTS_VOICES.get(self.name)
        self.initialized = True
        self.console.print("[green]CapCut TTS model is available.[/green]")
        logging.info("CapCut TTS initialised (voice=%s).", self.voice)
        return True

    async def generate_audio(self, text: str, output_path: str):
        """Generate MP3 audio for ``text`` and save it to ``output_path``.

        Applies a per-request wall-clock timeout and exponential-backoff retry
        for transient failures (network errors, task failures, rate limits).

        Each attempt writes to its own ``<output_path>.<attempt>.part`` file and
        only publishes it with :func:`os.replace` once the download succeeds.
        ``asyncio.wait_for`` cannot cancel the worker thread that
        ``asyncio.to_thread`` started, so a timed-out attempt keeps running in
        the background; giving every attempt a distinct temp file stops that
        orphan from clobbering the file a later attempt is writing, and
        ``os.replace`` guarantees readers never observe a half-written MP3.
        """
        if not self.initialized or self._client is None:
            raise RuntimeError("CapCut TTS has not been initialized.")

        from ..config import (
            CAPCUT_TTS_TIMEOUT,
            CAPCUT_TTS_MAX_RETRIES,
            CAPCUT_TTS_RETRY_BASE_DELAY,
            CAPCUT_TTS_RATE,
        )

        last_exception: BaseException | None = None

        for attempt in range(CAPCUT_TTS_MAX_RETRIES + 1):
            # Per-attempt temp file: an orphaned worker from an earlier timed-out
            # attempt can never race with this one on the same path.
            tmp_path = f"{output_path}.{attempt}.part"
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._generate_and_download,
                        text,
                        tmp_path,
                        CAPCUT_TTS_RATE,
                        CAPCUT_TTS_TIMEOUT,
                    ),
                    timeout=CAPCUT_TTS_TIMEOUT,
                )
                # Atomic publish: only a fully-downloaded file reaches output_path.
                os.replace(tmp_path, output_path)
                if attempt > 0:
                    logging.info(
                        "CapCut TTS recovered after %d retries for '%s...'",
                        attempt,
                        text[:50],
                    )
                return
            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError(
                    f"CapCut TTS timed out after {CAPCUT_TTS_TIMEOUT}s "
                    f"for text: '{text[:50]}...'"
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exception = exc
            finally:
                # On success ``os.replace`` already moved the temp file; on
                # failure remove the partial so it cannot be mistaken for real
                # output (and so retries do not accumulate ``.part`` litter).
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:  # noqa: BLE001 - cleanup is best effort
                        pass

            if not self._should_retry(last_exception):
                logging.error(
                    "CapCut TTS non-retryable error (attempt %d/%d) for '%s...': %s",
                    attempt + 1,
                    CAPCUT_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            if attempt >= CAPCUT_TTS_MAX_RETRIES:
                logging.error(
                    "CapCut TTS exhausted %d retries for '%s...': %s",
                    CAPCUT_TTS_MAX_RETRIES + 1,
                    text[:50],
                    last_exception,
                    exc_info=True,
                )
                raise last_exception

            delay = CAPCUT_TTS_RETRY_BASE_DELAY * (2 ** attempt)
            jitter = delay * 0.25 * (2 * random.random() - 1)  # ±25%
            wait_time = max(0.1, delay + jitter)

            logging.warning(
                "CapCut TTS retry %d/%d in %.1fs for '%s...': %s",
                attempt + 1,
                CAPCUT_TTS_MAX_RETRIES,
                wait_time,
                text[:50],
                type(last_exception).__name__,
            )
            await asyncio.sleep(wait_time)

    # ── Blocking work (run in a thread) ────────────────────────────────────

    def _generate_and_download(
        self, text: str, output_path: str, rate: str, timeout: float | None = None
    ) -> None:
        """Submit, poll, decode and download a CapCut TTS task (blocking)."""
        client = self._client

        # 1. Submit the synthesis task.  The SDK signature is
        # ``create_tts_task(texts, voice=..., resource_id=..., rate=...)``, so
        # every optional argument must be passed by keyword: a positional
        # ``rate`` would land in the ``resource_id`` slot.
        create = self._as_dict(
            client.create_tts_task(
                text,
                voice=self.voice,
                resource_id=self.resource_id,
                rate=rate,
            )
        )
        ret = create.get("ret")
        if str(ret) != "0":
            raise CapCutTaskError(
                "CapCut create_tts_task failed "
                f"(ret={ret!r}, msg={create.get('msg') or create.get('message') or ''!r})."
            )

        task_id, token = self._extract_task_ids(create)
        if task_id is None:
            raise CapCutTaskError("CapCut create_tts_task returned no task id.")

        # 2. Poll until the task succeeds (or the deadline expires).  The poll
        # budget is capped by the overall timeout so polling cannot outlive the
        # caller's ``wait_for`` window.
        poll_budget = _POLL_DEADLINE
        if timeout and timeout > 0:
            poll_budget = min(_POLL_DEADLINE, float(timeout))
        deadline = time.monotonic() + poll_budget
        while True:
            query = self._as_dict(client.query_tts_task(task_id, token))
            status = self._extract_status(query)

            if status in _SUCCESS_STATUSES:
                payload = self._extract_payload(query)
                url = self._extract_audio_url(payload)
                if not url:
                    # "no URL" is terminal — retrying would never help.
                    raise RuntimeError(
                        "CapCut TTS succeeded but returned no audio URL."
                    )
                self._download(url, output_path)
                return

            if status in _TERMINAL_STATUSES:
                raise CapCutTaskError(f"CapCut TTS task ended with status '{status}'.")

            if time.monotonic() >= deadline:
                raise CapCutTaskError(
                    "CapCut TTS task timed out while polling "
                    f"(after {poll_budget:.0f}s)."
                )

            time.sleep(_POLL_INTERVAL)

    def _download(self, url: str, output_path: str) -> None:
        """Stream ``url`` to ``output_path`` using ``requests``.

        Refuses non-HTTPS URLs, caps the transfer at
        :data:`_MAX_DOWNLOAD_BYTES`, and validates the MP3 magic bytes before
        returning so a corrupt/HTML error page is never published as audio.
        """
        import requests

        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"CapCut refused a non-HTTPS audio URL: {url!r}")

        with requests.get(url, stream=True, timeout=config.CAPCUT_TTS_TIMEOUT) as response:
            response.raise_for_status()

            headers = getattr(response, "headers", None)
            declared = None
            if headers is not None and hasattr(headers, "get"):
                try:
                    declared = int(headers.get("Content-Length"))
                except (TypeError, ValueError):
                    declared = None
            if declared is not None and declared > _MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    "CapCut audio download exceeds the "
                    f"{_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit."
                )

            written = 0
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > _MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "CapCut audio download exceeded the "
                            f"{_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit."
                        )
                    handle.write(chunk)

        with open(output_path, "rb") as handle:
            head = handle.read(3)
        if not self._looks_like_mp3(head):
            raise ValueError(
                "CapCut download does not look like MP3 audio "
                f"(first bytes: {head!r})."
            )

    @staticmethod
    def _looks_like_mp3(head: bytes) -> bool:
        """Return True if ``head`` begins with an ID3 tag or an MP3 frame sync."""
        if head.startswith(b"ID3"):
            return True
        return len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0

    # ── Response parsing helpers ───────────────────────────────────────────

    @staticmethod
    def _as_dict(response: Any) -> dict:
        """Coerce an SDK response into a ``dict`` (best effort)."""
        if isinstance(response, dict):
            return response
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                data = json_method()
            except Exception:  # noqa: BLE001 - malformed/opaque response
                return {}
            if isinstance(data, dict):
                return data
        return {}

    @staticmethod
    def _first_task(container: Any) -> dict | None:
        """Return ``container["tasks"][0]`` when it is a mapping, else ``None``.

        The SDK's ``generate_speech`` reads the task id/token/status from the
        first element of ``data["tasks"]``, so the parsers below consult that
        entry before the surrounding ``data``/top-level mappings.
        """
        if not isinstance(container, dict):
            return None
        tasks = container.get("tasks")
        if isinstance(tasks, (list, tuple)) and tasks and isinstance(tasks[0], dict):
            return tasks[0]
        return None

    @classmethod
    def _task_sources(cls, response: Any) -> list[dict]:
        """Ordered mappings to search: ``data["tasks"][0]``, ``data``, response."""
        sources: list[dict] = []
        if not isinstance(response, dict):
            return sources
        data = response.get("data")
        first = cls._first_task(data)
        if first is None:
            first = cls._first_task(response)
        if first is not None:
            sources.append(first)
        if isinstance(data, dict):
            sources.append(data)
        sources.append(response)
        return sources

    @classmethod
    def _extract_task_ids(cls, response: dict) -> tuple[Any, Any]:
        """Pull ``(task_id, token)`` out of a create response.

        Looks in ``data["tasks"][0]`` first, then the nested ``data`` mapping,
        then the top level.
        """
        task_id: Any = None
        token: Any = None
        for source in cls._task_sources(response):
            for key in _ID_KEYS:
                if task_id is None and source.get(key) is not None:
                    task_id = source[key]
            for key in _TOKEN_KEYS:
                if token is None and source.get(key) is not None:
                    token = source[key]

        return task_id, token

    @classmethod
    def _extract_status(cls, response: dict) -> str | None:
        """Return the lower-cased task status, or ``None`` if not present."""
        for source in cls._task_sources(response):
            for key in _STATUS_KEYS:
                value = source.get(key)
                if value is not None:
                    return str(value).lower()
        return None

    @classmethod
    def _extract_payload(cls, response: dict) -> Any:
        """Return the payload holding ``audio_subtitles`` (dict, decoded JSON or raw)."""
        sources = cls._task_sources(response)
        for source in sources:
            for key in ("payload", "result", "content", "audio_subtitles"):
                if key in source:
                    value = source[key]
                    if isinstance(value, str):
                        try:
                            return json.loads(value)
                        except (ValueError, TypeError):
                            return value
                    return value

        # Fall back to the ``data`` mapping (or the response itself) —
        # ``_extract_audio_url`` will find the URL wherever it is nested.
        if isinstance(response, dict) and isinstance(response.get("data"), dict):
            return response["data"]
        return response if isinstance(response, dict) else {}

    @classmethod
    def _extract_audio_url(cls, data: Any) -> str | None:
        """Recursively locate an audio URL inside a nested response structure.

        Handles direct URL strings, JSON-encoded strings, lists and dicts.
        Preferred keys (``speech_url`` and friends) are checked first.
        """
        if isinstance(data, str):
            if data.startswith(("http://", "https://")):
                return data
            try:
                parsed = json.loads(data)
            except (ValueError, TypeError):
                return None
            if parsed is data:
                return None
            return cls._extract_audio_url(parsed)

        if isinstance(data, dict):
            for key in _URL_KEYS:
                if key in data:
                    found = cls._extract_audio_url(data[key])
                    if found:
                        return found
            for value in data.values():
                found = cls._extract_audio_url(value)
                if found:
                    return found
            return None

        if isinstance(data, (list, tuple)):
            for item in data:
                found = cls._extract_audio_url(item)
                if found:
                    return found
            return None

        return None

    # ── Retry classification ───────────────────────────────────────────────

    @staticmethod
    def _should_retry(exc: BaseException) -> bool:
        """Return True if ``exc`` is a transient failure worth retrying.

        Retryable: ``requests`` transport errors, ``ConnectionError`` /
        timeouts, :class:`CapCutTaskError`, HTTP 429 and 5xx.

        Terminal: :class:`SignError`, ``ValueError``, and the ``RuntimeError``
        raised when a successful task yields no audio URL.
        """
        name = type(exc).__name__
        if name in _TERMINAL_EXC_NAMES:
            return False

        # Our own sign/validation errors are terminal.  (``CapCutTaskError``
        # inherits from ``Exception``, not ``ValueError``, so a plain
        # ``isinstance(exc, ValueError)`` check already excludes it.)
        if isinstance(exc, SignError):
            return False
        if isinstance(exc, ValueError):
            return False

        # Timeouts (ours, builtin, or asyncio's alias) are retryable.
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True

        # A task that failed on the server is worth re-submitting.
        if isinstance(exc, CapCutTaskError):
            return True

        # Builtin connection errors.
        if isinstance(exc, ConnectionError):
            return True

        # requests transport errors (import is optional/offline-safe).
        try:
            import requests

            if isinstance(exc, requests.exceptions.RequestException):
                return True
        except ImportError:
            pass

        # Name-based fallback for duck-typed transport errors.
        if name in _RETRYABLE_EXC_NAMES:
            return True

        # Any carried HTTP status code: 429 and 5xx retry, other 4xx terminal.
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int) or isinstance(status, bool):
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            if status == 429 or 500 <= status < 600:
                return True
            if 400 <= status < 500:
                return False

        # "No URL" RuntimeError and everything else is terminal.
        return False

    # ── TTSBase hooks ──────────────────────────────────────────────────────

    async def get_raw_timing_data(self, text: str, output_path: str):
        """CapCut provides no word-level timing data. Returns empty list."""
        return []

    async def warm_up(self):
        """Optionally synthesize a throwaway sentence to prime the session."""
        if not self.initialized:
            return

        from ..config import CAPCUT_TTS_WARMUP

        if not CAPCUT_TTS_WARMUP:
            return

        self.console.print("[bold cyan]Warming up the CapCut TTS model...[/bold cyan]")
        warmup_file = os.path.join(
            config.AUDIO_DATA_DIR, f".warmup_capcut.{self.output_format}"
        )
        try:
            await self.generate_audio("Ready.", warmup_file)
            self.console.print("[green]CapCut TTS model is ready.[/green]")
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            self.console.print(
                "[bold yellow]Warning: CapCut model warm-up failed.[/bold yellow]"
            )
            self.console.print(
                f"[yellow]This may indicate a network issue or invalid voice name: {self.voice}[/yellow]"
            )
            logging.warning("CapCut TTS model warm-up failed: %s", exc, exc_info=True)
        finally:
            if os.path.exists(warmup_file):
                try:
                    os.remove(warmup_file)
                except OSError:
                    pass
