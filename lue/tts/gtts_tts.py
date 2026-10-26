"""TTS implementation backed by Google Translate's TTS (``gTTS``).

Completely free and keyless.  gTTS requires a network connection and will
rate-limit aggressive use, but it handles its own 100-character chunking
internally, so text can be passed straight through.  Because every chunk is a
separate HTTP request, the provider budgets ``chunks * GTTS_TIMEOUT`` for the
whole sentence instead of a single per-request timeout.
"""

import asyncio
import io
import logging
import math
import re

from rich.console import Console

from .base import TTSBase, warn_overlap_deprecated
from .. import config

# Language codes Google Translate TTS accepts: a two/three letter language,
# optionally followed by a region subtag (e.g. ``en``, ``ja``, ``zh-CN``).
_LANG_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$")

# Endpoint domains Google serves: ``com``, ``co.uk``, ``co.in``, ``de``, ...
# A typo here makes every request fail, so it is rejected at startup rather
# than retried per sentence.
_TLD_PATTERN = re.compile(r"^[a-z]{2,3}(?:\.[a-z]{2})?$")

# Shown when the configured language is rejected, so the fix is obvious.
_LANG_EXAMPLES = "en, fr, de, es, it, ja, zh-CN"

# gTTS tokenises text into chunks of at most this many characters and performs
# one HTTP request per chunk, so the overall budget must scale with the number
# of chunks rather than being a single per-request timeout.
_GTTS_CHUNK_CHARS = 100

# Extra headroom on top of ``chunks * GTTS_TIMEOUT`` for DNS/TLS setup and the
# final chunk of audio arriving just inside its own timeout.
_GTTS_TOTAL_TIMEOUT_MARGIN = 10.0


def _chunk_count(text: str) -> int:
    """Estimate how many ~100-character chunks gTTS will request for ``text``."""
    return max(1, math.ceil(len(text) / _GTTS_CHUNK_CHARS))


class GTTTTS(TTSBase):
    """TTS implementation for Google Translate's TTS (gTTS)."""

    @property
    def name(self) -> str:
        return "gtts"

    @property
    def output_format(self) -> str:
        return "mp3"

    @property
    def supports_word_timing(self) -> bool:
        """gTTS does not return word-level timing metadata."""
        return False

    def __init__(self, console: Console, voice: str = None, lang: str = None):
        super().__init__(console, voice, lang)
        # gTTS selects a language code rather than a named voice. Accept
        # --lang first, then --voice, then config.GTTS_LANG (which itself
        # resolves LUE_GTTS_LANG, TTS_VOICES["gtts"] and
        # TTS_LANGUAGE_CODES["gtts"] in that order).  Blank values count as
        # "not given" so an empty flag falls back to the configured default.
        self.lang = (
            (lang or "").strip()
            or (voice or "").strip()
            or config.GTTS_LANG
        )
        self.voice = self.lang
        self.slow = config.GTTS_SLOW
        # Normalised defensively here as well: the value is interpolated into
        # the request URL, so stray whitespace/case would break every request.
        self.tld = (config.GTTS_TLD or "").strip().lower() or "com"
        self._gtts_cls = None

    def _validate_lang(self, gtts_cls) -> bool:
        """Reject unusable language codes before the first request.

        gTTS raises ``ValueError`` while handling a sentence when its language
        is unsupported, which the producer loop would retry indefinitely.
        Failing here disables TTS up front with an actionable message instead.
        """
        if not self.lang:
            self.console.print(
                "[bold red]Error: no gTTS language code configured.[/bold red]"
            )
            self.console.print(
                f"[yellow]Pass one with '--lang' (e.g. '--lang fr') or set "
                f"LUE_GTTS_LANG / TTS_VOICES[\"gtts\"] in lue/config.py. "
                f"Supported examples: {_LANG_EXAMPLES}.[/yellow]"
            )
            logging.error("gTTS language code is empty.")
            return False

        if not _LANG_PATTERN.match(self.lang):
            self.console.print(
                f"[bold red]Error: '{self.lang}' is not a valid gTTS language code.[/bold red]"
            )
            self.console.print(
                f"[yellow]Use a language code such as {_LANG_EXAMPLES}.[/yellow]"
            )
            logging.error("Invalid gTTS language code: %r", self.lang)
            return False

        # Constructing a gTTS instance validates the language locally (no
        # network) and normalises deprecated regional tags, e.g. 'en-uk'.
        try:
            probe = gtts_cls(text="a", lang=self.lang, slow=self.slow, tld=self.tld)
        except ValueError as exc:
            self.console.print(
                f"[bold red]Error: gTTS does not support language '{self.lang}'.[/bold red]"
            )
            self.console.print(
                f"[yellow]Use one of {_LANG_EXAMPLES} (see VOICES.md for the "
                f"full list), or pass '--lang' explicitly.[/yellow]"
            )
            logging.error("Unsupported gTTS language code %r: %s", self.lang, exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("gTTS language probe failed for %r: %s", self.lang, exc)
            return True

        normalized = getattr(probe, "lang", None)
        if normalized and normalized != self.lang:
            logging.info(
                "gTTS language '%s' normalised to '%s'.", self.lang, normalized
            )
            self.lang = normalized
            self.voice = normalized
        return True

    def _validate_tld(self) -> bool:
        """Reject a malformed ``LUE_GTTS_TLD`` before any request is made.

        gTTS interpolates the value straight into the request URL, so a typo
        such as ``LUE_GTTS_TLD=comm`` raises ``gTTSError`` for every sentence,
        which the pipeline would keep retrying.  Failing once at startup keeps
        the failure visible and actionable.
        """
        if _TLD_PATTERN.match(self.tld or ""):
            return True

        self.console.print(
            f"[bold red]Error: '{self.tld}' is not a valid gTTS top-level domain.[/bold red]"
        )
        self.console.print(
            "[yellow]Use a Google domain such as 'com', 'co.uk', 'co.in', "
            "'de', or 'fr'; LUE_GTTS_TLD is currently "
            f"'{self.tld}'.[/yellow]"
        )
        logging.error("Invalid gTTS top-level domain: %r", self.tld)
        return False

    async def initialize(self) -> bool:
        """Check the ``gTTS`` package, the language code and the endpoint domain."""
        try:
            from gtts import gTTS
        except ImportError:
            self.console.print(
                "[bold red]Error: 'gTTS' package not found.[/bold red]"
            )
            self.console.print(
                "[yellow]Install it with 'pip install lue-reader\\[free]' to use this TTS model.[/yellow]"
            )
            logging.error("'gTTS' is not installed.")
            return False

        if not self._validate_lang(gTTS):
            return False

        if not self._validate_tld():
            return False

        self._gtts_cls = gTTS
        self.initialized = True
        self.console.print("[green]gTTS model is available.[/green]")
        logging.info("gTTS TTS initialised (lang=%s, tld=%s, slow=%s).", self.lang, self.tld, self.slow)
        return True

    def _total_timeout(self, text: str) -> float:
        """Overall budget for synthesising ``text`` (per-request timeout × chunks).

        gTTS issues one HTTP request per ~100-character chunk; using the
        per-request timeout for the whole sentence would abort long sentences
        while their later chunks are still perfectly healthy.
        """
        return _chunk_count(text) * config.GTTS_TIMEOUT + _GTTS_TOTAL_TIMEOUT_MARGIN

    def get_overlap_seconds(self) -> float | None:
        """Deprecated; this provider has no overlap override (spec-v3 §9.1).

        Returns ``None`` so the caller keeps its own ``config.OVERLAP_SECONDS``
        default, and emits the one-time deprecation warning.
        """
        warn_overlap_deprecated()
        return None

    async def generate_audio(self, text: str, output_path: str):
        """Generate MP3 audio from ``text`` and save it to ``output_path``."""
        if not self.initialized or self._gtts_cls is None:
            raise RuntimeError("gTTS has not been initialized.")

        def _blocking_generate() -> bytes:
            # gTTS handles its own 100-char chunking internally.  Passing the
            # timeout through also bounds the underlying HTTP request, so the
            # worker thread exits instead of lingering after we stop waiting.
            tts = self._gtts_cls(
                text=text,
                lang=self.lang,
                slow=self.slow,
                tld=self.tld,
                timeout=config.GTTS_TIMEOUT,
            )
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            return buffer.getvalue()

        # asyncio.to_thread cannot be cancelled mid-flight: on timeout the
        # worker thread keeps running until its own per-request socket
        # timeout fires, and whatever it eventually returns is discarded.
        # Buffering the payload and writing only after a successful wait
        # therefore guarantees a timed-out attempt never leaves a partial or
        # stale file behind for the retry path.
        timeout = self._total_timeout(text)
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(_blocking_generate),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"gTTS request timed out after {timeout:.0f}s for "
                f"{_chunk_count(text)} chunk(s) of text: '{text[:50]}...'"
            ) from None

        with open(output_path, "wb") as f:
            f.write(payload)
