"""TTS implementation backed by Google Translate's TTS (``gTTS``).

Completely free and keyless.  gTTS requires a network connection and will
rate-limit aggressive use, but it handles its own 100-character chunking
internally, so text can be passed straight through.
"""

import asyncio
import io
import logging
import re

from rich.console import Console

from .base import TTSBase, warn_overlap_deprecated
from .. import config

# Language codes Google Translate TTS accepts: a two/three letter language,
# optionally followed by a region subtag (e.g. ``en``, ``ja``, ``zh-CN``).
_LANG_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$")

# Shown when the configured language is rejected, so the fix is obvious.
_LANG_EXAMPLES = "en, fr, de, es, it, ja, zh-CN"


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
        self.tld = config.GTTS_TLD
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

    async def initialize(self) -> bool:
        """Check the ``gTTS`` package and the configured language code."""
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

        self._gtts_cls = gTTS
        self.initialized = True
        self.console.print("[green]gTTS model is available.[/green]")
        logging.info("gTTS TTS initialised (lang=%s, tld=%s, slow=%s).", self.lang, self.tld, self.slow)
        return True

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

        # A worker thread cannot be cancelled, so buffer the payload and only
        # write it once the request finished in time.  A timed-out attempt
        # therefore never leaves a partial/stale file behind for the retry.
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(_blocking_generate),
                timeout=config.GTTS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"gTTS request timed out after {config.GTTS_TIMEOUT}s "
                f"for text: '{text[:50]}...'"
            ) from None

        with open(output_path, "wb") as f:
            f.write(payload)
