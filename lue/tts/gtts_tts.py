"""TTS implementation backed by Google Translate's TTS (``gTTS``).

Completely free and keyless.  gTTS requires a network connection and will
rate-limit aggressive use, but it handles its own 100-character chunking
internally, so text can be passed straight through.
"""

import asyncio
import io
import logging

from rich.console import Console

from .base import TTSBase, warn_overlap_deprecated
from .. import config


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
        # --lang first, then --voice, then the config/env default.
        self.lang = lang or voice or config.GTTS_LANG
        self.voice = self.lang
        self.slow = config.GTTS_SLOW
        self.tld = config.GTTS_TLD
        self._gtts_cls = None

    async def initialize(self) -> bool:
        """Check that the ``gTTS`` package is importable."""
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

        self._gtts_cls = gTTS
        self.initialized = True
        self.console.print("[green]gTTS model is available.[/green]")
        logging.info("gTTS TTS initialised (lang=%s, tld=%s, slow=%s).", self.lang, self.tld, self.slow)
        return True

    def get_overlap_seconds(self) -> float | None:
        """Deprecated; crossfade/overlap was removed (spec-v3 §9.1).

        The gapless persistent sink concatenates decoded PCM, so no overlap is
        applied.  This always returns ``None`` and emits a one-time warning.
        """
        warn_overlap_deprecated()
        return None

    async def generate_audio(self, text: str, output_path: str):
        """Generate MP3 audio from ``text`` and save it to ``output_path``."""
        if not self.initialized or self._gtts_cls is None:
            raise RuntimeError("gTTS has not been initialized.")

        def _blocking_generate():
            # gTTS handles its own 100-char chunking internally.
            tts = self._gtts_cls(
                text=text,
                lang=self.lang,
                slow=self.slow,
                tld=self.tld,
            )
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            with open(output_path, "wb") as f:
                f.write(buffer.getvalue())

        await asyncio.to_thread(_blocking_generate)
