import os
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
        """Generate audio from text using OpenAI Speech API and save to file."""
        if not self.initialized:
            raise RuntimeError("OpenAI TTS has not been initialized.")
        try:
            async with self.client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self.voice,
                input=text,
                response_format="mp3",
            ) as response:
                await response.stream_to_file(output_path)
        except Exception as e:
            logging.error(
                f"OpenAI TTS audio generation failed for text: '{text[:50]}...'",
                exc_info=True,
            )
            raise

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
