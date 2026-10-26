"""Configuration settings for the Lue eBook reader."""

import os
from platformdirs import user_data_dir, user_cache_dir

# Default TTS model
DEFAULT_TTS_MODEL = "edge"

# Default voices for TTS models
TTS_VOICES = {
    "edge": "en-US-JennyNeural",
    "kokoro": "af_heart",
    "openai": "alloy",
    "gtts": "en",  # free Google Translate TTS: default language code
}

# Language codes for TTS models that require them
TTS_LANGUAGE_CODES = {
    "kokoro": "a",  # a=English, e=Spanish, j=Japanese, etc.
    "gtts": "en",  # fallback alias for TTS_VOICES["gtts"]
}

# TTS model-specific seconds of overlap between sentences (overrides default OVERLAP_SECONDS if specified)
TTS_OVERLAP_SECONDS = {
    "kokoro": 0.6,
}

# ── Free TTS providers (no API key required) ───────────────────────────────

# gTTS (Google Translate TTS).


def _resolve_gtts_lang(env_lang: str, voices: dict, codes: dict) -> str:
    """Resolve the default gTTS language code.

    Precedence: ``LUE_GTTS_LANG``, then ``TTS_VOICES["gtts"]`` (the knob
    documented in VOICES.md), then ``TTS_LANGUAGE_CODES["gtts"]`` (fallback
    alias).  ``--lang``/``--voice`` are applied later by the provider and take
    priority over everything resolved here.
    """
    return (
        (env_lang or "").strip()
        or voices.get("gtts")
        or codes.get("gtts")
        or "en"
    )


GTTS_LANG = _resolve_gtts_lang(
    os.environ.get("LUE_GTTS_LANG", ""), TTS_VOICES, TTS_LANGUAGE_CODES
)
GTTS_TLD = os.environ.get("LUE_GTTS_TLD", "com")
GTTS_SLOW = os.environ.get("LUE_GTTS_SLOW", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Per-request timeout (seconds) for the blocking gTTS network call.  The call
# runs in a worker thread, so a hung request is abandoned instead of stalling
# the audio producer; the producer's retry path then takes over.
_gtts_timeout_raw = float(os.environ.get("LUE_GTTS_TIMEOUT", "30"))
GTTS_TIMEOUT = max(_gtts_timeout_raw, 5.0) if _gtts_timeout_raw > 0 else 30.0

# OpenAI TTS retry settings
# Per-request generation timeout (seconds). Requests that exceed this
# are cancelled and retried if retries remain.
_tts_timeout_raw = float(os.environ.get("LUE_OPENAI_TTS_TIMEOUT", "30"))
OPENAI_TTS_TIMEOUT = max(_tts_timeout_raw, 5.0) if _tts_timeout_raw > 0 else 30.0

# Maximum retry attempts for transient failures (timeouts, connection
# errors, server errors, rate limits).
_tts_retries_raw = int(os.environ.get("LUE_OPENAI_TTS_MAX_RETRIES", "3"))
OPENAI_TTS_MAX_RETRIES = max(min(_tts_retries_raw, 10), 0) if _tts_retries_raw >= 0 else 3

# Base delay in seconds for exponential backoff.  Attempt n waits
# base * 2**(n-1) seconds before retrying.
_tts_retry_delay_raw = float(os.environ.get("LUE_OPENAI_TTS_RETRY_BASE_DELAY", "1.0"))
OPENAI_TTS_RETRY_BASE_DELAY = max(_tts_retry_delay_raw, 0.1) if _tts_retry_delay_raw > 0 else 1.0

# Audio processing settings
AUDIO_DATA_DIR = user_cache_dir("lue")
os.makedirs(AUDIO_DATA_DIR, exist_ok=True)
AUDIO_BUFFERS = [os.path.join(AUDIO_DATA_DIR, f"buffer_{i}") for i in range(6)]
MAX_QUEUE_SIZE = 4
OVERLAP_SECONDS = 0.5  # Seconds of overlap between sentences

# Progress tracking settings
PROGRESS_FILE_DIR = user_data_dir("lue")
os.makedirs(PROGRESS_FILE_DIR, exist_ok=True)

# General settings
SHOW_ERRORS_ON_EXIT = True

# PDF parsing settings
PDF_FILTERS_ENABLED = (
    False  # You can also enable this with the --filter or -f command-line option
)
PDF_FILTER_HEADERS = True  # Filter headers in top margin of pages
PDF_FILTER_FOOTNOTES = (
    True  # Filter page numbers and footnotes in bottom margin of pages
)

# PDF filtering thresholds (only used when respective filters are enabled)
PDF_HEADER_MARGIN = 0.1  # Top 10% of page considered header area
PDF_FOOTNOTE_MARGIN = 0.1  # Bottom 10% of page considered footnote area

# UI settings
SMOOTH_SCROLLING_ENABLED = True  # Enable smooth scrolling for keyboard navigation
UI_MODE = 2  # 0=minimal (text only), 1=medium (top bar only), 2=full (default), 3=speed reading

# Highlighting settings
SENTENCE_HIGHLIGHTING_ENABLED = True  # Enable sentence-level highlighting
WORD_HIGHLIGHT_MODE = 1  # 0=off, 1=normal highlighting, 2=standout highlighting

# Keyboard settings
# Can be set to "default", "vim", or a path to a custom keyboard shortcuts JSON file
CUSTOM_KEYBOARD_SHORTCUTS = "default"
