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
    # CapCut's built-in streaming voice. Matches the SDK's fallback resource
    # (7102355709945188865), so it works even when Voice.json is unavailable.
    "capcut": "BV074_streaming",
}

# Language codes for TTS models that require them
TTS_LANGUAGE_CODES = {
    "kokoro": "a",  # a=English, e=Spanish, j=Japanese, etc.
}

# TTS model-specific seconds of overlap between sentences (overrides default OVERLAP_SECONDS if specified)
TTS_OVERLAP_SECONDS = {
    "kokoro": 0.6,
}

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

# ── CapCut TTS (K07VN/capcut-tts-api) ──────────────────────────────────────

# Per-request wall-clock timeout (seconds). The task creation + polling +
# download must finish within this window, otherwise the request is retried
# if retries remain. Defaults to 2 minutes, with a 30 second floor.
_capcut_timeout_raw = float(os.environ.get("LUE_CAPCUT_TTS_TIMEOUT", "120"))
CAPCUT_TTS_TIMEOUT = max(_capcut_timeout_raw, 30.0) if _capcut_timeout_raw > 0 else 120.0

# Maximum retry attempts for transient failures (network errors, task
# failures, rate limits).
_capcut_retries_raw = int(os.environ.get("LUE_CAPCUT_TTS_MAX_RETRIES", "2"))
CAPCUT_TTS_MAX_RETRIES = max(min(_capcut_retries_raw, 10), 0) if _capcut_retries_raw >= 0 else 2

# Base delay in seconds for exponential backoff. Attempt n waits
# base * 2**(n-1) seconds before retrying.
_capcut_retry_delay_raw = float(os.environ.get("LUE_CAPCUT_TTS_RETRY_BASE_DELAY", "2.0"))
CAPCUT_TTS_RETRY_BASE_DELAY = max(_capcut_retry_delay_raw, 0.5) if _capcut_retry_delay_raw > 0 else 2.0

# Speaking rate passed through to CapCut as a string (e.g. "1.0", "1.2").
# Parsed as a float and clamped to the range CapCut accepts (0.5-2.0); an
# unparseable value falls back to 1.0 rather than crashing at import time.
try:
    _capcut_rate_raw = float(os.environ.get("LUE_CAPCUT_TTS_RATE", "1.0"))
except (TypeError, ValueError):
    import logging

    logging.getLogger(__name__).warning(
        "Invalid LUE_CAPCUT_TTS_RATE=%r; falling back to 1.0.",
        os.environ.get("LUE_CAPCUT_TTS_RATE"),
    )
    _capcut_rate_raw = 1.0
CAPCUT_TTS_RATE = str(min(max(_capcut_rate_raw, 0.5), 2.0))

# Optional CapCut resource id override. ``None`` lets the SDK use its default.
CAPCUT_RESOURCE_ID = os.environ.get("LUE_CAPCUT_RESOURCE_ID") or None

# Whether to synthesize a throwaway sentence on warm-up to prime the session.
CAPCUT_TTS_WARMUP = os.environ.get("LUE_CAPCUT_TTS_WARMUP", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

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
