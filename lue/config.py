"""Configuration settings for the Lue eBook reader."""

import logging
import os
from platformdirs import user_data_dir, user_cache_dir

# Default TTS model
DEFAULT_TTS_MODEL = "edge"

# Default voices for TTS models
TTS_VOICES = {
    "edge": "en-US-JennyNeural",
    "kokoro": "af_heart",
    "openai": "alloy",
}

# Language codes for TTS models that require them
TTS_LANGUAGE_CODES = {
    "kokoro": "a",  # a=English, e=Spanish, j=Japanese, etc.
}

# TTS model-specific seconds of overlap between sentences (overrides default OVERLAP_SECONDS if specified)
TTS_OVERLAP_SECONDS = {
    "kokoro": 0.6,
}

# Audio processing settings
AUDIO_DATA_DIR = user_cache_dir("lue")
os.makedirs(AUDIO_DATA_DIR, exist_ok=True)
AUDIO_BUFFERS = [os.path.join(AUDIO_DATA_DIR, f"buffer_{i}") for i in range(6)]
MAX_QUEUE_SIZE = 12

# Pre-buffering settings (controls when playback starts)
# Playback begins once either threshold is reached
PREBUFFER_MIN_ITEMS = 3       # Minimum sentences to pre-generate
PREBUFFER_MIN_SECONDS = 10.0  # Minimum seconds of audio to pre-generate

# TTS cache settings
TTS_CACHE_ENABLED = True  # When False, skip all cache operations
TTS_CACHE_DIR = os.path.join(AUDIO_DATA_DIR, "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)
TTS_CACHE_MAX_SIZE_MB = 100
TTS_CACHE_MAX_SIZE_BYTES = TTS_CACHE_MAX_SIZE_MB * 1024 * 1024

# Lookahead pre-fetch buffer settings
_log = logging.getLogger(__name__)
_la_raw = int(os.environ.get("LUE_LOOKAHEAD_SENTENCES", "30"))
if _la_raw < 5:
    _log.warning("LUE_LOOKAHEAD_SENTENCES=%d clamped to lower bound 5", _la_raw)
    _la_raw = 5
elif _la_raw > 100:
    _log.warning("LUE_LOOKAHEAD_SENTENCES=%d clamped to upper bound 100", _la_raw)
    _la_raw = 100
LOOKAHEAD_SENTENCES = _la_raw
LOOKAHEAD_MAX_ERRORS = int(os.environ.get("LUE_LOOKAHEAD_MAX_ERRORS", "5"))
LOOKAHEAD_TEMP_DIR = os.path.join(AUDIO_DATA_DIR, "lookahead")
os.makedirs(LOOKAHEAD_TEMP_DIR, exist_ok=True)

# Clean up stale lookahead temp files from previous sessions on startup
for stale_file in os.listdir(LOOKAHEAD_TEMP_DIR):
    if stale_file.startswith("la_") and (stale_file.endswith(".mp3") or stale_file.endswith(".wav")):
        try:
            os.remove(os.path.join(LOOKAHEAD_TEMP_DIR, stale_file))
        except OSError:
            pass

# ═══════════════════════════════════════════════════════════════════════════
# TTS Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════
# These settings control the audio generation and playback pipeline.
# See LOOKAHEAD_TEMP_DIR for session temp files.

_tts_parallel_raw = os.environ.get("LUE_TTS_PARALLEL_ENABLED", "True")
TTS_PARALLEL_ENABLED = _tts_parallel_raw.lower() not in ("0", "false", "no")

_tts_max_raw = int(os.environ.get("LUE_TTS_MAX_CONCURRENT", "3"))
if _tts_max_raw < 1:
    _log.warning("LUE_TTS_MAX_CONCURRENT=%d clamped to lower bound 1", _tts_max_raw)
    _tts_max_raw = 1
elif _tts_max_raw > 8:
    _log.warning("LUE_TTS_MAX_CONCURRENT=%d clamped to upper bound 8", _tts_max_raw)
    _tts_max_raw = 8
TTS_MAX_CONCURRENT = _tts_max_raw

TTS_MAX_CONCURRENT_FALLBACK = 1

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
UI_COMPLEXITY_MODE = (
    2  # 0=minimal (text only), 1=medium (top bar only), 2=full (default)
)

# Highlighting settings
SENTENCE_HIGHLIGHTING_ENABLED = True  # Enable sentence-level highlighting
WORD_HIGHLIGHT_MODE = 1  # 0=off, 1=normal highlighting, 2=standout highlighting

# Keyboard settings
# Can be set to "default", "vim", or a path to a custom keyboard shortcuts JSON file
CUSTOM_KEYBOARD_SHORTCUTS = "default"
