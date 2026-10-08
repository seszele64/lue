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
}

# Language codes for TTS models that require them
TTS_LANGUAGE_CODES = {
    "kokoro": "a",  # a=English, e=Spanish, j=Japanese, etc.
}

# TTS model-specific seconds of overlap between sentences.
# DEPRECATED (spec-v3 §9.1): crossfade/overlap was removed in favour of a gapless
# persistent sink. The mapping is kept as an empty dict for config compatibility
# only; its values are ignored and `get_overlap_seconds()` now returns ``None``.
TTS_OVERLAP_SECONDS = {}

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
# Legacy per-file buffers (only used on the ``--legacy-audio`` path).
AUDIO_BUFFERS = [os.path.join(AUDIO_DATA_DIR, f"buffer_{i}") for i in range(6)]
MAX_QUEUE_SIZE = 4  # legacy player queue bound (items)

# ── Persistent sink (spec-v3 §11) ──────────────────────────────────────────
AUDIO_SINK_RATE = 48000
AUDIO_SINK_CHANNELS = 2
AUDIO_SINK_SAMPLE_FMT = "s16le"
BYTES_PER_SEC = AUDIO_SINK_RATE * 2 * AUDIO_SINK_CHANNELS  # 192000
SINK_PRIME_MS = 150  # buffering watermark ONLY; no sleep (§4.5)
SINK_DEVICE_LATENCY_MS = 100
SINK_LEAD_MS = int(os.environ.get("LUE_SINK_LEAD_MS", "250"))
SINK_LEAD = SINK_LEAD_MS / 1000.0
SINK_WRITE_TIMEOUT_S = 5.0
SINK_TERM_TIMEOUT_S = 0.5
SINK_KILL_TIMEOUT_S = 0.2
SINK_EOF_TIMEOUT_S = 3.0
SINK_STARTUP_GRACE_S = 0.1  # health-check delay after ffplay spawn (§3.1)
WRITER_CHUNK_MS = 20  # 3840 B/chunk
PCM_RING_MAX_MS = 2000  # 384000 B
PCM_LOW_WATER_MS = 300
UNDERRUN_PAD_MS = 100
MAX_UNDERRUN_PAD_MS = 2000
MAX_QUEUED_SENTENCES = 8
DECODE_FAIL_SILENCE_MS = 500
DECODE_TERM_TIMEOUT_S = 0.3
TASK_CANCEL_TIMEOUT_S = 1.0

# ── Speed (spec-v3 §6.3) ───────────────────────────────────────────────────
SPEED_MIN = 0.5
SPEED_MAX = 4.0
SPEED_STEP = 0.1
DEFAULT_SPEED = 1.0

# ── Sentences / overlap (P1-1) ─────────────────────────────────────────────
SENTENCE_GAP_SECONDS = 0.0
OVERLAP_SECONDS = 0.0  # DEPRECATED alias; kept for config compat

# ── Temp lifecycle (P1-4) ──────────────────────────────────────────────────
AUDIO_TMP_ROOT = user_cache_dir("lue")
AUDIO_TMP_MAX_AGE_S = 3600

# ── Migration flag (P1-5) ──────────────────────────────────────────────────
USE_PERSISTENT_SINK = os.environ.get("LUE_PERSISTENT_SINK", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# ── Optional conservative silence trimming (P1-4 / §7.3) ───────────────────
TRIM_SILENCE = False

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
