"""Configuration settings for the Lue eBook reader."""

import logging
import math
import os
from platformdirs import user_data_dir, user_cache_dir

# Default TTS model
DEFAULT_TTS_MODEL = "edge"

# Default voices for TTS models
TTS_VOICES = {
    "edge": "en-US-JennyNeural",
    "kokoro": "af_heart",
    "openai": "alloy",
    "nanogpt": "Rex",
    "speechify": "wren",
    "gtts": "en",  # free Google Translate TTS: default language code
    # CapCut's built-in streaming voice. Matches the SDK's fallback resource
    # (7102355709945188865), so it works even when Voice.json is unavailable.
    "capcut": "BV074_streaming",
}

# Language codes for TTS models that require them
TTS_LANGUAGE_CODES = {
    "kokoro": "a",  # a=English, e=Spanish, j=Japanese, etc.
    "gtts": "en",  # fallback alias for TTS_VOICES["gtts"]
}

# TTS model-specific seconds of overlap between sentences.
# DEPRECATED / DEAD since spec-v3 §9.1: providers return None from
# get_overlap_seconds(), so these values are ignored.  Overlap is governed by
# OVERLAP_SECONDS below (and, once gapless playback lands, by the gapless
# scheduler).  Kokoro's entry is kept only so the old config shape still loads.
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
# Endpoint top-level domain: "com" is Google's default, regional variants
# ("co.uk", "co.in", "de") change the accent.  Normalised here, validated by
# the provider at startup because a typo would otherwise fail on every
# sentence (see gtts_tts.py).
GTTS_TLD = (os.environ.get("LUE_GTTS_TLD", "com") or "").strip().lower() or "com"
GTTS_SLOW = os.environ.get("LUE_GTTS_SLOW", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Per-request timeout (seconds) for the blocking gTTS network call.  gTTS
# splits text into ~100-character chunks and fetches each one separately, so
# the provider derives its overall budget from this value (one timeout per
# chunk plus a small margin).  The call runs in a worker thread, so a hung
# request is abandoned instead of stalling the audio producer; the pipeline's
# error path then takes over.
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
# Parsed as a float and clamped to the range CapCut accepts (0.5-2.0).  An
# unparseable or non-finite value ("nan", "inf") falls back to 1.0: float()
# accepts those, and they would otherwise poison the clamp below (nan compares
# false against every bound, so it would survive as "nan").
try:
    _capcut_rate_raw = float(os.environ.get("LUE_CAPCUT_TTS_RATE", "1.0"))
except (TypeError, ValueError):
    _capcut_rate_raw = float("nan")

if not math.isfinite(_capcut_rate_raw):
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
MAX_QUEUE_SIZE = 12

# Pre-buffering settings (controls when playback starts)
# Playback begins once either threshold is reached
PREBUFFER_MIN_ITEMS = 4       # Minimum sentences to pre-generate
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

# NanoGPT TTS settings (xai-tts via https://nano-gpt.com/api/tts)
# The route rejects input longer than 5000 characters per request
# (error code TTS_INVALID_TEXT), so longer text is chunked by the provider.
NANOGPT_TTS_MAX_CHARS = 5000

# Synthesis speed multiplier (xai-tts accepts 0.7 - 1.5; default 1.0).
try:
    _ng_speed_raw = float(os.environ.get("LUE_NANOGPT_TTS_SPEED", "1"))
except ValueError:
    _ng_speed_raw = 1.0
NANOGPT_TTS_SPEED = min(max(_ng_speed_raw, 0.7), 1.5)

# Per-request generation timeout (seconds).
_ng_timeout_raw = float(os.environ.get("LUE_NANOGPT_TTS_TIMEOUT", "30"))
NANOGPT_TTS_TIMEOUT = max(_ng_timeout_raw, 5.0) if _ng_timeout_raw > 0 else 30.0

# Maximum retry attempts for transient failures (timeouts, connection
# errors, server errors, rate limits).
_ng_retries_raw = int(os.environ.get("LUE_NANOGPT_TTS_MAX_RETRIES", "3"))
NANOGPT_TTS_MAX_RETRIES = max(min(_ng_retries_raw, 10), 0) if _ng_retries_raw >= 0 else 3

# Base delay in seconds for exponential backoff.
_ng_retry_delay_raw = float(os.environ.get("LUE_NANOGPT_TTS_RETRY_BASE_DELAY", "1.0"))
NANOGPT_TTS_RETRY_BASE_DELAY = max(_ng_retry_delay_raw, 0.1) if _ng_retry_delay_raw > 0 else 1.0

# Async (HTTP 202) job polling for models that do not respond synchronously.
NANOGPT_TTS_POLL_INTERVAL = float(os.environ.get("LUE_NANOGPT_TTS_POLL_INTERVAL", "3"))
NANOGPT_TTS_POLL_MAX_ATTEMPTS = int(os.environ.get("LUE_NANOGPT_TTS_POLL_MAX_ATTEMPTS", "60"))

# Speechify TTS settings (POST https://api.speechify.ai/v1/audio/speech)
# The endpoint rejects input longer than 2000 characters per request
# (HTTP 400 "Field input must not exceed 2000 characters"), so longer text
# is chunked by the provider.
SPEECHIFY_TTS_MAX_CHARS = 2000

# Maximum simultaneous in-flight requests. The API enforces a per-plan
# concurrency limit (1 request on the free plan); raising this above your
# plan's limit produces HTTP 429 concurrency_limit_reached responses.
_spc_raw = int(os.environ.get("LUE_SPEECHIFY_TTS_MAX_CONCURRENT", "1"))
SPEECHIFY_TTS_MAX_CONCURRENT = max(min(_spc_raw, 8), 1) if _spc_raw >= 1 else 1

# Whether to run Speechify's text normalization (expand numbers,
# abbreviations) before synthesis.
_spc_norm_raw = os.environ.get("LUE_SPEECHIFY_TTS_TEXT_NORMALIZATION", "False")
SPEECHIFY_TTS_TEXT_NORMALIZATION = _spc_norm_raw.lower() in ("1", "true", "yes")

# Per-request generation timeout (seconds).
_spc_timeout_raw = float(os.environ.get("LUE_SPEECHIFY_TTS_TIMEOUT", "30"))
SPEECHIFY_TTS_TIMEOUT = max(_spc_timeout_raw, 5.0) if _spc_timeout_raw > 0 else 30.0

# Maximum retry attempts for transient failures (timeouts, connection
# errors, server errors, rate/concurrency limits).
_spc_retries_raw = int(os.environ.get("LUE_SPEECHIFY_TTS_MAX_RETRIES", "3"))
SPEECHIFY_TTS_MAX_RETRIES = max(min(_spc_retries_raw, 10), 0) if _spc_retries_raw >= 0 else 3

# Base delay in seconds for exponential backoff. Retry-After is honoured
# when the server sends one and is longer than this schedule.
_spc_delay_raw = float(os.environ.get("LUE_SPEECHIFY_TTS_RETRY_BASE_DELAY", "1.0"))
SPEECHIFY_TTS_RETRY_BASE_DELAY = max(_spc_delay_raw, 0.1) if _spc_delay_raw > 0 else 1.0

# TTS pipeline UI settings
SHOW_BUFFER_STATUS = os.environ.get("LUE_SHOW_BUFFER_STATUS", "").lower() in ("1", "true", "yes")

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
