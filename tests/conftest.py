"""
Shared fixtures and configuration for the lue-reader test suite.

Isolation is applied at *import time* (module level), not via autouse
fixtures, because ``lue.config`` runs ``os.makedirs()`` as a side effect of
being imported.  A fixture only runs after collection has already imported
the test modules, which is too late.  Concretely, this module:

  - redirects ``platformdirs.user_cache_dir`` / ``user_data_dir`` into a
    throwaway temp tree, so real user data is never touched, and
  - pins environment variables (short timeouts, zero retries, small delays,
    parallel TTS off, empty provider API keys) so a stray network call fails
    fast instead of hanging or billing an account.
"""

from __future__ import annotations

import atexit
import os
import platformdirs
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest
from rich.console import Console


# ── Import-time isolation (must run before lue.config is imported) ─────────

_TEST_ROOT = tempfile.mkdtemp(prefix="lue-test-")

# Bind the user- and cache-dir helpers to a temp tree *before* lue.config
# computes AUDIO_DATA_DIR / PROGRESS_FILE_DIR at import time.
platformdirs.user_cache_dir = lambda *args, **kwargs: os.path.join(_TEST_ROOT, "cache")
platformdirs.user_data_dir = lambda *args, **kwargs: os.path.join(_TEST_ROOT, "data")

# Low timeouts prevent hangs on network calls; zero retries make transient
# failures surface immediately instead of delaying the suite; small delays
# keep the retry-path tests fast.
os.environ.update(
    {
        "LUE_OPENAI_TTS_TIMEOUT": "1",
        "LUE_OPENAI_TTS_MAX_RETRIES": "0",
        "LUE_OPENAI_TTS_RETRY_BASE_DELAY": "0.1",
        # Deterministic, single-threaded TTS behaviour.
        "LUE_TTS_PARALLEL_ENABLED": "False",
        "LUE_LOOKAHEAD_SENTENCES": "5",
        "LUE_TTS_MAX_CONCURRENT": "1",
        # Suppress UI-related side effects.
        "LUE_SHOW_BUFFER_STATUS": "false",
        # Never let a test pick up a real provider credential.
        "OPENAI_API_KEY": "",
        "NANOGPT_API_KEY": "",
        "SPEECHIFY_API_KEY": "",
    }
)


@atexit.register
def _cleanup_test_root() -> None:
    """Remove the throwaway cache/data tree when the session ends."""
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


# ── pytest configuration ───────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Tests that verify individual functions/units in isolation")
    config.addinivalue_line("markers", "integration: Tests that verify interactions between multiple components")


# ── Rich Console ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_console() -> Mock:
    """
    Return a ``Mock`` that replaces ``rich.console.Console``.

    The mock's ``print`` method is a silent no-op so that tests can call
    ``console.print(...)`` without producing terminal output or requiring
    a real terminal.
    """
    console = Mock(spec=Console)
    console.print = Mock(return_value=None)
    return console


@pytest.fixture
def real_console() -> Generator[Console, None, None]:
    """
    Return a genuine ``rich.console.Console`` connected to ``/dev/null``.

    Use this fixture when the code under test inspects the console object
    itself (e.g. ``isinstance`` checks, attribute access) rather than just
    calling ``.print()``.
    """
    devnull = open(os.devnull, "w")
    try:
        yield Console(file=devnull)
    finally:
        devnull.close()


# ── Temporary output directories ──────────────────────────────────────────

@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """
    Return a temporary directory for test-generated files (audio, cache, etc.).

    The directory is automatically cleaned up after each test.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Return a temporary directory that simulates the TTS cache."""
    cache_dir = tmp_path / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# ── Sample content helpers ─────────────────────────────────────────────────

@pytest.fixture
def sample_text() -> str:
    """A short piece of plain text suitable for TTS or content-parser tests."""
    return (
        "This is a sample sentence for testing. "
        "It contains multiple sentences. "
        "Even a third one for good measure!"
    )


@pytest.fixture
def sample_paragraphs() -> list[str]:
    """A list of paragraphs (plain text) for pipeline or chapter tests."""
    return [
        "This is the first paragraph. It has two sentences here.",
        "This is the second paragraph. It also has two sentences. And a third.",
        "Short last paragraph.",
    ]


@pytest.fixture
def sample_html() -> str:
    """A minimal HTML document for content-parser tests."""
    return """<html><body>
<h1>Chapter One</h1>
<p>This is a paragraph in the first chapter.</p>
<p>This is another paragraph.</p>
</body></html>"""
