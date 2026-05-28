"""
Shared fixtures and configuration for the lue-reader test suite.

All tests automatically inherit:
  - Isolated environment variables (test_env) to prevent accidental API calls
    and speed up test execution
  - A temporary AUDIO_DATA_DIR to prevent side effects from config.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest
from rich.console import Console


# ── pytest configuration ───────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Tests that verify individual functions/units in isolation")
    config.addinivalue_line("markers", "integration: Tests that verify interactions between multiple components")


# ── Environment ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set environment variables for a controlled test environment.

    * Low timeouts prevent tests from hanging on network calls.
    * Zero retries make transient failures surface immediately as errors
      rather than delaying the test.
    * Small delay values speed up retry-path tests when retries *are* used.
    * Audio/cache paths are redirected to a throwaway temp directory so that
      real user data is never touched.
    """
    monkeypatch.setenv("LUE_OPENAI_TTS_TIMEOUT", "1")
    monkeypatch.setenv("LUE_OPENAI_TTS_MAX_RETRIES", "0")
    monkeypatch.setenv("LUE_OPENAI_TTS_RETRY_BASE_DELAY", "0.1")

    # Disable parallel TTS by default in tests for deterministic behaviour.
    monkeypatch.setenv("LUE_TTS_PARALLEL_ENABLED", "False")

    # Small lookahead / concurrency to keep tests fast.
    monkeypatch.setenv("LUE_LOOKAHEAD_SENTENCES", "5")
    monkeypatch.setenv("LUE_TTS_MAX_CONCURRENT", "1")

    # Suppress UI-related side effects.
    monkeypatch.setenv("LUE_SHOW_BUFFER_STATUS", "false")


@pytest.fixture(autouse=True)
def temp_audio_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Redirect every directory that ``lue.config`` creates on import into an
    empty per-test temporary directory.

    Because ``config.py`` runs ``os.makedirs()`` at module level we must
    patch before the first import.  This fixture (via ``conftest.py``
    autouse) ensures that by the time any test module is collected the
    patches are in place.
    """
    # Bind the user- and cache-dir helpers to a temp tree.
    test_root = tmp_path / "lue_test_data"
    test_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("platformdirs.user_data_dir", lambda appname: str(test_root / "data"))
    monkeypatch.setattr("platformdirs.user_cache_dir", lambda appname: str(test_root / "cache"))


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
