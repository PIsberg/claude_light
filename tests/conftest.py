"""
Pytest configuration and shared fixtures for claude_light tests.

This file:
- Configures pytest plugins and test discovery
- Provides shared fixtures used across test suites
- Sets up common test infrastructure (mocks, paths, environment)
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path so tests can import claude_light
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set dummy API key to prevent import errors in test mode
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-mock-key")

# Most tests were written for the single dense model; tests of the hybrid
# index (#61) switch it on themselves with patch.object(config, ...).
os.environ.setdefault("CLAUDE_LIGHT_HYBRID", "0")

# Keep the suite's fake token usage out of the real ~/.claude_light_stats.json.
# Set before any claude_light import: config reads it at import time.
import tempfile as _tempfile
os.environ["CLAUDE_LIGHT_STATS_FILE"] = os.path.join(
    _tempfile.mkdtemp(prefix="claude_light_stats_"), "stats.json")
# Same for the conversations saved for --continue (~/.claude_light_sessions).
os.environ["CLAUDE_LIGHT_SESSION_DIR"] = _tempfile.mkdtemp(prefix="claude_light_sessions_")

import threading

import pytest


@pytest.fixture(autouse=True)
def _fresh_session_model():
    """ROUTE_ONCE is on by default: a model pinned by one test's first route
    must not carry into the next test."""
    import claude_light.state as state
    state.session_model = None
    yield
    state.session_model = None


@pytest.fixture(autouse=True)
def _no_real_llmlingua_loader():
    """A test that runs main() unpatched starts the real LLMLingua loader. It
    keeps loading after the test ends and sets the compressor singleton under
    a later test's feet (#74). Patch _start_llmlingua_load in such tests."""
    import claude_light.compressor as compressor
    before = compressor._load_thread
    yield
    thread = compressor._load_thread
    if thread is not before and isinstance(thread, threading.Thread):
        thread.join(timeout=300)  # leave nothing loading for the next test
        compressor._load_thread = None
        compressor._load_done.clear()
        compressor._load_attempted = False
        compressor._compressor = None
        compressor._load_error = None
        pytest.fail("test started a real LLMLingua loader thread; patch "
                    "claude_light.main._start_llmlingua_load", pytrace=False)


@pytest.fixture(scope="session")
def project_root_fixture():
    """Provide the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def tests_dir():
    """Provide the tests directory."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def fixtures_dir():
    """Provide the fixtures directory."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture(scope="session")
def claude_light_module():
    """Import and provide the claude_light module."""
    import claude_light
    return claude_light
