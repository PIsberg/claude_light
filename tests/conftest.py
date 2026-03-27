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

import pytest


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
