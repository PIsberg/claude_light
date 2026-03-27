"""
Test utilities and mode detection for claude_light.

This module provides:
- Test mode detection and management
- Mock API key generation
- Common test utilities and fixtures
- Test-related constants and helpers

This module should ONLY be imported when needed for testing, or by config.py
for mode detection. It should not be used in production application code paths.
"""

import sys


# ============================================================================
# Test Mode Detection
# ============================================================================

def is_test_mode_enabled() -> bool:
    """
    Check if the application is running in test mode.
    
    Returns:
        bool: True if --test-mode flag was provided on command line.
    """
    return "--test-mode" in sys.argv


def get_test_api_key() -> str:
    """
    Get a mock API key for testing.
    
    Returns:
        str: A dummy Anthropic API key for test mode.
    """
    return "sk-ant-test-mock-key"


# ============================================================================
# UI Constants for Test Mode
# ============================================================================

# ANSI color codes (duplicated from ui.py for independence)
_ANSI_BLUE    = "\033[34m"
_ANSI_RESET   = "\033[0m"

# Test mode UI indicator
TEST_MODE_TAG = f"{_ANSI_BLUE}[Test Mode]{_ANSI_RESET}"
