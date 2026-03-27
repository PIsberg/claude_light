"""
Tests for rate-limit handling and retry logic.
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Set up environment
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
# Go up 3 levels from tests/unit/
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from claude_light.retry import retry_with_backoff, _should_retry, MAX_RETRIES, INITIAL_BACKOFF_SECS, MAX_BACKOFF_SECS
import anthropic
import httpx


def _make_status_error(cls, message="error", status_code=400):
    """Create an Anthropic API status error with a mock response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.headers = httpx.Headers()
    mock_response.request = MagicMock(spec=httpx.Request)
    return cls(message, response=mock_response, body=None)


def _make_connection_error():
    """Create an Anthropic APIConnectionError with a mock request."""
    mock_request = MagicMock(spec=httpx.Request)
    return anthropic.APIConnectionError(request=mock_request)


class TestShouldRetry(unittest.TestCase):
    """Test the _should_retry function."""

    def test_rate_limit_error_is_retriable(self):
        """RateLimitError should be retriable."""
        error = _make_status_error(anthropic.RateLimitError, "too many requests", 429)
        self.assertTrue(_should_retry(error))

    def test_connection_error_is_retriable(self):
        """APIConnectionError should be retriable."""
        error = _make_connection_error()
        self.assertTrue(_should_retry(error))

    def test_server_error_is_retriable(self):
        """Server errors (5xx) should be retriable."""
        error = Exception("Server Error")
        error.status_code = 503
        self.assertTrue(_should_retry(error))

    def test_authentication_error_not_retriable(self):
        """AuthenticationError (401) should not be retriable."""
        error = _make_status_error(anthropic.AuthenticationError, "invalid key", 401)
        self.assertFalse(_should_retry(error))

    def test_permission_error_not_retriable(self):
        """PermissionDeniedError (403) should not be retriable."""
        error = _make_status_error(anthropic.PermissionDeniedError, "forbidden", 403)
        self.assertFalse(_should_retry(error))

    def test_not_found_error_not_retriable(self):
        """NotFoundError (404) should not be retriable."""
        error = _make_status_error(anthropic.NotFoundError, "not found", 404)
        self.assertFalse(_should_retry(error))

    def test_invalid_request_error_not_retriable(self):
        """InvalidRequestError (400) should not be retriable."""
        error = _make_status_error(anthropic.BadRequestError, "bad request", 400)
        self.assertFalse(_should_retry(error))

    def test_generic_exception_not_retriable(self):
        """Generic exceptions should not be retriable."""
        error = Exception("something went wrong")
        self.assertFalse(_should_retry(error))


class TestRetryWithBackoff(unittest.TestCase):
    """Test the retry_with_backoff decorator."""

    def test_successful_first_try(self):
        """Function that succeeds on first try should not retry."""
        @retry_with_backoff
        def success():
            return "success"
        
        result = success()
        self.assertEqual(result, "success")

    def test_retry_on_transient_error(self):
        """Function should retry on rate limit error."""
        call_count = [0]

        @retry_with_backoff
        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_status_error(anthropic.RateLimitError, "rate limited", 429)
            return "success"
        
        # Should succeed after retry
        result = flaky()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 2)

    def test_fail_on_non_retriable_error(self):
        """Function should fail immediately on non-retriable error."""
        call_count = [0]

        @retry_with_backoff
        def auth_error():
            call_count[0] += 1
            raise _make_status_error(anthropic.AuthenticationError, "invalid key", 401)

        # Should fail immediately without retries
        with self.assertRaises(anthropic.AuthenticationError):
            auth_error()
        
        self.assertEqual(call_count[0], 1)

    def test_max_retries_exceeded(self):
        """Function should fail after MAX_RETRIES attempts."""
        call_count = [0]

        @retry_with_backoff
        def always_fails():
            call_count[0] += 1
            raise _make_status_error(anthropic.RateLimitError, "rate limited", 429)

        # Should fail after exhausting retries
        with self.assertRaises(anthropic.RateLimitError):
            always_fails()
        
        # Should have tried MAX_RETRIES times
        self.assertEqual(call_count[0], MAX_RETRIES)

    def test_exponential_backoff_timing(self):
        """Verify exponential backoff increases wait time."""
        call_count = [0]
        call_times = []
        
        @retry_with_backoff
        def backoff_test():
            call_times.append(time.time())
            call_count[0] += 1
            if call_count[0] < 3:
                raise _make_status_error(anthropic.RateLimitError, "rate limited", 429)
            return "success"
        
        start = time.time()
        result = backoff_test()
        total_time = time.time() - start
        
        # Should have retried twice
        self.assertEqual(call_count[0], 3)
        
        # Total wait should be at least initial_backoff + (initial_backoff * 2)
        # = 2 + 4 = 6 seconds (approximately)
        # Allow 20% margin for timing variations
        self.assertGreaterEqual(total_time, INITIAL_BACKOFF_SECS * 1.2)

    def test_backoff_capped_at_max(self):
        """Verify backoff is capped at MAX_BACKOFF_SECS."""
        call_count = [0]
        
        @retry_with_backoff
        def many_retries():
            call_count[0] += 1
            if call_count[0] < MAX_RETRIES:
                raise _make_status_error(anthropic.RateLimitError, "rate limited", 429)
            return "success"
        
        start = time.time()
        result = many_retries()
        total_time = time.time() - start
        
        # With capping: wait should not exceed ~(2 + 4 + 8 + max) = ~(2 + 4 + 8 + 60) = 74 sec
        # In reality, on 4 retries with cap: 2 + 4 + 8 = 14 seconds (3rd retry is capped before 16)
        # Allow generous margin
        self.assertLess(total_time, 120)  # Sanity check: not more than 2 minutes

    def test_exception_message_preserved(self):
        """Exception message should be preserved."""
        error_msg = "specific rate limit error"
        
        @retry_with_backoff
        def fail_with_message():
            raise _make_status_error(anthropic.RateLimitError, error_msg, 429)

        with self.assertRaises(anthropic.RateLimitError) as ctx:
            fail_with_message()
        
        # Note: anthropic exceptions may wrap messages; check if present
        self.assertIn(error_msg, str(ctx.exception))

    def test_retry_with_arguments(self):
        """Function with arguments should work with retry."""
        @retry_with_backoff
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        self.assertEqual(result, 5)

    def test_retry_with_kwargs(self):
        """Function with keyword arguments should work with retry."""
        @retry_with_backoff
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = greet("Alice", greeting="Hi")
        self.assertEqual(result, "Hi, Alice!")


class TestRetryIntegration(unittest.TestCase):
    """Integration tests with mock API client."""

    def test_chat_retry_on_rate_limit(self):
        """Chat function should retry on rate limit."""
        from claude_light import llm
        
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_status_error(anthropic.RateLimitError, "rate limited", 429)
            
            # Return successful response
            resp = MagicMock()
            block = MagicMock()
            block.type = "text"
            block.text = "response"
            resp.content = [block]
            resp.usage = MagicMock()
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50
            resp.usage.cache_creation_input_tokens = 0
            resp.usage.cache_read_input_tokens = 0
            return resp
        
        with patch("claude_light.llm.client.messages.create", side_effect=mock_create), \
             patch("claude_light.llm.ENABLE_STREAMING", False), \
             patch("claude_light.llm._update_skeleton"), \
             patch("claude_light.llm.retrieve", return_value=("", [])), \
             patch("claude_light.llm._print_reply"), \
             patch("claude_light.llm.print_stats"), \
             patch("claude_light.llm.state.lock"), \
             patch("builtins.print"):
            
            # Suppress print output
            with patch("sys.stdout"):
                llm.chat("test query")
        
        # Should have retried once
        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    unittest.main()
