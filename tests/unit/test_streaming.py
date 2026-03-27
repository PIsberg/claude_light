"""
Tests for streaming response handling.

Validates that streaming correctly captures tokens, handles thinking blocks,
and maintains consistency with non-streaming responses.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Iterator

# Add parent to path - go up 3 levels from tests/unit/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from claude_light.streaming import (
    StreamingResponseHandler, stream_chat_response,
    calculate_usage_cost, accumulate_usage_from_dict
)
import claude_light.state as state


class MockStreamEvent:
    """Mock streaming event from Anthropic API."""
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestStreamingResponseHandler:
    """Test StreamingResponseHandler class."""

    def setup_method(self):
        """Reset state before each test."""
        state.session_cost = 0.0
        state.session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}

    def test_handler_initialization(self):
        """Test that handler initializes correctly."""
        handler = StreamingResponseHandler()
        assert handler.buffer == ""
        assert handler.input_tokens == 0
        assert handler.output_tokens == 0
        assert handler.cache_creation_tokens == 0
        assert handler.cache_read_tokens == 0

    def test_get_usage_dict(self):
        """Test usage dict generation."""
        handler = StreamingResponseHandler()
        handler.input_tokens = 100
        handler.output_tokens = 50
        handler.cache_creation_tokens = 10
        handler.cache_read_tokens = 20

        usage = handler._get_usage_dict()
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["cache_creation_tokens"] == 10
        assert usage["cache_read_tokens"] == 20

    def test_process_stream_basic(self, capsys):
        """Test basic stream processing."""
        handler = StreamingResponseHandler()

        # Create mock stream events
        events = [
            MockStreamEvent('content_block_start', content_block=Mock(type='text')),
            MockStreamEvent('content_block_delta', delta=Mock(text="Hello ")),
            MockStreamEvent('content_block_delta', delta=Mock(text="world")),
            MockStreamEvent('message_delta', usage=Mock(output_tokens=2)),
            MockStreamEvent('content_block_stop'),
            MockStreamEvent('message_start', message=Mock(usage=Mock(
                input_tokens=100,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0
            ))),
        ]

        response_text, usage = handler.process_stream(iter(events))

        assert response_text == "Hello world"
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 2

    def test_process_stream_with_thinking(self, capsys):
        """Test stream processing with thinking blocks."""
        handler = StreamingResponseHandler()

        events = [
            MockStreamEvent('content_block_start', content_block=Mock(type='thinking')),
            MockStreamEvent('content_block_delta', delta=Mock(thinking="Let me think...")),
            MockStreamEvent('content_block_stop'),
            MockStreamEvent('content_block_start', content_block=Mock(type='text')),
            MockStreamEvent('content_block_delta', delta=Mock(text="The answer is 42")),
            MockStreamEvent('message_delta', usage=Mock(output_tokens=5)),
            MockStreamEvent('content_block_stop'),
        ]

        response_text, usage = handler.process_stream(iter(events))

        assert response_text == "The answer is 42"
        assert "Let me think..." not in response_text
        assert usage["output_tokens"] == 5

    def test_process_stream_with_cache_metrics(self, capsys):
        """Test that cache metrics are captured correctly."""
        handler = StreamingResponseHandler()

        events = [
            MockStreamEvent('message_start', message=Mock(usage=Mock(
                input_tokens=100,
                cache_creation_input_tokens=50,
                cache_read_input_tokens=25
            ))),
            MockStreamEvent('content_block_delta', delta=Mock(text="Response")),
            MockStreamEvent('message_delta', usage=Mock(output_tokens=10)),
        ]

        response_text, usage = handler.process_stream(iter(events))

        assert usage["cache_creation_tokens"] == 50
        assert usage["cache_read_tokens"] == 25

    def test_print_methods_no_exception(self, capsys):
        """Test that print methods don't raise exceptions."""
        handler = StreamingResponseHandler()

        # These should not raise
        handler._print_stream_start()
        handler._print_thinking_start()
        handler._print_thinking_chunk("thought")
        handler._print_thinking_stop()
        handler._print_text_chunk("text")
        handler._print_stream_end()

        captured = capsys.readouterr()
        assert "text" in captured.out


class TestCalculateUsageCost:
    """Test cost calculation from streaming usage."""

    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        usage = {
            "input_tokens": 1_000_000,  # $3.00
            "output_tokens": 1_000_000,  # $15.00
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
        cost = calculate_usage_cost(usage)
        assert abs(cost - 18.0) < 0.01  # $3 + $15

    def test_calculate_cost_with_cache(self):
        """Test cost calculation with cache hits."""
        usage = {
            "input_tokens": 1_000_000,  # $3.00
            "output_tokens": 1_000_000,  # $15.00
            "cache_creation_tokens": 1_000_000,  # $3.75
            "cache_read_tokens": 1_000_000,  # $0.30
        }
        cost = calculate_usage_cost(usage)
        # $3.00 + $15.00 + $3.75 + $0.30 = $22.05
        assert abs(cost - 22.05) < 0.01

    def test_calculate_cost_empty_usage(self):
        """Test that empty usage dict returns zero cost."""
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
        cost = calculate_usage_cost(usage)
        assert cost == 0.0

    def test_calculate_cost_with_missing_keys(self):
        """Test cost calculation with missing keys (using defaults)."""
        usage = {
            "input_tokens": 1_000_000,
        }
        # Should not raise; missing keys default to 0
        cost = calculate_usage_cost(usage)
        assert abs(cost - 3.00) < 0.01  # Just the input cost


class TestAccumulateUsageFromDict:
    """Test accumulating usage from streaming dict."""

    def setup_method(self):
        """Reset state before each test."""
        state.session_cost = 0.0
        state.session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}

    def test_accumulate_basic_usage(self):
        """Test accumulating basic usage."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
        accumulate_usage_from_dict(usage)

        assert state.session_tokens["input"] == 100
        assert state.session_tokens["output"] == 50

    def test_accumulate_with_cache(self):
        """Test accumulating with cache metrics."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_tokens": 25,
            "cache_read_tokens": 30,
        }
        accumulate_usage_from_dict(usage)

        assert state.session_tokens["input"] == 100
        assert state.session_tokens["cache_write"] == 25
        assert state.session_tokens["cache_read"] == 30
        assert state.session_tokens["output"] == 50

    def test_accumulate_multiple_calls(self):
        """Test that multiple accumulate calls add up."""
        for _ in range(3):
            usage = {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            }
            accumulate_usage_from_dict(usage)

        assert state.session_tokens["input"] == 300
        assert state.session_tokens["output"] == 150

    def test_accumulate_thread_safety(self):
        """Test that accumulation is thread-safe."""
        import threading

        def accumulate():
            for _ in range(10):
                usage = {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                }
                accumulate_usage_from_dict(usage)

        threads = [threading.Thread(target=accumulate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state.session_tokens["input"] == 50  # 5 threads * 10 updates
        assert state.session_tokens["output"] == 50


class TestStreamingIntegration:
    """Integration tests for streaming feature."""

    def test_stream_chat_response_mock(self):
        """Test stream_chat_response with mocked client."""
        # Create mock client
        mock_client = MagicMock()

        # Create mock stream
        events = [
            MockStreamEvent('message_start', message=Mock(usage=Mock(
                input_tokens=100,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0
            ))),
            MockStreamEvent('content_block_delta', delta=Mock(text="Hello world")),
            MockStreamEvent('message_delta', usage=Mock(output_tokens=2)),
        ]

        # Mock the context manager
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = iter(events)
        mock_stream.__exit__.return_value = False

        mock_client.messages.stream.return_value = mock_stream

        # Call the function
        response_text, usage = stream_chat_response(
            mock_client,
            model="claude-opus-4-1",
            max_tokens=4096,
            system=[{"type": "text", "text": "system"}],
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert response_text == "Hello world"
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 2


class TestStreamingErrorHandling:
    """Test error handling in streaming."""

    def test_handler_graceful_degradation(self):
        """Test that handler degrades gracefully with missing attributes."""
        handler = StreamingResponseHandler()

        # Event with partial data
        events = [
            MockStreamEvent('content_block_start'),  # No content_block
            MockStreamEvent('content_block_delta', delta=Mock(text="Text")),
            MockStreamEvent('message_delta'),  # No usage
        ]

        # Should not raise
        response_text, usage = handler.process_stream(iter(events))
        assert response_text == "Text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
