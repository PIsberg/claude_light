"""
Streaming response handler for Claude Light.

Provides real-time token streaming from the Anthropic API with incremental
display of Claude's response and proper handling of edit blocks.
"""

import sys
import anthropic
from typing import Iterator, Tuple, Dict, Any
from claude_light.config import PRICE_INPUT, PRICE_WRITE, PRICE_READ, PRICE_OUTPUT
from claude_light.ui import _ANSI_RESET, _ANSI_CYAN, _ANSI_DIM
import claude_light.state as state


class StreamingResponseHandler:
    """Manages streaming responses from the Claude API."""

    def __init__(self):
        self.buffer = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.thinking_buffer = ""
        self.is_thinking = False

    def process_stream(self, stream: Iterator) -> Tuple[str, Dict[str, int]]:
        """
        Process a streaming response from the API.

        Args:
            stream: Iterator of delta events from client.messages.stream()

        Returns:
            Tuple of (full_response_text, usage_dict)
            where usage_dict contains token counts for later processing
        """
        self._print_stream_start()

        for event in stream:
            if hasattr(event, 'type'):
                if event.type == 'content_block_start':
                    if hasattr(event, 'content_block'):
                        if event.content_block.type == 'thinking':
                            self.is_thinking = True
                            self._print_thinking_start()

                elif event.type == 'content_block_delta':
                    if hasattr(event, 'delta'):
                        if hasattr(event.delta, 'thinking'):
                            self.thinking_buffer += event.delta.thinking
                            self._print_thinking_chunk(event.delta.thinking)
                        elif hasattr(event.delta, 'text'):
                            self.buffer += event.delta.text
                            self._print_text_chunk(event.delta.text)

                elif event.type == 'content_block_stop':
                    if self.is_thinking:
                        self._print_thinking_stop()
                        self.is_thinking = False

                elif event.type == 'message_delta':
                    if hasattr(event, 'usage'):
                        self.output_tokens = event.usage.output_tokens
                    if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                        pass  # End of message

                elif event.type == 'message_start':
                    if hasattr(event, 'message'):
                        msg = event.message
                        if hasattr(msg, 'usage'):
                            self.input_tokens = msg.usage.input_tokens
                            self.cache_creation_tokens = getattr(msg.usage, 'cache_creation_input_tokens', 0)
                            self.cache_read_tokens = getattr(msg.usage, 'cache_read_input_tokens', 0)

        self._print_stream_end()
        return self.buffer, self._get_usage_dict()

    def _get_usage_dict(self) -> Dict[str, int]:
        """Get usage metrics as a dict."""
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cache_creation_tokens': self.cache_creation_tokens,
            'cache_read_tokens': self.cache_read_tokens,
        }

    def _print_stream_start(self):
        """Print visual indicator that streaming is starting."""
        print()  # Blank line before response

    def _print_stream_end(self):
        """Print visual indicator that streaming is complete."""
        print()  # Blank line after response

    def _print_thinking_start(self):
        """Print thinking mode indicator."""
        print(f"{_ANSI_DIM}[Thinking...]{_ANSI_RESET}", end="", flush=True)

    def _print_thinking_chunk(self, chunk: str):
        """Print a chunk of thinking text (usually hidden from user, just for feedback)."""
        # Thinking is typically not shown to the user, just indicate it's happening
        pass

    def _print_thinking_stop(self):
        """Print thinking completion indicator."""
        print(f"\n", end="", flush=True)

    def _print_text_chunk(self, chunk: str):
        """Print a chunk of response text in real-time."""
        print(chunk, end="", flush=True)


def stream_chat_response(client, **create_kwargs) -> Tuple[str, Dict[str, int]]:
    """
    Execute a streaming chat request and return full response with usage.

    Args:
        client: Anthropic client instance
        **create_kwargs: Arguments to pass to client.messages.stream()

    Returns:
        Tuple of (response_text, usage_dict)

    Usage:
        response_text, usage = stream_chat_response(
            client,
            model="claude-opus-4-1",
            max_tokens=4096,
            system=system_blocks,
            messages=messages
        )
    """
    handler = StreamingResponseHandler()

    with client.messages.stream(**create_kwargs) as stream:
        response_text, usage = handler.process_stream(stream)

    return response_text, usage


def calculate_usage_cost(usage_dict: Dict[str, int]) -> float:
    """Calculate cost from usage metrics."""
    inp = usage_dict.get('input_tokens', 0)
    cw = usage_dict.get('cache_creation_tokens', 0)
    cr = usage_dict.get('cache_read_tokens', 0)
    out = usage_dict.get('output_tokens', 0)

    cost_inp = (inp / 1_000_000) * PRICE_INPUT
    cost_cw = (cw / 1_000_000) * PRICE_WRITE
    cost_cr = (cr / 1_000_000) * PRICE_READ
    cost_out = (out / 1_000_000) * PRICE_OUTPUT

    return cost_inp + cost_cw + cost_cr + cost_out


def accumulate_usage_from_dict(usage_dict: Dict[str, int]):
    """Accumulate usage from a streaming usage dict."""
    with state.lock:
        state.session_tokens["input"] += usage_dict.get('input_tokens', 0)
        state.session_tokens["cache_write"] += usage_dict.get('cache_creation_tokens', 0)
        state.session_tokens["cache_read"] += usage_dict.get('cache_read_tokens', 0)
        state.session_tokens["output"] += usage_dict.get('output_tokens', 0)
