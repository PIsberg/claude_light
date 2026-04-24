"""
Streaming response handler for Claude Light.

Provides real-time token streaming from the Anthropic API with incremental
display of Claude's response and proper handling of edit blocks.
"""

import sys
import time
import anthropic
from typing import Iterator, Tuple, Dict, Any
from claude_light.config import PRICE_INPUT, PRICE_WRITE, PRICE_READ, PRICE_OUTPUT
from claude_light.ui import _ANSI_RESET, _ANSI_CYAN, _ANSI_DIM, _ANSI_GREEN, _ANSI_BOLD, _SYM_RESP, _UNICODE
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
        self._thinking_chars = 0
        self._write_buf = ""
        self._last_flush = 0.0

    def process_stream(self, stream: Iterator) -> Tuple[str, Dict[str, int]]:
        """
        Process a streaming response from the API.

        Returns:
            Tuple of (full_response_text, usage_dict)
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
                    delta = getattr(event, 'delta', None)
                    if not delta:
                        continue

                    if self.is_thinking:
                        thinking = getattr(delta, 'thinking', None)
                        if thinking:
                            self.thinking_buffer += thinking
                            self._print_thinking_chunk(thinking)
                    else:
                        text = getattr(delta, 'text', None)
                        if text:
                            self.buffer += text
                            self._print_text_chunk(text)

                elif event.type == 'content_block_stop':
                    if self.is_thinking:
                        self._print_thinking_stop()
                        self.is_thinking = False

                elif event.type == 'message_delta':
                    if hasattr(event, 'usage'):
                        self.output_tokens = event.usage.output_tokens
                    if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                        pass

                elif event.type == 'message_start':
                    if hasattr(event, 'message'):
                        self.input_tokens = getattr(event.message.usage, 'input_tokens', 0)
                        self.cache_creation_tokens = getattr(event.message.usage, 'cache_creation_input_tokens', 0)
                        self.cache_read_tokens = getattr(event.message.usage, 'cache_read_input_tokens', 0)

        self._print_stream_end()
        return self.buffer, self._get_usage_dict()

    def _get_usage_dict(self) -> Dict[str, int]:
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cache_creation_tokens': self.cache_creation_tokens,
            'cache_read_tokens': self.cache_read_tokens,
        }

    def _print_stream_start(self):
        """Print the ◆ Claude response header before streaming begins."""
        print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)

    def _print_stream_end(self):
        """Flush any buffered text, then print a blank line after the response."""
        if self._write_buf:
            sys.stdout.write(self._write_buf)
            self._write_buf = ""
        sys.stdout.flush()
        print()

    def _print_thinking_start(self):
        """Show a compact thinking indicator."""
        ellipsis = "…" if _UNICODE else "..."
        print(f"\r\033[K  {_ANSI_DIM}*  Thinking{ellipsis}{_ANSI_RESET}", end="", flush=True)
        self._thinking_chars = 0

    def _print_thinking_chunk(self, chunk: str):
        """Advance the thinking indicator (dot every ~50 chars)."""
        self._thinking_chars += len(chunk)
        if self._thinking_chars >= 50:
            print(".", end="", flush=True)
            self._thinking_chars = 0

    def _print_thinking_stop(self):
        """Clear the thinking line before the response."""
        print(f"\r\033[K\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)

    def _print_text_chunk(self, chunk: str):
        """Buffer response text and flush in batches to reduce console I/O."""
        self._write_buf += chunk
        now = time.monotonic()
        if "\n" in chunk or len(self._write_buf) >= 40 or (now - self._last_flush) >= 0.05:
            sys.stdout.write(self._write_buf)
            sys.stdout.flush()
            self._write_buf = ""
            self._last_flush = now


def stream_chat_response(client, **create_kwargs) -> Tuple[str, Dict[str, int]]:
    """
    Execute a streaming chat request and return full response with usage.

    Returns:
        Tuple of (response_text, usage_dict)
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
