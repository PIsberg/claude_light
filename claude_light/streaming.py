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

try:
    from rich.live import Live as _RichLive
    from rich.markdown import Markdown as _RichMarkdown
    from rich.text import Text as _RichText
    _RICH_LIVE_AVAILABLE = True
except ImportError:
    _RICH_LIVE_AVAILABLE = False


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
        """Process a streaming response; uses rich.Live when available."""
        from claude_light.ui import console
        if _RICH_LIVE_AVAILABLE and console is not None:
            return self._process_with_live(stream, console)
        return self._process_plain(stream)

    # ------------------------------------------------------------------
    # Rich live-rendering path
    # ------------------------------------------------------------------

    def _process_with_live(self, stream, console) -> Tuple[str, Dict[str, int]]:
        """Stream into rich.Live for real-time Markdown rendering."""
        # Print the ◆ header above the live area then let Live own the content.
        console.print()
        console.print(
            _RichText(f"{_SYM_RESP} ", style="bold cyan"),
            end="",
            highlight=False,
        )

        with _RichLive(
            "",
            console=console,
            refresh_per_second=12,
            vertical_overflow="visible",
        ) as live:
            for event in self._iter_events(stream):
                pass  # events handled inside _iter_events via side-effects
                if self.is_thinking:
                    dots = "." * min(self._thinking_chars // 50, 20)
                    ellipsis = "…" if _UNICODE else "..."
                    live.update(
                        _RichText(f"Thinking{ellipsis}{dots}", style="dim")
                    )
                elif self.buffer:
                    live.update(_RichMarkdown(self.buffer))

        console.print()
        return self.buffer, self._get_usage_dict()

    def _iter_events(self, stream):
        """Iterate stream events, updating self.buffer / self.is_thinking."""
        for event in stream:
            if not hasattr(event, "type"):
                yield event
                continue

            if event.type == "content_block_start":
                if hasattr(event, "content_block"):
                    if event.content_block.type == "thinking":
                        self.is_thinking = True
                        self._thinking_chars = 0

            elif event.type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if not delta:
                    yield event
                    continue
                if self.is_thinking:
                    thinking = getattr(delta, "thinking", None)
                    if thinking:
                        self.thinking_buffer += thinking
                        self._thinking_chars += len(thinking)
                else:
                    text = getattr(delta, "text", None)
                    if text:
                        self.buffer += text

            elif event.type == "content_block_stop":
                if self.is_thinking:
                    self.is_thinking = False

            elif event.type == "message_delta":
                if hasattr(event, "usage"):
                    self.output_tokens = event.usage.output_tokens

            elif event.type == "message_start":
                if hasattr(event, "message"):
                    self.input_tokens = getattr(event.message.usage, "input_tokens", 0)
                    self.cache_creation_tokens = getattr(
                        event.message.usage, "cache_creation_input_tokens", 0
                    )
                    self.cache_read_tokens = getattr(
                        event.message.usage, "cache_read_input_tokens", 0
                    )

            yield event

    # ------------------------------------------------------------------
    # Plain fallback path (no rich.Live)
    # ------------------------------------------------------------------

    def _process_plain(self, stream) -> Tuple[str, Dict[str, int]]:
        self._print_stream_start()
        for event in stream:
            if not hasattr(event, "type"):
                continue
            if event.type == "content_block_start":
                if hasattr(event, "content_block"):
                    if event.content_block.type == "thinking":
                        self.is_thinking = True
                        self._print_thinking_start()
            elif event.type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if not delta:
                    continue
                if self.is_thinking:
                    thinking = getattr(delta, "thinking", None)
                    if thinking:
                        self.thinking_buffer += thinking
                        self._print_thinking_chunk(thinking)
                else:
                    text = getattr(delta, "text", None)
                    if text:
                        self.buffer += text
                        self._print_text_chunk(text)
            elif event.type == "content_block_stop":
                if self.is_thinking:
                    self._print_thinking_stop()
                    self.is_thinking = False
            elif event.type == "message_delta":
                if hasattr(event, "usage"):
                    self.output_tokens = event.usage.output_tokens
            elif event.type == "message_start":
                if hasattr(event, "message"):
                    self.input_tokens = getattr(event.message.usage, "input_tokens", 0)
                    self.cache_creation_tokens = getattr(
                        event.message.usage, "cache_creation_input_tokens", 0
                    )
                    self.cache_read_tokens = getattr(
                        event.message.usage, "cache_read_input_tokens", 0
                    )
        self._print_stream_end()
        return self.buffer, self._get_usage_dict()

    def _get_usage_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }

    def _print_stream_start(self):
        print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)

    def _print_stream_end(self):
        if self._write_buf:
            sys.stdout.write(self._write_buf)
            self._write_buf = ""
        sys.stdout.flush()
        print()

    def _print_thinking_start(self):
        ellipsis = "…" if _UNICODE else "..."
        print(f"\r\033[K  {_ANSI_DIM}*  Thinking{ellipsis}{_ANSI_RESET}", end="", flush=True)
        self._thinking_chars = 0

    def _print_thinking_chunk(self, chunk: str):
        self._thinking_chars += len(chunk)
        if self._thinking_chars >= 50:
            print(".", end="", flush=True)
            self._thinking_chars = 0

    def _print_thinking_stop(self):
        print(f"\r\033[K\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)

    def _print_text_chunk(self, chunk: str):
        self._write_buf += chunk
        now = time.monotonic()
        if "\n" in chunk or len(self._write_buf) >= 40 or (now - self._last_flush) >= 0.05:
            sys.stdout.write(self._write_buf)
            sys.stdout.flush()
            self._write_buf = ""
            self._last_flush = now


def stream_chat_response(client, **create_kwargs) -> Tuple[str, Dict[str, int]]:
    """Execute a streaming chat request and return full response with usage."""
    handler = StreamingResponseHandler()
    with client.messages.stream(**create_kwargs) as stream:
        response_text, usage = handler.process_stream(stream)
    return response_text, usage


def calculate_usage_cost(usage_dict: Dict[str, int]) -> float:
    """Calculate cost from usage metrics."""
    inp = usage_dict.get("input_tokens", 0)
    cw = usage_dict.get("cache_creation_tokens", 0)
    cr = usage_dict.get("cache_read_tokens", 0)
    out = usage_dict.get("output_tokens", 0)
    return (
        (inp / 1_000_000) * PRICE_INPUT
        + (cw / 1_000_000) * PRICE_WRITE
        + (cr / 1_000_000) * PRICE_READ
        + (out / 1_000_000) * PRICE_OUTPUT
    )


def accumulate_usage_from_dict(usage_dict: Dict[str, int]):
    """Accumulate usage from a streaming usage dict."""
    with state.lock:
        state.session_tokens["input"] += usage_dict.get("input_tokens", 0)
        state.session_tokens["cache_write"] += usage_dict.get("cache_creation_tokens", 0)
        state.session_tokens["cache_read"] += usage_dict.get("cache_read_tokens", 0)
        state.session_tokens["output"] += usage_dict.get("output_tokens", 0)
