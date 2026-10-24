"""
Streaming response handler for Claude Light.

Consumes a token stream from the Anthropic API, renders it through
ui.ResponseView as it arrives, and collects the usage figures.
"""

from typing import Iterator, Tuple, Dict, Any
from claude_light.config import model_prices
from claude_light.ui import ResponseView
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

    def process_stream(self, stream: Iterator, view=None) -> Tuple[str, Dict[str, int]]:
        """Consume a stream, rendering the reply into `view`.

        A tool-loop turn passes one view for all its rounds; without one the
        handler opens and closes its own.
        """
        own_view = view is None
        if own_view:
            view = ResponseView("Thinking").__enter__()
        try:
            for event in stream:
                self._handle(event, view)
        finally:
            if own_view:
                view.close()
        return self.buffer, self._get_usage_dict()

    def _handle(self, event, view):
        """Update buffers and usage from one stream event."""
        if not hasattr(event, "type"):
            return
        if event.type == "content_block_start":
            block = getattr(event, "content_block", None)
            if block is not None and getattr(block, "type", None) == "thinking":
                self.is_thinking = True
                view.status("Thinking")
        elif event.type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if not delta:
                return
            if self.is_thinking:
                thinking = getattr(delta, "thinking", None)
                if thinking:
                    self.thinking_buffer += thinking
            else:
                text = getattr(delta, "text", None)
                if text:
                    self.buffer += text
                    view.text(text)
        elif event.type == "content_block_stop":
            if self.is_thinking:
                self.is_thinking = False
                view.status("Writing")
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

    def _get_usage_dict(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
        }


def stream_chat_response(client, view=None, **create_kwargs) -> Tuple[str, Dict[str, int]]:
    """Execute a streaming chat request and return full response with usage."""
    handler = StreamingResponseHandler()
    with client.messages.stream(**create_kwargs) as stream:
        response_text, usage = handler.process_stream(stream, view=view)
        # The complete message (stop_reason, tool_use blocks, exact usage) for
        # the tool loop in llm.py. Streams without it (mocks) simply omit it.
        try:
            usage["message"] = stream.get_final_message()
        except Exception:
            pass
    return response_text, usage


def calculate_usage_cost(usage_dict: Dict[str, int], model=None) -> float:
    """Calculate cost from usage metrics at `model`'s rates."""
    p = model_prices(model)
    inp = usage_dict.get("input_tokens", 0)
    cw = usage_dict.get("cache_creation_tokens", 0)
    cr = usage_dict.get("cache_read_tokens", 0)
    out = usage_dict.get("output_tokens", 0)
    return (
        (inp / 1_000_000) * p["input"]
        + (cw / 1_000_000) * p["write"]
        + (cr / 1_000_000) * p["read"]
        + (out / 1_000_000) * p["output"]
    )


def accumulate_usage_from_dict(usage_dict: Dict[str, int]):
    """Accumulate usage from a streaming usage dict."""
    with state.lock:
        state.session_tokens["input"] += usage_dict.get("input_tokens", 0)
        state.session_tokens["cache_write"] += usage_dict.get("cache_creation_tokens", 0)
        state.session_tokens["cache_read"] += usage_dict.get("cache_read_tokens", 0)
        state.session_tokens["output"] += usage_dict.get("output_tokens", 0)
