"""
Tests for claude_light.llm — route_query, _extract_text, _build_system_blocks,
_accumulate_usage, _summarize_turns, _maybe_compress_history, chat, one_shot, warm_cache.
"""

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.llm import (
    route_query, _extract_text, _build_system_blocks,
    _accumulate_usage, _summarize_turns, _maybe_compress_history,
)
from claude_light.config import SYSTEM_PROMPT, MAX_HISTORY_TURNS
import claude_light.state as _state


# ---------------------------------------------------------------------------
# route_query
# ---------------------------------------------------------------------------

class TestRouteQuery(unittest.TestCase):

    def test_low_effort_simple_lookup(self):
        print("\n  ▶ TestRouteQuery.test_low_effort_simple_lookup")
        model, effort, tokens = route_query("list all files")
        self.assertEqual(effort, "low")

    def test_medium_effort_default(self):
        print("\n  ▶ TestRouteQuery.test_medium_effort_default")
        model, effort, tokens = route_query("what does this function do")
        self.assertEqual(effort, "medium")

    def test_high_effort_code_modification(self):
        print("\n  ▶ TestRouteQuery.test_high_effort_code_modification")
        model, effort, tokens = route_query("refactor the parse method to handle exceptions")
        self.assertEqual(effort, "high")

    def test_max_effort_architectural(self):
        print("\n  ▶ TestRouteQuery.test_max_effort_architectural")
        model, effort, tokens = route_query("evaluate the scalability trade-offs of the current microservices architecture deeply")
        self.assertEqual(effort, "max")

    def test_long_query_routes_high_or_max(self):
        print("\n  ▶ TestRouteQuery.test_long_query_routes_high_or_max")
        query = " ".join(["word"] * 35)
        _, effort, _ = route_query(query)
        self.assertIn(effort, ("high", "max"))

    def test_low_effort_tokens(self):
        print("\n  ▶ TestRouteQuery.test_low_effort_tokens")
        _, effort, max_tokens = route_query("list all files")
        if effort == "low":
            self.assertEqual(max_tokens, 2048)

    def test_max_hits_two_signals(self):
        print("\n  ▶ TestRouteQuery.test_max_hits_two_signals")
        _, effort, _ = route_query("evaluate the architecture and scalability deeply")
        self.assertEqual(effort, "max")


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

class TestExtractText(unittest.TestCase):

    def _block(self, type_, text=None):
        b = MagicMock()
        b.type = type_
        if text is not None:
            b.text = text
        return b

    def test_empty_list(self):
        print("\n  ▶ TestExtractText.test_empty_list")
        self.assertEqual(_extract_text([]), "")

    def test_single_text_block(self):
        print("\n  ▶ TestExtractText.test_single_text_block")
        blocks = [self._block("text", "Hello world")]
        self.assertEqual(_extract_text(blocks), "Hello world")

    def test_multiple_text_blocks(self):
        print("\n  ▶ TestExtractText.test_multiple_text_blocks")
        blocks = [self._block("text", "foo"), self._block("text", "bar")]
        self.assertEqual(_extract_text(blocks), "foobar")

    def test_skips_thinking_blocks(self):
        print("\n  ▶ TestExtractText.test_skips_thinking_blocks")
        blocks = [
            self._block("thinking", "internal thought"),
            self._block("text", "real answer"),
        ]
        self.assertEqual(_extract_text(blocks), "real answer")

    def test_skips_tool_use_blocks(self):
        print("\n  ▶ TestExtractText.test_skips_tool_use_blocks")
        blocks = [
            self._block("tool_use"),
            self._block("text", "answer"),
        ]
        self.assertEqual(_extract_text(blocks), "answer")

    def test_mixed_blocks(self):
        print("\n  ▶ TestExtractText.test_mixed_blocks")
        blocks = [
            self._block("thinking", "thought"),
            self._block("text", "part1 "),
            self._block("tool_use"),
            self._block("text", "part2"),
        ]
        self.assertEqual(_extract_text(blocks), "part1 part2")


# ---------------------------------------------------------------------------
# _build_system_blocks
# ---------------------------------------------------------------------------

class TestBuildSystemBlocks(unittest.TestCase):

    def test_returns_two_blocks(self):
        print("\n  ▶ TestBuildSystemBlocks.test_returns_two_blocks")
        blocks = _build_system_blocks("skeleton text")
        self.assertEqual(len(blocks), 2)

    def test_first_block_is_system_prompt(self):
        print("\n  ▶ TestBuildSystemBlocks.test_first_block_is_system_prompt")
        blocks = _build_system_blocks("skeleton text")
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[0]["text"], SYSTEM_PROMPT)

    def test_second_block_has_skeleton(self):
        print("\n  ▶ TestBuildSystemBlocks.test_second_block_has_skeleton")
        blocks = _build_system_blocks("my skeleton")
        self.assertEqual(blocks[1]["type"], "text")
        self.assertEqual(blocks[1]["text"], "my skeleton")

    def test_second_block_has_cache_control(self):
        print("\n  ▶ TestBuildSystemBlocks.test_second_block_has_cache_control")
        blocks = _build_system_blocks("my skeleton")
        self.assertIn("cache_control", blocks[1])
        self.assertEqual(blocks[1]["cache_control"]["type"], "ephemeral")


# ---------------------------------------------------------------------------
# _accumulate_usage
# ---------------------------------------------------------------------------

class TestAccumulateUsage(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        self._orig = dict(cl.session_tokens)
        cl.session_tokens.update({"input": 0, "cache_write": 0, "cache_read": 0, "output": 0})

    def tearDown(self):
        import claude_light as cl
        cl.session_tokens.update(self._orig)

    def _make_usage(self, inp=0, out=0, cw=0, cr=0):
        usage = MagicMock()
        usage.input_tokens = inp
        usage.output_tokens = out
        usage.cache_creation_input_tokens = cw
        usage.cache_read_input_tokens = cr
        return usage

    def test_accumulates(self):
        print("\n  ▶ TestAccumulateUsage.test_accumulates")
        import claude_light as cl
        usage = self._make_usage(inp=100, out=200, cw=300, cr=400)
        _accumulate_usage(usage)
        self.assertEqual(cl.session_tokens["input"], 100)
        self.assertEqual(cl.session_tokens["output"], 200)
        self.assertEqual(cl.session_tokens["cache_write"], 300)
        self.assertEqual(cl.session_tokens["cache_read"], 400)

    def test_multiple_accumulate(self):
        print("\n  ▶ TestAccumulateUsage.test_multiple_accumulate")
        import claude_light as cl
        _accumulate_usage(self._make_usage(inp=50, out=50))
        _accumulate_usage(self._make_usage(inp=50, out=50))
        self.assertEqual(cl.session_tokens["input"], 100)
        self.assertEqual(cl.session_tokens["output"], 100)


# ---------------------------------------------------------------------------
# _summarize_turns
# ---------------------------------------------------------------------------

class TestSummarizeTurns(unittest.TestCase):

    def test_summarize_turns_calls_api(self):
        print("\n  ▶ TestSummarizeTurns.test_summarize_turns_calls_api")
        messages = [
            {"role": "user", "content": "What is X?"},
            {"role": "assistant", "content": "X is a thing."},
        ]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary of conversation")]
        mock_response.usage = MagicMock()

        with patch("claude_light.llm.client.messages.create", return_value=mock_response):
            summary, usage = _summarize_turns(messages)
        self.assertEqual(summary, "Summary of conversation")

    def test_summarize_turns_with_list_content(self):
        print("\n  ▶ TestSummarizeTurns.test_summarize_turns_with_list_content")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Question?"}]},
            {"role": "assistant", "content": "Answer."},
        ]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_response.usage = MagicMock()

        with patch("claude_light.llm.client.messages.create", return_value=mock_response):
            summary, _ = _summarize_turns(messages)
        self.assertEqual(summary, "Summary")

    def test_empty_messages_no_crash(self):
        print("\n  ▶ TestSummarizeTurns.test_empty_messages_no_crash")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Empty summary")]
        mock_response.usage = MagicMock()
        with patch("claude_light.llm.client.messages.create", return_value=mock_response):
            summary, _ = _summarize_turns([])
        self.assertEqual(summary, "Empty summary")


# ---------------------------------------------------------------------------
# _maybe_compress_history
# ---------------------------------------------------------------------------

class TestMaybeCompressHistory(unittest.TestCase):

    def setUp(self):
        self._orig_hist = list(_state.conversation_history)
        import claude_light as cl
        self._orig_cost = cl.session_cost

    def tearDown(self):
        _state.conversation_history[:] = self._orig_hist
        import claude_light as cl
        cl.session_cost = self._orig_cost

    def test_short_history_not_compressed(self):
        print("\n  ▶ TestMaybeCompressHistory.test_short_history_not_compressed")
        _state.conversation_history[:] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        orig_len = len(_state.conversation_history)
        _maybe_compress_history()
        self.assertEqual(len(_state.conversation_history), orig_len)

    def test_long_history_triggers_compression(self):
        print("\n  ▶ TestMaybeCompressHistory.test_long_history_triggers_compression")
        long_hist = []
        for i in range(20):
            long_hist.append({"role": "user", "content": f"question {i}"})
            long_hist.append({"role": "assistant", "content": f"answer {i}"})
        _state.conversation_history[:] = long_hist

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0

        with patch("claude_light.llm._summarize_turns") as mock_summ:
            mock_summ.return_value = ("summary text", mock_usage)
            _maybe_compress_history()
            mock_summ.assert_called_once()
        self.assertLess(len(_state.conversation_history), 40)

    def test_compression_failure_truncates(self):
        print("\n  ▶ TestMaybeCompressHistory.test_compression_failure_truncates")
        long_hist = []
        for i in range(20):
            long_hist.append({"role": "user", "content": f"q{i}"})
            long_hist.append({"role": "assistant", "content": f"a{i}"})
        _state.conversation_history[:] = long_hist

        with patch("claude_light.llm._summarize_turns", side_effect=Exception("API down")):
            _maybe_compress_history()
        self.assertLessEqual(len(_state.conversation_history), MAX_HISTORY_TURNS * 2)


# ---------------------------------------------------------------------------
# chat — fully mocked
# ---------------------------------------------------------------------------

class TestChat(unittest.TestCase):
    """Tests for claude_light.llm.chat — fully isolated."""

    def _mock_retrieve(self):
        return patch("claude_light.llm.retrieve", return_value=("", []))

    def _mock_compress(self):
        return patch("claude_light.llm._maybe_compress_history")

    def _make_usage(self):
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_creation_input_tokens = 0
        usage.cache_read_input_tokens = 0
        return usage

    def _mock_api(self, text="Hello there!"):
        return patch("claude_light.llm._make_streaming_api_call", return_value=(text, self._make_usage(), False))

    def test_chat_simple_query(self):
        print("\n  ▶ TestChat.test_chat_simple_query")
        import claude_light as cl
        orig_hist = list(_state.conversation_history)
        try:
            with self._mock_api("This is the answer."), \
                 self._mock_retrieve(), \
                 self._mock_compress(), \
                 patch("claude_light.llm._print_reply"), \
                 patch("claude_light.llm.print_stats"):
                cl.chat("What does foo do?")
            self.assertGreater(len(_state.conversation_history), 0)
        finally:
            _state.conversation_history[:] = orig_hist

    def test_chat_stores_turns_in_history(self):
        print("\n  ▶ TestChat.test_chat_stores_turns_in_history")
        import claude_light as cl
        orig_hist = list(_state.conversation_history)
        try:
            with self._mock_api("The answer is 42."), \
                 self._mock_retrieve(), \
                 self._mock_compress(), \
                 patch("claude_light.llm._print_reply"), \
                 patch("claude_light.llm.print_stats"):
                cl.chat("What is the answer?")
            self.assertEqual(len(_state.conversation_history), 2)
            self.assertEqual(_state.conversation_history[0]["role"], "user")
            self.assertEqual(_state.conversation_history[1]["role"], "assistant")
        finally:
            _state.conversation_history[:] = orig_hist

    def test_chat_keyboard_interrupt(self):
        print("\n  ▶ TestChat.test_chat_keyboard_interrupt")
        import claude_light as cl
        orig_hist = list(_state.conversation_history)
        try:
            with patch("claude_light.llm._make_streaming_api_call", side_effect=KeyboardInterrupt), \
                 self._mock_retrieve(), \
                 self._mock_compress(), \
                 patch("claude_light.llm._print_reply"), \
                 patch("claude_light.llm.print_stats"):
                cl.chat("some query")  # Should not raise
        finally:
            _state.conversation_history[:] = orig_hist

    def test_chat_api_exception(self):
        print("\n  ▶ TestChat.test_chat_api_exception")
        import claude_light as cl
        orig_hist = list(_state.conversation_history)
        try:
            with patch("claude_light.llm._make_streaming_api_call", side_effect=Exception("API error")), \
                 self._mock_retrieve(), \
                 self._mock_compress(), \
                 patch("claude_light.llm._print_reply"), \
                 patch("claude_light.llm.print_stats"):
                cl.chat("some query")  # Should not raise
        finally:
            _state.conversation_history[:] = orig_hist


# ---------------------------------------------------------------------------
# warm_cache — mocked
# ---------------------------------------------------------------------------

class TestWarmCache(unittest.TestCase):

    def _make_response(self):
        resp = MagicMock()
        resp.usage = MagicMock()
        resp.usage.input_tokens = 50
        resp.usage.output_tokens = 1
        resp.usage.cache_creation_input_tokens = 0
        resp.usage.cache_read_input_tokens = 0
        return resp

    def test_warm_cache_success(self):
        print("\n  ▶ TestWarmCache.test_warm_cache_success")
        from claude_light import warm_cache
        resp = self._make_response()
        with patch("claude_light.llm.client.messages.create", return_value=resp), \
             patch("claude_light.print_stats"):
            warm_cache(quiet=True)

    def test_warm_cache_exception_no_crash(self):
        print("\n  ▶ TestWarmCache.test_warm_cache_exception_no_crash")
        from claude_light import warm_cache
        with patch("claude_light.llm.client.messages.create", side_effect=Exception("net error")):
            warm_cache(quiet=True)


if __name__ == "__main__":
    unittest.main()
