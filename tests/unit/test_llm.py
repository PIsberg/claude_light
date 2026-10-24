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

import numpy as np

from claude_light.llm import (
    route_query, _extract_text, _build_system_blocks,
    _accumulate_usage, _summarize_turns, _maybe_compress_history,
    _PROTO_CACHE,
)
from claude_light.config import SYSTEM_PROMPT, MAX_HISTORY_TURNS, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS
import claude_light.state as _state
import claude_light.llm as _llm


# ---------------------------------------------------------------------------
# route_query
# ---------------------------------------------------------------------------

class TestRouteQuery(unittest.TestCase):

    def setUp(self):
        # Ensure embedder is None and state is clean for every test
        _state.embedder = None
        _state.embedder_ready.clear()
        _state.EMBED_MODEL = None
        _state.conversation_history = []
        _PROTO_CACHE.update(model=None, mat=None)

    def tearDown(self):
        _state.embedder = None
        _state.embedder_ready.clear()
        _state.EMBED_MODEL = None
        _state.conversation_history = []
        _PROTO_CACHE.update(model=None, mat=None)

    def test_low_effort_simple_lookup(self):
        print("\n  ▶ TestRouteQuery.test_low_effort_simple_lookup")
        model, effort, tokens = route_query("list all files")
        self.assertEqual(effort, "low")

    def test_medium_effort_default(self):
        print("\n  ▶ TestRouteQuery.test_medium_effort_default")
        # Single arch signal ("compare"), no length bonus → score 4.0 → medium.
        model, effort, tokens = route_query("compare different approaches")
        self.assertEqual(effort, "medium")

    def test_low_effort_long_lookup(self):
        # Regression: an infra-only lookup that happens to be 5+ words used to
        # be promoted to medium by the word-count multiplier. It belongs on
        # Haiku — "list all public classes or modules in this codebase" is
        # still a lookup.
        print("\n  ▶ TestRouteQuery.test_low_effort_long_lookup")
        for q in (
            "list all public classes or modules in this codebase",
            "where is the main entry point defined",
            "what python version is required by this project",
        ):
            _, effort, _ = route_query(q)
            self.assertEqual(effort, "low", msg=f"query: {q!r}")

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

    def test_embedder_none_fallback_matches_baseline(self):
        # With embedder=None the fused score == pure keyword score.
        # The existing tier expectations must all hold.
        print("\n  ▶ TestRouteQuery.test_embedder_none_fallback_matches_baseline")
        cases = [
            ("list all files", "low"),
            ("refactor the parse method to handle exceptions", "high"),
            ("evaluate the architecture and scalability deeply", "max"),
        ]
        for q, expected in cases:
            _, effort, _ = route_query(q)
            self.assertEqual(effort, expected, msg=f"query: {q!r}")

    def test_semantic_boost_promotes(self):
        # A query with NO keyword signals should be promoted when semantic
        # similarity to arch prototypes is high (sim > FLOOR).
        print("\n  ▶ TestRouteQuery.test_semantic_boost_promotes")

        # Baseline: embedder None → purely keyword, "overview of the system" = score 0 → low.
        _, effort_no_embed, _ = route_query("overview of the system")
        self.assertEqual(effort_no_embed, "low", msg="baseline (no embedder) should be low")

        # Now install a mock embedder that returns vectors with high arch similarity.
        # 9 prototype phrases → shape (9, 4); all arch prototypes (rows 0-2) point to dim 0.
        all_proto_vecs = np.zeros((9, 4), dtype=np.float32)
        all_proto_vecs[:3, 0] = 1.0   # arch protos: high sim to query below
        all_proto_vecs[3:6, 1] = 1.0  # logic protos: orthogonal
        all_proto_vecs[6:9, 2] = 1.0  # infra protos: orthogonal

        query_vec = np.array([0.99, 0.0, 0.0, 0.14], dtype=np.float32)
        query_vec /= np.linalg.norm(query_vec)

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = lambda texts, **kw: (
            all_proto_vecs if isinstance(texts, list) else query_vec
        )

        _PROTO_CACHE.update(model=None, mat=None)
        _state.embedder = mock_embedder
        _state.EMBED_MODEL = "mock-model"
        _state.embedder_ready.set()

        _, effort_with_embed, _ = route_query("overview of the system")

        # Semantic arch score should lift the query above low.
        self.assertNotEqual(effort_with_embed, "low",
                            msg="semantic boost should lift keyword-free arch query above low")

    def test_cost_ceiling_demotes_borderline_opus(self):
        # A borderline score with a very high escalation margin → demote to high.
        # "evaluate the architecture deeply" → 3 arch signals = score 12, +word bonus 1.5 = 13.5.
        # With margin=0.99: required_score = 12 * 1.99 = 23.88 > 13.5 → demoted.
        print("\n  ▶ TestRouteQuery.test_cost_ceiling_demotes_borderline_opus")
        with patch.object(_llm, 'ROUTER_COST_CEILING', 0.001), \
             patch.object(_llm, 'ROUTER_ESCALATE_MARGIN', 0.99):
            _, effort, _ = route_query("evaluate the architecture deeply")
        self.assertNotEqual(effort, "max",
                            msg="extremely high margin should demote even a clearly architectural query")

    def test_high_confidence_lookup_downgrades_to_haiku(self):
        # A verbose infra-dominant query must stay on Haiku regardless of word count.
        print("\n  ▶ TestRouteQuery.test_high_confidence_lookup_downgrades_to_haiku")
        q = "list all public classes or modules or files or directories in this entire codebase"
        model, effort, tokens = route_query(q)
        self.assertEqual(effort, "low")
        self.assertEqual(model, MODEL_HAIKU)
        self.assertEqual(tokens, 2048)

    def test_prototype_cache_built_once(self):
        # The prototype embedder.encode should only be called once for the
        # prototype build; a second route_query reuses the cache.
        print("\n  ▶ TestRouteQuery.test_prototype_cache_built_once")
        from unittest.mock import MagicMock

        # 9 prototype phrases batched into one call → shape (9, 4).
        all_proto_vecs = np.zeros((9, 4), dtype=np.float32)
        all_proto_vecs[:, 0] = 1.0
        query_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        call_log = []

        def fake_encode(texts, **kw):
            call_log.append(texts)
            return all_proto_vecs if isinstance(texts, list) else query_vec

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = fake_encode

        _state.embedder = mock_embedder
        _state.EMBED_MODEL = "cache-test-model"
        _state.embedder_ready.set()

        route_query("where is main defined")
        route_query("where is the entry point")

        # First call: 1 prototype batch + 1 query encode = 2 encode calls.
        # Second call: cache hit (no prototype build) + 1 query encode = 1 encode call.
        # Total proto-build calls = 1, i.e. exactly one list-valued call.
        list_calls = [c for c in call_log if isinstance(c, list)]
        self.assertEqual(len(list_calls), 1,
                         msg=f"prototype encode should run only once; got {len(list_calls)}")


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


# ---------------------------------------------------------------------------
# _make_cli_subprocess_call — OAUTH stream-json path
# ---------------------------------------------------------------------------

def _fake_popen(stdout_lines, returncode=0, stderr=""):
    """Build a mock Popen that yields the given stream-json lines on stdout."""
    mock = MagicMock()
    mock.stdout = iter(l + "\n" for l in stdout_lines)
    mock.stderr = io.StringIO(stderr)
    mock.wait.return_value = returncode
    mock.poll.return_value = returncode  # watchdog's poll sees "done"
    mock.returncode = returncode

    def _kill():
        mock.poll.return_value = returncode
    mock.kill = _kill
    return mock


class TestCliSubprocessStreaming(unittest.TestCase):
    """Verify the Popen/stream-json parsing in _make_cli_subprocess_call."""

    def _call_with_events(self, events, returncode=0, stderr=""):
        """Run _make_cli_subprocess_call with a mocked Popen emitting events."""
        import json as _json
        from claude_light import llm

        lines = [_json.dumps(e) for e in events]
        mock_proc = _fake_popen(lines, returncode=returncode, stderr=stderr)

        # Swallow streamed stdout so tests don't spam the terminal.
        # shutil.which is patched because on Linux/CI the `claude` CLI is
        # likely not installed and _make_cli_subprocess_call raises before
        # Popen is ever reached. On Windows the function bypasses which()
        # altogether (uses shell=True), so the patch is a no-op there.
        captured = io.StringIO()
        with patch("claude_light.llm.subprocess.Popen", return_value=mock_proc), \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch("sys.stdout", captured):
            result = llm._make_cli_subprocess_call("hello")
        return result, captured.getvalue(), mock_proc

    def test_streams_content_block_delta_text(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_streams_content_block_delta_text")
        events = [
            {"type": "system", "subtype": "init"},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello "},
            }},
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "world"},
            }},
            {"type": "result", "subtype": "success", "result": "Hello world",
             "usage": {"input_tokens": 10, "output_tokens": 2,
                       "cache_creation_input_tokens": 0,
                       "cache_read_input_tokens": 100},
             "session_id": "sess-xyz"},
        ]
        (reply, usage, sid), output, _ = self._call_with_events(events)
        self.assertEqual(reply, "Hello world")
        self.assertEqual(sid, "sess-xyz")
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 2)
        self.assertEqual(usage.cache_read_input_tokens, 100)
        self.assertIn("Hello ", output)
        self.assertIn("world", output)

    def test_falls_back_to_assistant_event_when_no_deltas(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_falls_back_to_assistant_event_when_no_deltas")
        # Older CLI that doesn't honor --include-partial-messages emits
        # only a complete `assistant` event before `result`.
        events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Complete reply"},
            ]}},
            {"type": "result", "subtype": "success", "result": "Complete reply",
             "usage": {"input_tokens": 5, "output_tokens": 3}},
        ]
        (reply, _usage, _sid), output, _ = self._call_with_events(events)
        self.assertEqual(reply, "Complete reply")
        self.assertIn("Complete reply", output)

    def test_skips_assistant_event_if_deltas_already_streamed(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_skips_assistant_event_if_deltas_already_streamed")
        # Modern CLI emits BOTH stream_event deltas AND a complete assistant
        # event; we must not double-print.
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Streamed"},
            }},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Streamed"},
            ]}},
            {"type": "result", "subtype": "success", "result": "Streamed",
             "usage": {"input_tokens": 1, "output_tokens": 1}},
        ]
        (reply, _, _), output, _ = self._call_with_events(events)
        # Reply should be "Streamed" once, not duplicated.
        self.assertEqual(reply, "Streamed")
        self.assertEqual(output.count("Streamed"), 1)

    def test_fallback_to_result_text_when_no_events_seen(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_fallback_to_result_text_when_no_events_seen")
        # If neither stream_event nor assistant events arrive (e.g. the CLI
        # emits a single result-only message), we should still return the
        # result text so the caller has something to show.
        events = [
            {"type": "result", "subtype": "success", "result": "Only in result",
             "usage": {"input_tokens": 2, "output_tokens": 4}},
        ]
        (reply, _, _), output, _ = self._call_with_events(events)
        self.assertEqual(reply, "Only in result")
        self.assertIn("Only in result", output)

    def test_nonzero_exit_raises_runtime_error(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_nonzero_exit_raises_runtime_error")
        from claude_light import llm
        with self.assertRaises(RuntimeError):
            self._call_with_events(
                [{"type": "result", "subtype": "error", "result": "boom"}],
                returncode=1,
                stderr="some CLI error",
            )

    def test_not_logged_in_raises_dedicated_exception(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_not_logged_in_raises_dedicated_exception")
        from claude_light.llm import ClaudeNotLoggedIn
        with self.assertRaises(ClaudeNotLoggedIn):
            self._call_with_events(
                [],
                returncode=1,
                stderr="Not logged in",
            )

    def test_ignores_malformed_json_lines(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_ignores_malformed_json_lines")
        from claude_light import llm
        # Real CLI occasionally emits warning lines that aren't valid JSON
        # (e.g. Node deprecation warnings); we must not crash on them.
        lines = [
            "(node:1234) DeprecationWarning: blah",  # non-JSON
            '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}}',
            '{"type":"result","subtype":"success","result":"ok","usage":{}}',
        ]
        mock_proc = _fake_popen(lines, returncode=0)
        captured = io.StringIO()
        with patch("claude_light.llm.subprocess.Popen", return_value=mock_proc), \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch("sys.stdout", captured):
            reply, _, _ = llm._make_cli_subprocess_call("hi")
        self.assertEqual(reply, "ok")

    def test_heartbeat_stopped_after_success(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_heartbeat_stopped_after_success")
        # After a successful call, the heartbeat thread must be joined —
        # a stray daemon thread would keep painting "Processing…" over
        # later output.
        import threading as _threading
        before = {t.name for t in _threading.enumerate()}
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hi"},
            }},
            {"type": "result", "subtype": "success", "result": "hi", "usage": {}},
        ]
        self._call_with_events(events)
        # Allow a brief moment for the heartbeat's .join(timeout=1.5) to finish
        import time as _time
        _time.sleep(0.1)
        after = {t.name for t in _threading.enumerate()}
        # No new threads named anything heartbeat-ish should be left running.
        self.assertEqual(
            before, after,
            msg=f"Leaked threads: {after - before}",
        )

    def test_heartbeat_stopped_on_error_path(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_heartbeat_stopped_on_error_path")
        # Error in subprocess must still clean up the heartbeat — otherwise
        # after a failed call the animated line ticks forever over the next
        # prompt.
        import threading as _threading
        before = {t.name for t in _threading.enumerate()}
        with self.assertRaises(RuntimeError):
            self._call_with_events(
                [{"type": "result", "subtype": "error", "result": "boom"}],
                returncode=1,
                stderr="some CLI error",
            )
        import time as _time
        _time.sleep(0.1)
        after = {t.name for t in _threading.enumerate()}
        self.assertEqual(
            before, after,
            msg=f"Leaked threads on error path: {after - before}",
        )

    def test_isolates_home_to_neutralize_user_claude_md(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_isolates_home_to_neutralize_user_claude_md")
        # The subprocess env must have HOME/USERPROFILE pointing at an
        # isolated temp dir (not the real user HOME), with an empty CLAUDE.md
        # inside. Otherwise the Claude CLI auto-discovers ~/.claude/CLAUDE.md
        # and the model hallucinates tool calls from instructions like
        # "always prefer ctx_read/ctx_shell".
        import os as _os
        import pathlib as _pl
        from claude_light import llm

        real_home = _os.path.expanduser("~")
        captured = io.StringIO()
        mock_proc = _fake_popen(
            ['{"type":"result","subtype":"success","result":"","usage":{}}'],
            returncode=0,
        )
        with patch("claude_light.llm.subprocess.Popen", return_value=mock_proc) as popen_mock, \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch("sys.stdout", captured):
            llm._make_cli_subprocess_call("hi")

        env = popen_mock.call_args.kwargs.get("env") or {}
        isolated_home = env.get("HOME")
        self.assertIsNotNone(isolated_home, "HOME must be set on the subprocess env")
        self.assertNotEqual(
            _os.path.normcase(isolated_home),
            _os.path.normcase(real_home),
            "HOME must NOT be the real user HOME",
        )
        self.assertEqual(env.get("USERPROFILE"), isolated_home)
        # An empty CLAUDE.md must exist to suppress auto-discovery
        claude_md = _pl.Path(isolated_home) / ".claude" / "CLAUDE.md"
        # Note: by the time we assert here, the finally block has cleaned up
        # the tempdir. We just check that the env pointed there and that the
        # subprocess would have seen the right shape.
        self.assertTrue(str(claude_md).endswith("CLAUDE.md"))

    def test_user_claude_md_excluded_even_from_ancestor_walk(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_user_claude_md_excluded_even_from_ancestor_walk")
        # For a repo under the user's home (~/code/x), the CLI's walk up
        # parent directories finds ~/.claude/CLAUDE.md as an ancestor's
        # project memory, so the HOME override alone does not keep it out
        # (verified live with claude 2.1.310). It must be excluded by path.
        import json as _json
        import os as _os
        from claude_light import llm
        seen = {}

        def fake_popen(cmd, **kwargs):
            idx = cmd.index("--settings")
            with open(cmd[idx + 1], encoding="utf-8") as fh:
                seen.update(_json.load(fh))
            return _fake_popen(['{"type":"result","subtype":"success","result":"","usage":{}}'])

        with patch("claude_light.llm.subprocess.Popen", side_effect=fake_popen), \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch("sys.stdout", io.StringIO()):
            llm._make_cli_subprocess_call("hi")

        real = (_os.path.expanduser("~").replace("\\", "/") + "/.claude/CLAUDE.md")
        self.assertIn(real, seen.get("claudeMdExcludes", []))

    def test_command_uses_stream_json_flags(self):
        print("\n  ▶ TestCliSubprocessStreaming.test_command_uses_stream_json_flags")
        # Lock in the CLI flags we rely on — if any of these change, the
        # stream-json parser above will silently fail to produce output.
        from claude_light import llm
        captured = io.StringIO()
        mock_proc = _fake_popen(
            ['{"type":"result","subtype":"success","result":"","usage":{}}'],
            returncode=0,
        )
        with patch("claude_light.llm.subprocess.Popen", return_value=mock_proc) as popen_mock, \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch("sys.stdout", captured):
            llm._make_cli_subprocess_call("hi")
        # First positional arg is the command list
        cmd = popen_mock.call_args.args[0]
        self.assertIn("--output-format", cmd)
        idx = cmd.index("--output-format")
        self.assertEqual(cmd[idx + 1], "stream-json")
        self.assertIn("--verbose", cmd)
        self.assertIn("--include-partial-messages", cmd)


def _text_block_events(text):
    """stream-json events for one streamed text block, as the CLI emits them."""
    return [
        {"type": "stream_event", "event": {
            "type": "content_block_start", "content_block": {"type": "text", "text": ""}}},
        {"type": "stream_event", "event": {
            "type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "stream_event", "event": {"type": "content_block_stop"}},
    ]


def _tool_use_event(tool_id, name, tool_input):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]}}


class TestCliAgentActivity(unittest.TestCase):
    """In OAuth mode the CLI agent reads and edits files on its own. The user
    must see those actions as they happen, and narration from separate
    agent steps must not run together into one paragraph."""

    _call_with_events = TestCliSubprocessStreaming._call_with_events

    _result = {"type": "result", "subtype": "success", "result": "", "usage": {}}

    def test_tool_calls_are_shown_with_relative_paths(self):
        print("\n  ▶ TestCliAgentActivity.test_tool_calls_are_shown_with_relative_paths")
        import os as _os
        abs_path = _os.path.join(_os.getcwd(), "claude_light", "retry.py")
        events = (
            _text_block_events("Let me look.")
            + [_tool_use_event("t1", "Read", {"file_path": abs_path}),
               _tool_use_event("t2", "Grep", {"pattern": "MAX_RETRIES", "path": "."})]
            + _text_block_events("It is 3.")
            + [self._result]
        )
        (reply, _, _), output, _ = self._call_with_events(events)
        self.assertIn("Read", output)
        self.assertIn("claude_light/retry.py", output.replace("\\", "/"))
        self.assertNotIn(_os.getcwd(), output)
        self.assertIn("Grep", output)
        self.assertIn("MAX_RETRIES", output)
        # Tool lines are UI only; they must not leak into the stored reply.
        self.assertNotIn("Read", reply)

    def test_text_blocks_from_separate_steps_are_separated(self):
        print("\n  ▶ TestCliAgentActivity.test_text_blocks_from_separate_steps_are_separated")
        events = (
            _text_block_events("First step.")
            + [_tool_use_event("t1", "Read", {"file_path": "a.py"})]
            + _text_block_events("Second step.")
            + [self._result]
        )
        (reply, _, _), output, _ = self._call_with_events(events)
        self.assertEqual(reply, "First step.\n\nSecond step.")
        self.assertNotIn("First step.Second", output)

    def test_repeated_tool_event_is_shown_once(self):
        print("\n  ▶ TestCliAgentActivity.test_repeated_tool_event_is_shown_once")
        ev = _tool_use_event("t1", "Edit", {"file_path": "a.py"})
        (_, _, _), output, _ = self._call_with_events([ev, ev, self._result])
        self.assertEqual(output.count("Edit"), 1)


class TestChatWithoutIndex(unittest.TestCase):
    """While the first index build runs (minutes on a cold start), OAuth
    queries go straight to the CLI agent, which can search the code itself."""

    def setUp(self):
        from claude_light import state
        self._hist = list(state.conversation_history)

    def tearDown(self):
        from claude_light import state
        state.conversation_history[:] = self._hist

    def test_use_index_false_skips_retrieval(self):
        print("\n  ▶ TestChatWithoutIndex.test_use_index_false_skips_retrieval")
        from claude_light import llm
        captured = {}

        def fake_call(**kwargs):
            captured.update(kwargs)
            usage = MagicMock(input_tokens=1, output_tokens=1,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0)
            return "answer", usage, True

        out = io.StringIO()
        with patch.object(llm, "retrieve") as retrieve_mock,              patch.object(llm, "_make_streaming_api_call", side_effect=fake_call),              patch.object(llm, "route_query", return_value=("m", "medium", 4096)),              patch.object(llm, "record_retrieval"),              patch.object(llm, "print_stats"),              patch("sys.stdout", out):
            llm.chat("where is the retry loop?", use_index=False)

        retrieve_mock.assert_not_called()
        self.assertNotIn("Retrieved Codebase Context", str(captured["messages"]))
        self.assertIn("index", out.getvalue().lower())


class _PacedProc:
    """Popen stand-in whose stdout emits lines on a schedule.

    ``schedule`` is a list of (delay_secs, line). A delay of None means "go
    silent until killed" (a hung CLI). kill() ends the stream immediately.
    """

    def __init__(self, schedule):
        import threading as _threading
        self._schedule = schedule
        self._killed = _threading.Event()
        self._done = False
        self.stderr = io.StringIO("")
        self.returncode = 0

    @property
    def stdout(self):
        for delay, line in self._schedule:
            # Hung CLI: block until killed, with a safety cap so a broken
            # implementation fails the test instead of hanging the suite.
            if self._killed.wait(5.0 if delay is None else delay):
                break
            yield line + "\n"
        self._done = True

    def poll(self):
        return -9 if self._killed.is_set() else (0 if self._done else None)

    def wait(self):
        return -9 if self._killed.is_set() else 0

    def kill(self):
        self._killed.set()


class TestCliSubprocessTimeouts(unittest.TestCase):
    """The CLI runs a full agent loop (read, edit, re-read), so a long task is
    normal. Only a CLI that stops producing events should be killed."""

    def _run(self, schedule, idle=0.6, cap=30.0):
        from claude_light import llm
        proc = _PacedProc(schedule)
        with patch("claude_light.llm.subprocess.Popen", return_value=proc), \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch.object(llm, "_CLI_IDLE_TIMEOUT_SECS", idle), \
             patch.object(llm, "_CLI_MAX_SECS", cap), \
             patch("sys.stdout", io.StringIO()):
            return llm._make_cli_subprocess_call("hi")

    def test_long_but_active_run_is_not_killed(self):
        print("\n  ▶ TestCliSubprocessTimeouts.test_long_but_active_run_is_not_killed")
        delta = '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}}'
        # 8 events 0.25 s apart: 2 s total, well past the 0.6 s idle limit,
        # but never silent for longer than 0.25 s.
        schedule = [(0.25, delta)] * 8 + [
            (0.0, '{"type":"result","subtype":"success","result":"xxxxxxxx","usage":{}}')]
        reply, _, _ = self._run(schedule)
        self.assertEqual(reply, "xxxxxxxx")

    def test_silent_cli_is_killed_with_idle_message(self):
        print("\n  ▶ TestCliSubprocessTimeouts.test_silent_cli_is_killed_with_idle_message")
        schedule = [
            (0.0, '{"type":"system","subtype":"init"}'),
            (None, ""),  # hangs
        ]
        with self.assertRaises(RuntimeError) as ctx:
            self._run(schedule)
        self.assertIn("no output", str(ctx.exception))

    def test_total_cap_still_applies(self):
        print("\n  ▶ TestCliSubprocessTimeouts.test_total_cap_still_applies")
        delta = '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}}'
        schedule = [(0.2, delta)] * 40  # 8 s of steady output
        with self.assertRaises(RuntimeError) as ctx:
            self._run(schedule, idle=5.0, cap=1.0)
        self.assertIn("timed out", str(ctx.exception))


def _pid_alive(pid):
    import os as _os
    import subprocess as _sp
    if _os.name == "nt":
        out = _sp.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                      capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        _os.kill(pid, 0)
    except OSError:
        return False
    return True


class TestCliProcessIsStopped(unittest.TestCase):
    """The CLI agent edits files with permission prompts disabled, so when a
    turn ends early (Ctrl+C, idle timeout) the agent process itself must be
    dead, not just the shell wrapper around it (on Windows the command runs
    via cmd.exe, so killing the Popen handle alone leaves claude.exe alive)."""

    _SCRIPT = (
        "import os, sys, time, json\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init'}), flush=True)\n"
        "for _ in range(600):\n"
        "    if sys.argv[2] == 'chatty':\n"
        "        print(json.dumps({'type': 'system', 'subtype': 'status'}), flush=True)\n"
        "    time.sleep(0.1)\n"
    )

    def _run(self, mode, interrupt_after=None, idle=30.0):
        import subprocess as _sp
        import sys as _sys
        import tempfile as _tf
        import threading as _threading
        import _thread
        import time as _time
        from claude_light import llm

        import shutil as _shutil
        work = _tf.mkdtemp(prefix="cl_agent_")
        self.addCleanup(_shutil.rmtree, work, True)
        pid_file = os.path.join(work, "agent.pid")
        script = os.path.join(work, "agent.py")
        with open(script, "w") as fh:
            fh.write(self._SCRIPT)
        real_popen = _sp.Popen

        def fake_popen(cmd, **kwargs):
            if isinstance(cmd, list) and cmd and cmd[0] == "taskkill":
                return real_popen(cmd, **kwargs)
            # Same shell/env/pipes as the real call, but a stand-in agent.
            return real_popen([_sys.executable, script, pid_file, mode], **kwargs)

        if interrupt_after is not None:
            _threading.Timer(interrupt_after, _thread.interrupt_main).start()
        start = _time.monotonic()
        error = None
        with patch("claude_light.llm.subprocess.Popen", side_effect=fake_popen), \
             patch("shutil.which", return_value=_sys.executable), \
             patch.object(llm, "_CLI_IDLE_TIMEOUT_SECS", idle), \
             patch("sys.stdout", io.StringIO()):
            try:
                llm._make_cli_subprocess_call("hi")
            except (KeyboardInterrupt, RuntimeError) as e:
                error = e
        elapsed = _time.monotonic() - start
        with open(pid_file) as fh:
            agent_pid = int(fh.read())
        _time.sleep(0.5)
        return error, elapsed, agent_pid

    def test_ctrl_c_kills_the_agent_process(self):
        print("\n  ▶ TestCliProcessIsStopped.test_ctrl_c_kills_the_agent_process")
        error, _, agent_pid = self._run("chatty", interrupt_after=1.5)
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertFalse(_pid_alive(agent_pid), "agent process survived Ctrl+C")

    def test_idle_timeout_kills_the_agent_process_promptly(self):
        print("\n  ▶ TestCliProcessIsStopped.test_idle_timeout_kills_the_agent_process_promptly")
        error, elapsed, agent_pid = self._run("silent", idle=1.0)
        self.assertIsInstance(error, RuntimeError)
        self.assertLess(elapsed, 15, "turn did not end until the agent exited on its own")
        self.assertFalse(_pid_alive(agent_pid), "agent process survived the idle timeout")


class TestAgentEditDetection(unittest.TestCase):
    """Verify _git_modified_snapshot + _commit_agent_edits — the OAUTH safety
    net that catches files the Claude CLI agent edited directly (outside our
    SEARCH/REPLACE pipeline) and auto-commits them."""

    def test_snapshot_returns_modified_files_when_in_repo(self):
        print("\n  ▶ TestAgentEditDetection.test_snapshot_returns_modified_files_when_in_repo")
        from claude_light import llm
        with patch("claude_light.git_manager.is_git_repo", return_value=True), \
             patch("claude_light.git_manager.get_modified_files",
                   return_value=["a.py", "b.md"]):
            snap = llm._git_modified_snapshot()
        self.assertEqual(snap, {"a.py", "b.md"})

    def test_snapshot_is_empty_outside_repo(self):
        print("\n  ▶ TestAgentEditDetection.test_snapshot_is_empty_outside_repo")
        from claude_light import llm
        with patch("claude_light.git_manager.is_git_repo", return_value=False):
            snap = llm._git_modified_snapshot()
        self.assertEqual(snap, set())

    def test_commit_agent_edits_calls_auto_commit(self):
        print("\n  ▶ TestAgentEditDetection.test_commit_agent_edits_calls_auto_commit")
        from claude_light import llm
        captured = io.StringIO()
        with patch("sys.stdout", captured), \
             patch("claude_light.llm.subprocess.run") as mock_diff, \
             patch("claude_light.git_manager.auto_commit") as mock_commit:
            mock_diff.return_value = MagicMock(stdout="+ new line\n- old line\n")
            llm._commit_agent_edits({"foo.py"}, "updated foo", auto_apply=True)
        mock_commit.assert_called_once()
        args, kwargs = mock_commit.call_args
        # auto_commit(files, explanation)
        self.assertEqual(args[0], ["foo.py"])
        self.assertEqual(args[1], "updated foo")

    def test_commit_agent_edits_noop_when_empty(self):
        print("\n  ▶ TestAgentEditDetection.test_commit_agent_edits_noop_when_empty")
        from claude_light import llm
        with patch("claude_light.git_manager.auto_commit") as mock_commit:
            llm._commit_agent_edits(set(), "explanation", auto_apply=True)
        mock_commit.assert_not_called()

    def test_commit_agent_edits_respects_declined_interactive_prompt(self):
        print("\n  ▶ TestAgentEditDetection.test_commit_agent_edits_respects_declined_interactive_prompt")
        from claude_light import llm
        captured = io.StringIO()
        # Force interactive path with stdin.isatty() true, then user says 'n'
        with patch("sys.stdout", captured), \
             patch("claude_light.llm.subprocess.run") as mock_diff, \
             patch("claude_light.git_manager.auto_commit") as mock_commit, \
             patch("sys.stdin") as mock_stdin, \
             patch("builtins.input", return_value="n"):
            mock_stdin.isatty.return_value = True
            mock_diff.return_value = MagicMock(stdout="")
            llm._commit_agent_edits({"foo.py"}, "explanation", auto_apply=False)
        mock_commit.assert_not_called()
        self.assertIn("not committed", captured.getvalue().lower())


class TestHeartbeat(unittest.TestCase):
    """Verify the _Heartbeat context manager's lifecycle and rendering."""

    def test_emits_elapsed_seconds_counter(self):
        print("\n  ▶ TestHeartbeat.test_emits_elapsed_seconds_counter")
        from claude_light.llm import _Heartbeat
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            with _Heartbeat("Working", interval=0.05):
                import time as _time
                # Let ~3 ticks fire. interval=0.05 so this is fast.
                _time.sleep(0.22)
        output = captured.getvalue()
        # The initial frame has "Working…" without a counter; later frames
        # add "(Ns)". We should see at least one timestamped frame.
        self.assertIn("Working…", output)
        self.assertRegex(output, r"Working… \(\d+s\)")

    def test_stop_clears_line_and_is_idempotent(self):
        print("\n  ▶ TestHeartbeat.test_stop_clears_line_and_is_idempotent")
        from claude_light.llm import _Heartbeat
        hb = _Heartbeat("X", interval=0.05)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            hb.__enter__()
            hb.stop()
            hb.stop()  # must not raise or re-print
            hb.__exit__(None, None, None)  # also idempotent
        output = captured.getvalue()
        # Last output chunk should contain the \r\033[K clear sequence.
        self.assertIn("\r\x1b[K", output)

    def test_no_thread_leak_after_exit(self):
        print("\n  ▶ TestHeartbeat.test_no_thread_leak_after_exit")
        import threading as _threading
        from claude_light.llm import _Heartbeat
        before = {t.name for t in _threading.enumerate()}
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            with _Heartbeat("Z", interval=0.05):
                import time as _time
                _time.sleep(0.1)
        import time as _time
        _time.sleep(0.1)  # let join finish
        after = {t.name for t in _threading.enumerate()}
        self.assertEqual(before, after, msg=f"Leaked: {after - before}")


if __name__ == "__main__":
    unittest.main()
