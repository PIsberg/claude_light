"""
Tests for claude_light.ui — calculate_cost, print_stats, print_session_summary, _print_reply.
"""

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.ui import calculate_cost, print_stats, print_session_summary, _print_reply


class TestCalculateCost(unittest.TestCase):

    def _make_usage(self, input_tokens=0, output_tokens=0, cache_creation=0, cache_read=0):
        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        usage.cache_creation_input_tokens = cache_creation
        usage.cache_read_input_tokens = cache_read
        return usage

    def test_zero_usage(self):
        print("\n  ▶ TestCalculateCost.test_zero_usage")
        usage = self._make_usage()
        self.assertEqual(calculate_cost(usage), 0.0)

    def test_output_only(self):
        print("\n  ▶ TestCalculateCost.test_output_only")
        usage = self._make_usage(output_tokens=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 15.0, places=4)

    def test_input_only(self):
        print("\n  ▶ TestCalculateCost.test_input_only")
        usage = self._make_usage(input_tokens=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 3.0, places=4)

    def test_cache_write(self):
        print("\n  ▶ TestCalculateCost.test_cache_write")
        usage = self._make_usage(cache_creation=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 3.75, places=4)

    def test_cache_read(self):
        print("\n  ▶ TestCalculateCost.test_cache_read")
        usage = self._make_usage(cache_read=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 0.30, places=4)

    def test_combined(self):
        print("\n  ▶ TestCalculateCost.test_combined")
        usage = self._make_usage(
            input_tokens=100_000,
            output_tokens=100_000,
            cache_creation=100_000,
            cache_read=100_000,
        )
        expected = (100_000/1_000_000)*3.0 + (100_000/1_000_000)*15.0 + (100_000/1_000_000)*3.75 + (100_000/1_000_000)*0.30
        self.assertAlmostEqual(calculate_cost(usage), expected, places=6)

    def test_no_cache_attrs(self):
        print("\n  ▶ TestCalculateCost.test_no_cache_attrs")
        usage = MagicMock(spec=["input_tokens", "output_tokens"])
        usage.input_tokens = 500_000
        usage.output_tokens = 500_000
        cost = calculate_cost(usage)
        self.assertAlmostEqual(cost, 1.5 + 7.5, places=4)


class TestPrintStats(unittest.TestCase):

    def _make_usage(self, inp=0, out=0, cw=0, cr=0):
        usage = MagicMock()
        usage.input_tokens = inp
        usage.output_tokens = out
        usage.cache_creation_input_tokens = cw
        usage.cache_read_input_tokens = cr
        return usage

    def test_prints_without_error(self):
        print("\n  ▶ TestPrintStats.test_prints_without_error")
        captured = io.StringIO()
        usage = self._make_usage(inp=1000, out=500, cw=200, cr=100)
        print_stats(usage, label="Test", file=captured)
        out = captured.getvalue()
        self.assertIn("1,300", out)  # total_input = 1000 + 200 + 100

    def test_zero_usage_no_crash(self):
        print("\n  ▶ TestPrintStats.test_zero_usage_no_crash")
        captured = io.StringIO()
        usage = self._make_usage()
        print_stats(usage, file=captured)
        self.assertGreater(len(captured.getvalue()), 0)

    def test_savings_shown(self):
        print("\n  ▶ TestPrintStats.test_savings_shown")
        captured = io.StringIO()
        usage = self._make_usage(inp=1_000_000, out=1_000_000, cw=500_000, cr=500_000)
        print_stats(usage, file=captured)
        self.assertIn("saved", captured.getvalue())


class TestPrintSessionSummary(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        self._orig_tokens = dict(cl.session_tokens)
        self._orig_hist = list(cl.conversation_history)

    def tearDown(self):
        import claude_light as cl
        cl.session_tokens.update(self._orig_tokens)
        cl.conversation_history[:] = self._orig_hist

    def test_prints_summary_table(self):
        print("\n  ▶ TestPrintSessionSummary.test_prints_summary_table")
        import claude_light as cl
        cl.session_tokens.update({"input": 1000, "cache_write": 500, "cache_read": 200, "output": 300})
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_session_summary()
        out = captured.getvalue()
        self.assertIn("Session Token Summary", out)
        self.assertIn("TOTAL", out)
        self.assertIn("Cache read", out)

    def test_zero_tokens_no_crash(self):
        print("\n  ▶ TestPrintSessionSummary.test_zero_tokens_no_crash")
        import claude_light as cl
        cl.session_tokens.update({"input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_session_summary()
        self.assertIn("Session Token Summary", captured.getvalue())

    def test_shows_turn_count(self):
        print("\n  ▶ TestPrintSessionSummary.test_shows_turn_count")
        import claude_light as cl
        cl.session_tokens.update({"input": 500, "cache_write": 200, "cache_read": 100, "output": 300})
        cl.conversation_history[:] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_session_summary()
        out = captured.getvalue()
        self.assertIn("Turns:", out)


class TestPrintReply(unittest.TestCase):

    def test_plain_text_fallback(self):
        print("\n  ▶ TestPrintReply.test_plain_text_fallback")
        import claude_light as cl
        orig = cl._RICH_AVAILABLE
        try:
            cl._RICH_AVAILABLE = False
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                _print_reply("hello world")
            self.assertIn("hello world", captured.getvalue())
        finally:
            cl._RICH_AVAILABLE = orig

    def test_rich_path_called(self):
        print("\n  ▶ TestPrintReply.test_rich_path_called")
        import claude_light as cl
        orig = cl._RICH_AVAILABLE
        orig_console = cl.console
        try:
            cl._RICH_AVAILABLE = True
            mock_console = MagicMock()
            cl.console = mock_console
            _print_reply("## Hello")
            mock_console.print.assert_called()
        finally:
            cl._RICH_AVAILABLE = orig
            cl.console = orig_console

    def test_plain_text_with_newlines(self):
        print("\n  ▶ TestPrintReply.test_plain_text_with_newlines")
        import claude_light as cl
        orig = cl._RICH_AVAILABLE
        try:
            cl._RICH_AVAILABLE = False
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                _print_reply("Line 1\nLine 2\nLine 3")
            out = captured.getvalue()
            self.assertIn("Line 1", out)
            self.assertIn("Line 3", out)
        finally:
            cl._RICH_AVAILABLE = orig


class TestSpinner(unittest.TestCase):

    def test_spinner_context_manager(self):
        print("\n  ▶ TestSpinner.test_spinner_context_manager")
        from claude_light.ui import _Spinner
        with patch("builtins.print"):
            with _Spinner("Testing") as sp:
                self.assertIsNotNone(sp)
                sp.update("Updated label")

    def test_spinner_does_not_crash(self):
        print("\n  ▶ TestSpinner.test_spinner_does_not_crash")
        from claude_light.ui import _Spinner
        with patch("builtins.print"), patch("time.sleep"):
            with _Spinner("Working"):
                pass


if __name__ == "__main__":
    unittest.main()
