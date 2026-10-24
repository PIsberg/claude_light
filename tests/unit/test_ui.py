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
        self.assertAlmostEqual(calculate_cost(usage), 10.0, places=4)

    def test_input_only(self):
        print("\n  ▶ TestCalculateCost.test_input_only")
        usage = self._make_usage(input_tokens=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 2.0, places=4)

    def test_cache_write(self):
        print("\n  ▶ TestCalculateCost.test_cache_write")
        usage = self._make_usage(cache_creation=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 2.5, places=4)

    def test_cache_read(self):
        print("\n  ▶ TestCalculateCost.test_cache_read")
        usage = self._make_usage(cache_read=1_000_000)
        self.assertAlmostEqual(calculate_cost(usage), 0.10, places=4)

    def test_combined(self):
        print("\n  ▶ TestCalculateCost.test_combined")
        usage = self._make_usage(
            input_tokens=100_000,
            output_tokens=100_000,
            cache_creation=100_000,
            cache_read=100_000,
        )
        expected = (100_000/1_000_000)*2.0 + (100_000/1_000_000)*10.0 + (100_000/1_000_000)*2.5 + (100_000/1_000_000)*0.10
        self.assertAlmostEqual(calculate_cost(usage), expected, places=6)

    def test_no_cache_attrs(self):
        print("\n  ▶ TestCalculateCost.test_no_cache_attrs")
        usage = MagicMock(spec=["input_tokens", "output_tokens"])
        usage.input_tokens = 500_000
        usage.output_tokens = 500_000
        cost = calculate_cost(usage)
        self.assertAlmostEqual(cost, 1.0 + 5.0, places=4)


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


class TestSessionFooter(unittest.TestCase):
    """Exit used to print two boxed tables every time; now one line, with
    the tables on /cost."""

    def setUp(self):
        import claude_light.state as st
        self._tokens = dict(st.session_tokens)
        self._costs = dict(st.session_costs)
        self._turns = st.turn_count

    def tearDown(self):
        import claude_light.state as st
        st.session_tokens.update(self._tokens)
        st.session_costs.update(self._costs)
        st.turn_count = self._turns

    def _footer(self, without_cache, actual):
        import claude_light.state as st
        from claude_light.ui import print_session_footer
        st.session_tokens.update({"input": 10, "cache_write": 600, "cache_read": 400, "output": 50})
        st.session_costs.update({"input": 0.0, "cache_write": actual, "cache_read": 0.0,
                                 "output": 0.0, "without_cache": without_cache})
        st.turn_count = 7
        out = io.StringIO()
        with patch("sys.stdout", out):
            print_session_footer()
        return out.getvalue()

    def test_footer_is_one_line_with_turns_tokens_and_cost(self):
        print("\n  ▶ TestSessionFooter.test_footer_is_one_line_with_turns_tokens_and_cost")
        out = self._footer(without_cache=0.02, actual=0.01)
        self.assertEqual(len([l for l in out.splitlines() if l.strip()]), 1)
        self.assertIn("7 turns", out)
        self.assertIn("1,060", out)
        self.assertIn("0.0100", out)
        self.assertIn("saved 50.0%", out)

    def test_a_loss_is_not_called_saved(self):
        print("\n  ▶ TestSessionFooter.test_a_loss_is_not_called_saved")
        # Live, /cost printed "Saved: -5.9%" in green after a cache-write turn.
        out = self._footer(without_cache=0.01, actual=0.0106)
        self.assertNotIn("saved", out)
        self.assertIn("overhead 6.0%", out)

    def test_nothing_to_report_prints_nothing(self):
        print("\n  ▶ TestSessionFooter.test_nothing_to_report_prints_nothing")
        import claude_light.state as st
        from claude_light.ui import print_session_footer
        for k in st.session_tokens:
            st.session_tokens[k] = 0
        out = io.StringIO()
        with patch("sys.stdout", out):
            print_session_footer()
        self.assertEqual(out.getvalue(), "")

    def test_summary_reports_turns_from_the_counter_not_the_history(self):
        print("\n  ▶ TestSessionFooter.test_summary_reports_turns_from_the_counter_not_the_history")
        # History compression shrinks the stored history; 7 turns are still 7.
        import claude_light.state as st
        st.turn_count = 7
        out = io.StringIO()
        with patch("sys.stdout", out):
            print_session_summary()
        self.assertRegex(out.getvalue(), r"Turns: \S*7")

    def test_summary_shows_a_loss_as_overhead(self):
        print("\n  ▶ TestSessionFooter.test_summary_shows_a_loss_as_overhead")
        import claude_light.state as st
        st.session_tokens.update({"input": 10, "cache_write": 600, "cache_read": 0, "output": 50})
        st.session_costs.update({"input": 0.0, "cache_write": 0.0106, "cache_read": 0.0,
                                 "output": 0.0, "without_cache": 0.01})
        out = io.StringIO()
        with patch("sys.stdout", out):
            print_session_summary()
        self.assertNotIn("Saved: \x1b[32m-", out.getvalue())
        self.assertIn("Overhead", out.getvalue())


class TestDiffDisplay(unittest.TestCase):

    def _lines(self, fn):
        import re
        import claude_light.ui as ui
        out = io.StringIO()
        with patch("sys.stdout", out):
            fn(ui)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
        return [l.strip(" │|") for l in plain.splitlines()]

    def test_edit_diff_is_not_double_spaced(self):
        print("\n  ▶ TestDiffDisplay.test_edit_diff_is_not_double_spaced")
        # The lines kept their newline and were joined with another one, so
        # every diff line was followed by a blank line.
        for rich_available in (True, False):
            with patch("claude_light.ui._RICH_AVAILABLE", rich_available):
                lines = self._lines(lambda ui: ui.show_diff("x.py", "a\nB\nc\n", old_content="a\nb\nc\n"))
            i = next(i for i, l in enumerate(lines) if l.strip() == "-b")
            self.assertEqual(lines[i + 1].strip(), "+B", (rich_available, lines))

    def test_path_markup_is_shown_literally(self):
        print("\n  ▶ TestDiffDisplay.test_path_markup_is_shown_literally")
        # The path comes from the model's edit block; rich markup in it
        # could restyle or disguise the title of the diff being approved.
        evil = "x[/cyan][green]safe.py"
        with patch("claude_light.ui._RICH_AVAILABLE", True):
            text = "\n".join(self._lines(lambda ui: ui.show_unified_diff(evil, "@@ -1 +1 @@\n-a\n+b")))
            new = "\n".join(self._lines(lambda ui: ui.show_diff(evil, "b\n", old_content="")))
        self.assertIn("[/cyan][green]", text)
        self.assertIn("[/cyan][green]", new)

    def test_long_diff_is_capped_with_a_count(self):
        print("\n  ▶ TestDiffDisplay.test_long_diff_is_capped_with_a_count")
        diff = "@@ -1,200 +1,200 @@\n" + "\n".join(f"+line{i}" for i in range(200))
        text = "\n".join(self._lines(lambda ui: ui.show_unified_diff("big.py", diff, max_lines=50)))
        self.assertIn("+line48", text)
        self.assertNotIn("+line60", text)
        self.assertIn("151 more lines", text)


class TestAccessLabel(unittest.TestCase):
    """The welcome box shows where access comes from through a fixed table,
    never the resolver's own string: it is returned in the same tuple as the
    key, and CodeQL flagged printing it as clear-text logging (alerts #22,
    #24, #28, #34, #35)."""

    def test_known_sources_get_readable_labels(self):
        print("\n  ▶ TestAccessLabel.test_known_sources_get_readable_labels")
        from claude_light.ui import access_label
        self.assertEqual(access_label("Claude CLI (Pro Subscription)"), "Claude subscription")
        self.assertEqual(access_label("Environment"), "API key from ANTHROPIC_API_KEY")
        self.assertEqual(access_label("Dotfile (.env)"), "API key from .env")

    def test_anything_else_is_not_echoed(self):
        print("\n  ▶ TestAccessLabel.test_anything_else_is_not_echoed")
        from claude_light.ui import access_label
        self.assertEqual(access_label("sk-ant-api03-secret"), "")


class TestChoose(unittest.TestCase):
    """The approval menu is drawn inline under the diff it asks about. The
    old prompt was a full-screen dialog that covered that diff, or a bare
    y/n input()."""

    _OPTS = [("yes", "Yes"), ("always", "Yes, and don't ask again"), ("no", "No")]

    def _pick(self, keys):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from claude_light.ui import choose
        with create_pipe_input() as inp, patch("sys.stdout", io.StringIO()):
            inp.send_text(keys)
            return choose("Apply 1 change?", self._OPTS, _input=inp, _output=DummyOutput())

    def test_enter_takes_the_default(self):
        print("\n  ▶ TestChoose.test_enter_takes_the_default")
        self.assertEqual(self._pick("\r"), "yes")

    def test_arrow_keys_move_the_selection(self):
        print("\n  ▶ TestChoose.test_arrow_keys_move_the_selection")
        self.assertEqual(self._pick("\x1b[B\x1b[B\r"), "no")
        self.assertEqual(self._pick("\x1b[B\x1b[B\x1b[A\r"), "always")

    def test_a_number_picks_at_once_but_a_letter_does_not(self):
        print("\n  ▶ TestChoose.test_a_number_picks_at_once_but_a_letter_does_not")
        self.assertEqual(self._pick("2"), "always")
        # Single letters are what a user types into the next prompt while a
        # turn runs; "a" must not grant "don't ask again".
        self.assertEqual(self._pick("a\x1b[B\x1b[B\r"), "no")

    def test_keys_typed_before_the_menu_are_discarded(self):
        print("\n  ▶ TestChoose.test_keys_typed_before_the_menu_are_discarded")
        from claude_light import ui
        order = []
        with patch.object(ui, "_discard_typeahead", side_effect=lambda: order.append("flush")), \
             patch.object(ui, "_choose_menu", side_effect=lambda *a: order.append("menu") or "no"), \
             patch("sys.stdout", io.StringIO()):
            ui.choose("Apply?", self._OPTS)
        self.assertEqual(order, ["flush", "menu"])

    def test_discard_typeahead_never_raises(self):
        print("\n  ▶ TestChoose.test_discard_typeahead_never_raises")
        from claude_light import ui
        ui._discard_typeahead()  # stdin is not a console under pytest

    def test_ctrl_c_cancels(self):
        print("\n  ▶ TestChoose.test_ctrl_c_cancels")
        self.assertIsNone(self._pick("\x03"))

    def test_typed_fallback_without_a_terminal(self):
        print("\n  ▶ TestChoose.test_typed_fallback_without_a_terminal")
        from claude_light.ui import choose
        for typed, expected in (("y", "yes"), ("3", "no"), ("always", "always"), ("", None), ("zzz", None)):
            with patch("builtins.input", return_value=typed), patch("sys.stdout", io.StringIO()):
                self.assertEqual(choose("Apply?", self._OPTS, _force_plain=True), expected, typed)

    def test_the_choice_stays_on_screen(self):
        print("\n  ▶ TestChoose.test_the_choice_stays_on_screen")
        # The menu erases itself; one line records what was picked.
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from claude_light.ui import choose
        out = io.StringIO()
        with create_pipe_input() as inp, patch("sys.stdout", out):
            inp.send_text("3")
            choose("Apply 1 change?", self._OPTS, _input=inp, _output=DummyOutput())
        self.assertIn("Apply 1 change?", out.getvalue())
        self.assertIn("No", out.getvalue())


class TestStartupHeader(unittest.TestCase):

    def test_header_names_the_model_access_and_help(self):
        print("\n  ▶ TestStartupHeader.test_header_names_the_model_access_and_help")
        # The model line and help hint used to appear only after the first
        # query; they belong on the welcome screen.
        import claude_light.ui as ui
        for rich_available in (True, False):
            out = io.StringIO()
            with patch.object(ui, "_RICH_AVAILABLE", rich_available), patch("sys.stdout", out):
                ui.print_startup_header("claude-sonnet-5-5", "Claude subscription")
            text = out.getvalue()
            self.assertIn("claude light", text)
            self.assertIn("sonnet-5-5", text)
            self.assertIn("Claude subscription", text)
            self.assertIn("/help", text)


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



class TestPerModelPricing(unittest.TestCase):
    """Costs are priced at the model that ran. Rates from
    platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-24).
    One Sonnet 4.6 rate for every model overstated a Haiku 5.5 turn 30x."""

    def _usage(self, inp=0, out=0, cw=0, cr=0, cw_1h=None, cost_usd=None):
        usage = MagicMock(spec=["input_tokens", "output_tokens",
                                "cache_creation_input_tokens", "cache_read_input_tokens",
                                "cache_creation", "cost_usd"])
        usage.input_tokens = inp
        usage.output_tokens = out
        usage.cache_creation_input_tokens = cw
        usage.cache_read_input_tokens = cr
        usage.cache_creation = (None if cw_1h is None
                                else {"ephemeral_1h_input_tokens": cw_1h,
                                      "ephemeral_5m_input_tokens": cw - cw_1h})
        usage.cost_usd = cost_usd
        return usage

    def test_each_model_has_its_own_rates(self):
        print("\n  ▶ TestPerModelPricing.test_each_model_has_its_own_rates")
        from claude_light.config import MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS
        m = 1_000_000
        self.assertAlmostEqual(calculate_cost(self._usage(inp=m), MODEL_HAIKU), 0.10)
        self.assertAlmostEqual(calculate_cost(self._usage(out=m), MODEL_HAIKU), 0.50)
        self.assertAlmostEqual(calculate_cost(self._usage(inp=m), MODEL_SONNET), 2.00)
        self.assertAlmostEqual(calculate_cost(self._usage(cr=m), MODEL_SONNET), 0.10)
        self.assertAlmostEqual(calculate_cost(self._usage(out=m), MODEL_OPUS), 20.00)
        self.assertAlmostEqual(calculate_cost(self._usage(cr=m), MODEL_OPUS), 0.20)

    def test_one_hour_cache_writes_cost_double_input(self):
        print("\n  ▶ TestPerModelPricing.test_one_hour_cache_writes_cost_double_input")
        from claude_light.config import MODEL_SONNET
        m = 1_000_000
        # The skeleton is cached with ttl=1h: those writes cost 2x input, not 1.25x.
        self.assertAlmostEqual(calculate_cost(self._usage(cw=m, cw_1h=m), MODEL_SONNET), 4.00)
        self.assertAlmostEqual(calculate_cost(self._usage(cw=m, cw_1h=0), MODEL_SONNET), 2.50)
        self.assertAlmostEqual(calculate_cost(self._usage(cw=m), MODEL_SONNET), 2.50)

    def test_cost_reported_by_the_cli_wins(self):
        print("\n  ▶ TestPerModelPricing.test_cost_reported_by_the_cli_wins")
        from claude_light.config import MODEL_SONNET
        # The CLI may run several models in one turn and reports the exact total.
        usage = self._usage(inp=1000, out=1000, cost_usd=0.0723477)
        self.assertAlmostEqual(calculate_cost(usage, MODEL_SONNET), 0.0723477)

    def test_unknown_model_falls_back_to_default_rates(self):
        print("\n  ▶ TestPerModelPricing.test_unknown_model_falls_back_to_default_rates")
        from claude_light.config import MODEL
        u = self._usage(inp=1_000_000)
        self.assertAlmostEqual(calculate_cost(u, "claude-something-new"), calculate_cost(u, MODEL))

    def test_session_summary_adds_turns_at_their_own_rates(self):
        print("\n  ▶ TestPerModelPricing.test_session_summary_adds_turns_at_their_own_rates")
        import re
        import claude_light.state as state
        from claude_light import llm
        from claude_light.config import MODEL_HAIKU, MODEL_OPUS
        saved_tokens = dict(state.session_tokens)
        saved_costs = dict(getattr(state, "session_costs", {}))
        try:
            for k in state.session_tokens:
                state.session_tokens[k] = 0
            for k in getattr(state, "session_costs", {}):
                state.session_costs[k] = 0.0
            with patch.object(state, "save_global_stats"):
                llm._accumulate_usage(self._usage(out=1_000_000), MODEL_HAIKU)   # $0.50
                llm._accumulate_usage(self._usage(out=1_000_000), MODEL_OPUS)    # $20.00
            out = io.StringIO()
            with patch("sys.stdout", out):
                print_session_summary()
            total_line = next(l for l in out.getvalue().splitlines() if "TOTAL" in l)
            self.assertIn("$20.5000", total_line)
        finally:
            state.session_tokens.update(saved_tokens)
            if hasattr(state, "session_costs"):
                state.session_costs.update(saved_costs)



def _live_console(width=60, height=24):
    """A rich Console that behaves like a colour terminal but writes to a buffer."""
    from rich.console import Console
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, color_system=None,
                   width=width, height=height, legacy_windows=False), buf


class TestStreamedMarkdownSplit(unittest.TestCase):
    """A streamed reply is printed block by block: a block is final once a
    blank line follows it outside a code fence."""

    def test_paragraph_is_committed_once_a_blank_line_follows(self):
        print("\n  ▶ TestStreamedMarkdownSplit.test_paragraph_is_committed_once_a_blank_line_follows")
        from claude_light.ui import _split_committable
        done, rest = _split_committable("First paragraph.\n\nSecond is still arr")
        self.assertEqual(done, "First paragraph.\n\n")
        self.assertEqual(rest, "Second is still arr")

    def test_nothing_is_committed_inside_an_open_code_fence(self):
        print("\n  ▶ TestStreamedMarkdownSplit.test_nothing_is_committed_inside_an_open_code_fence")
        from claude_light.ui import _split_committable
        text = "```python\nx = 1\n\ny = 2\n"
        self.assertEqual(_split_committable(text), ("", text))

    def test_a_closed_fence_is_committed(self):
        print("\n  ▶ TestStreamedMarkdownSplit.test_a_closed_fence_is_committed")
        from claude_light.ui import _split_committable
        done, rest = _split_committable("```\na\n\nb\n```\nAfter")
        self.assertEqual(done, "```\na\n\nb\n```\n")
        self.assertEqual(rest, "After")

    def test_table_rows_stay_together(self):
        print("\n  ▶ TestStreamedMarkdownSplit.test_table_rows_stay_together")
        from claude_light.ui import _split_committable
        text = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        self.assertEqual(_split_committable(text), ("", text))


class TestResponseViewTerminal(unittest.TestCase):
    """In a terminal the reply renders as markdown. Live, the CLI path printed
    the raw text: tables showed as pipes and **bold** as asterisks."""

    def _render(self, chunks, tools_after=None):
        from claude_light.ui import ResponseView
        con, buf = _live_console()
        with ResponseView(console=con, animate=False) as view:
            for i, chunk in enumerate(chunks):
                view.text(chunk)
                if tools_after and i in tools_after:
                    view.tool(tools_after[i])
        return buf.getvalue()  # animate=False: no redrawn frames to strip

    def test_markdown_is_rendered_not_echoed(self):
        print("\n  ▶ TestResponseViewTerminal.test_markdown_is_rendered_not_echoed")
        out = self._render(["Some **bold** and `code`.\n\n",
                            "| Module | Lines |\n|---|---|\n| llm.py | 1617 |\n"])
        self.assertIn("bold", out)
        self.assertNotIn("**bold**", out)
        self.assertNotIn("|---|", out)
        self.assertIn("llm.py", out)

    def test_reply_is_complete_when_streamed_in_tiny_pieces(self):
        print("\n  ▶ TestResponseViewTerminal.test_reply_is_complete_when_streamed_in_tiny_pieces")
        text = "Alpha paragraph.\n\nBeta paragraph.\n\n```py\nx = 1\n```\nGamma."
        out = self._render(list(text))
        for word in ("Alpha", "Beta", "x = 1", "Gamma"):
            self.assertEqual(out.count(word), 1, word)

    def test_h1_heading_is_left_aligned(self):
        print("\n  ▶ TestResponseViewTerminal.test_h1_heading_is_left_aligned")
        out = self._render(["# Title\n\nBody."])
        line = next(l for l in out.splitlines() if "Title" in l)
        self.assertLess(line.index("Title"), 6)

    def test_tool_call_gets_its_own_line_between_text_segments(self):
        print("\n  ▶ TestResponseViewTerminal.test_tool_call_gets_its_own_line_between_text_segments")
        out = self._render(["Let me look.", "Found it."], tools_after={0: "Read claude_light/llm.py"})
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        i_look = next(i for i, l in enumerate(lines) if "Let me look." in l)
        i_tool = next(i for i, l in enumerate(lines) if "Read claude_light/llm.py" in l)
        i_found = next(i for i, l in enumerate(lines) if "Found it." in l)
        self.assertLess(i_look, i_tool)
        self.assertLess(i_tool, i_found)
        self.assertNotIn("Let me look.", lines[i_tool])

    def test_edit_blocks_become_one_line(self):
        print("\n  ▶ TestResponseViewTerminal.test_edit_blocks_become_one_line")
        # Live, the raw SEARCH/REPLACE block filled the screen and the diff
        # below it showed the same change again.
        out = self._render([
            "I'll add it.\n\n",
            "```python:calc.py\n<<<<<<< SEARCH\nold = 1\n=======\nnew = 2\n>>>>>>> REPLACE\n```\n",
            "```python:pkg/new_mod.py\nx = 1\n```\n",
            "\nDone.",
        ])
        self.assertNotIn("<<<<<<<", out)
        self.assertNotIn("old = 1", out)
        self.assertNotIn("x = 1", out)
        self.assertIn("calc.py", out)
        self.assertIn("pkg/new_mod.py", out)
        self.assertIn("Done.", out)

    def test_unfinished_edit_block_shows_as_writing(self):
        print("\n  ▶ TestResponseViewTerminal.test_unfinished_edit_block_shows_as_writing")
        from claude_light.ui import ResponseView
        con, buf = _live_console()
        view = ResponseView(console=con, animate=False)
        view.text("```python:calc.py\n<<<<<<< SEARCH\nold = 1\n")
        con.print(view)  # what the live region draws for the tail
        out = buf.getvalue()
        self.assertIn("writing", out)
        self.assertIn("calc.py", out)
        self.assertNotIn("<<<<<<<", out)

    def test_plain_code_blocks_still_render(self):
        print("\n  ▶ TestResponseViewTerminal.test_plain_code_blocks_still_render")
        out = self._render(["Example:\n\n```python\nvalue = 42\n```\n"])
        self.assertIn("value = 42", out)

    def test_tail_taller_than_the_screen_is_cropped_to_its_last_lines(self):
        print("\n  ▶ TestResponseViewTerminal.test_tail_taller_than_the_screen_is_cropped_to_its_last_lines")
        from claude_light.ui import _Gutter, _reply_markdown
        con, buf = _live_console(height=10)
        body = "\n".join(f"line{i}" for i in range(40))
        con.print(_Gutter(_reply_markdown(f"```\n{body}\n```"), max_lines=5))
        out = buf.getvalue()
        self.assertIn("line39", out)
        self.assertNotIn("line10", out)


class TestResponseViewPiped(unittest.TestCase):
    """With stdout redirected, the reply goes to it as plain text and every
    progress mark goes to stderr, so `claude_light "q" > answer.md` holds
    only the answer. Live, the file started with \\r spinner frames."""

    def test_piped_stdout_gets_only_the_reply(self):
        print("\n  ▶ TestResponseViewPiped.test_piped_stdout_gets_only_the_reply")
        from claude_light.ui import ResponseView
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdout", out), patch("sys.stderr", err):
            with ResponseView() as view:
                view.text("Hello **world**")
                view.tool("Grep retry")
                view.text("Done.")
        self.assertEqual(out.getvalue(), "Hello **world**\n\nDone.\n")
        self.assertIn("Grep retry", err.getvalue())


class TestResponseViewCleanup(unittest.TestCase):
    """A turn that ends in an error or Ctrl+C must not leave the spinner
    running: it would keep drawing over the next prompt."""

    def _threads(self):
        import threading
        return {t.ident for t in threading.enumerate()}

    def test_live_spinner_stops_when_the_turn_raises(self):
        print("\n  ▶ TestResponseViewCleanup.test_live_spinner_stops_when_the_turn_raises")
        from claude_light.ui import ResponseView
        con, _ = _live_console()
        before = self._threads()
        with self.assertRaises(KeyboardInterrupt):
            with ResponseView(console=con) as view:
                view.text("partial")
                self.assertGreater(len(self._threads() - before), 0)  # refresh thread runs
                raise KeyboardInterrupt
        # rich signals its refresh thread and lets it exit on its next tick
        # (0.1 s); it re-checks the flag under the lock, so it never draws.
        import time
        deadline = time.monotonic() + 2
        while self._threads() - before and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self._threads() - before, set())

    def test_stderr_ticker_stops_when_the_turn_raises(self):
        print("\n  ▶ TestResponseViewCleanup.test_stderr_ticker_stops_when_the_turn_raises")
        from claude_light.ui import ResponseView
        out, err = io.StringIO(), io.StringIO()
        err.isatty = lambda: True
        before = self._threads()
        with patch("sys.stdout", out), patch("sys.stderr", err):
            with self.assertRaises(RuntimeError):
                with ResponseView():
                    self.assertGreater(len(self._threads() - before), 0)
                    raise RuntimeError("boom")
        self.assertEqual(self._threads() - before, set())
        self.assertIn("Thinking", err.getvalue())
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
