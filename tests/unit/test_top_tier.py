"""Top-tier confirmation and the router/session split.

route_query() only scores the question. _finalize_route() applies the
session: a question routed to Opus asks whether to run on Opus 5.5, Fable 5.1
or Sonnet 5.5 (Sonnet without a terminal), ROUTE_ONCE keeps the session's
model, and the /budget cap drops to Haiku.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light.config as config
import claude_light.state as state
from claude_light import llm

OPUS_MAX = (llm.MODEL_OPUS, "max", 16_000)
SONNET_HIGH = (llm.MODEL_SONNET, "high", 8_192)
HAIKU_LOW = (llm.MODEL_HAIKU, "low", 2_048)


class _SessionCase(unittest.TestCase):

    def setUp(self):
        self._saved = (state.session_budget, state.session_cost, state.top_tier_declined)
        state.session_budget = None
        state.session_cost = 0.0
        state.top_tier_declined = False
        self._patches = [patch("claude_light.config.ROUTE_ONCE", True),
                         patch("claude_light.llm.sys.stdin")]
        for p in self._patches:
            p.start()
        llm.sys.stdin.isatty.return_value = True

    def tearDown(self):
        for p in self._patches:
            p.stop()
        state.session_budget, state.session_cost, state.top_tier_declined = self._saved

    def _finalize(self, route, answer=None):
        with patch("claude_light.llm.choose", return_value=answer) as choose:
            result = llm._finalize_route(*route)
        return result, choose


class TestRouterIsPure(unittest.TestCase):
    """Benchmarks and /why call route_query(); it must not pin or prompt."""

    def test_route_query_does_not_pin_the_session_model(self):
        with patch("claude_light.config.ROUTE_ONCE", True), \
             patch("claude_light.llm.choose") as choose:
            first, _, _ = llm.route_query("list all files")
            second, _, _ = llm.route_query("refactor the parse method to handle exceptions")
        self.assertEqual((first, second), (llm.MODEL_HAIKU, llm.MODEL_SONNET))
        self.assertIsNone(state.session_model)
        choose.assert_not_called()

    def test_route_query_still_reports_opus_for_the_top_tier(self):
        with patch("claude_light.llm.choose") as choose:
            model, effort, _ = llm.route_query(
                "evaluate the scalability trade-offs of the current microservices architecture deeply")
        self.assertEqual((model, effort), (llm.MODEL_OPUS, "max"))
        choose.assert_not_called()


class TestTopTierChoice(_SessionCase):

    def test_each_answer_picks_its_model(self):
        for answer, model in (("opus", llm.MODEL_OPUS), ("fable", llm.MODEL_FABLE),
                              ("sonnet", llm.MODEL_SONNET)):
            state.session_model = None
            (got, effort, _), choose = self._finalize(OPUS_MAX, answer)
            self.assertEqual((got, effort), (model, "max"), answer)
            choose.assert_called_once()

    def test_cancel_falls_back_to_sonnet(self):
        (model, _, _), _ = self._finalize(OPUS_MAX, None)
        self.assertEqual(model, llm.MODEL_SONNET)

    def test_no_terminal_falls_back_to_sonnet_without_asking(self):
        llm.sys.stdin.isatty.return_value = False
        (model, _, _), choose = self._finalize(OPUS_MAX, "opus")
        self.assertEqual(model, llm.MODEL_SONNET)
        choose.assert_not_called()

    def test_lower_tiers_never_ask(self):
        for route in (SONNET_HIGH, HAIKU_LOW):
            state.session_model = None
            (model, _, _), choose = self._finalize(route, "opus")
            self.assertEqual(model, route[0])
            choose.assert_not_called()

    def test_never_stops_asking_and_uses_sonnet(self):
        self._finalize(OPUS_MAX, "never")
        state.session_model = None
        (model, _, _), choose = self._finalize(OPUS_MAX, "opus")
        self.assertEqual(model, llm.MODEL_SONNET)
        choose.assert_not_called()

    def test_over_budget_drops_to_haiku_without_asking(self):
        state.session_budget, state.session_cost = 1.0, 2.0
        (model, effort, _), choose = self._finalize(OPUS_MAX, "fable")
        self.assertEqual((model, effort), (llm.MODEL_HAIKU, "low"))
        choose.assert_not_called()


class TestTopTierWithRouteOnce(_SessionCase):

    def test_a_later_top_tier_question_can_move_the_session_up(self):
        self._finalize(HAIKU_LOW)
        (model, _, _), choose = self._finalize(OPUS_MAX, "fable")
        self.assertEqual(model, llm.MODEL_FABLE)
        choose.assert_called_once()
        (model, effort, _), _ = self._finalize(HAIKU_LOW)
        self.assertEqual((model, effort), (llm.MODEL_FABLE, "low"))

    def test_declining_moves_the_session_to_sonnet(self):
        self._finalize(HAIKU_LOW)
        self._finalize(OPUS_MAX, "sonnet")
        (model, _, _), _ = self._finalize(HAIKU_LOW)
        self.assertEqual(model, llm.MODEL_SONNET)

    def test_a_session_already_on_a_top_model_is_not_asked_again(self):
        self._finalize(OPUS_MAX, "opus")
        (model, _, _), choose = self._finalize(OPUS_MAX, "fable")
        self.assertEqual(model, llm.MODEL_OPUS)
        choose.assert_not_called()

    def test_per_turn_mode_asks_every_top_tier_question(self):
        with patch("claude_light.config.ROUTE_ONCE", False):
            self._finalize(OPUS_MAX, "fable")
            (model, _, _), choose = self._finalize(OPUS_MAX, "opus")
            (later, _, _), _ = self._finalize(HAIKU_LOW)
        self.assertEqual((model, later), (llm.MODEL_OPUS, llm.MODEL_HAIKU))
        choose.assert_called_once()
        self.assertIsNone(state.session_model)


class TestTopTierReachesTheRequest(_SessionCase):

    def _chat(self, route, answer):
        sent = []

        def fake_call(**kwargs):
            sent.append(json.loads(json.dumps(kwargs, default=str)))
            usage = MagicMock(input_tokens=100, output_tokens=50,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0)
            return "answer", usage, False

        saved = (list(state.conversation_history), state.turn_count)
        state.conversation_history[:] = []
        try:
            with patch("claude_light.llm.route_query", return_value=route), \
                 patch("claude_light.llm.choose", return_value=answer), \
                 patch("claude_light.providers.claude._make_streaming_api_call", side_effect=fake_call), \
                 patch("claude_light.llm.retrieve", return_value=("", [])), \
                 patch("claude_light.llm._maybe_compress_history"), \
                 patch("claude_light.llm._print_reply"), \
                 patch("claude_light.llm.print_stats"):
                llm.chat("question")
        finally:
            state.conversation_history[:], state.turn_count = saved
        return sent[0]

    def test_chat_sends_the_chosen_model(self):
        self.assertEqual(self._chat(OPUS_MAX, "fable")["model"], llm.MODEL_FABLE)

    def test_chat_sends_sonnet_when_declined(self):
        self.assertEqual(self._chat(OPUS_MAX, "sonnet")["model"], llm.MODEL_SONNET)


class TestFablePrices(unittest.TestCase):

    def test_fable_has_its_own_rates(self):
        # claude-api skill, shared/models.md: $10 / $50 per MTok, cache reads
        # $0.25; writes at the standard 1.25x (5 min) and 2x (1 h) of input.
        self.assertEqual(config.model_prices(config.MODEL_FABLE),
                         {"input": 10.00, "write": 12.50, "write_1h": 20.00,
                          "read": 0.25, "output": 50.00})


if __name__ == "__main__":
    unittest.main()
