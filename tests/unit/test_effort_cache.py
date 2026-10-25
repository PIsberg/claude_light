"""Effort routing that keeps the prompt cache.

A top-level `effort` change invalidates the cached conversation (and on some
models the cached tools and system prompt). So every request sends the same
top-level effort, and each turn's routed effort travels as an effort-only
system message just before the question.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light.state as state
from claude_light import llm

HAIKU_LOW = ("low", llm.MODEL_HAIKU, 2_048)
SONNET_MEDIUM = ("medium", llm.MODEL_SONNET, 4_096)
SONNET_HIGH = ("high", llm.MODEL_SONNET, 8_192)


def _usage():
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0
    return usage


def _normalized(messages):
    """Messages as the cache sees them: no breakpoints, str content as a text block."""
    out = []
    for m in messages:
        m = dict(m)
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        m["content"] = [{k: v for k, v in b.items() if k != "cache_control"} for b in content]
        out.append(m)
    return json.dumps(out, sort_keys=True)


def _history_part(messages):
    """Everything before this turn's own messages (its effort message and question)."""
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    if last_user and messages[last_user - 1]["role"] == "system":
        last_user -= 1
    return messages[:last_user]


def _run_turns(routes):
    """Run one chat() turn per (effort, model, max_tokens); return each request's kwargs."""
    sent = []

    def fake_call(**kwargs):
        sent.append(json.loads(json.dumps(kwargs, default=str)))
        return f"answer {len(sent)}", _usage(), False

    queue = list(routes)
    with patch("claude_light.llm.route_query",
               side_effect=lambda q: llm._route_result(*queue.pop(0))), \
         patch("claude_light.llm._make_streaming_api_call", side_effect=fake_call), \
         patch("claude_light.llm.retrieve", return_value=("", [])), \
         patch("claude_light.llm._maybe_compress_history"), \
         patch("claude_light.llm._print_reply"), \
         patch("claude_light.llm.print_stats"):
        for i in range(len(routes)):
            llm.chat(f"question {i + 1}")
    return sent


class TestPerMessageEffort(unittest.TestCase):

    def setUp(self):
        self._saved = (list(state.conversation_history), state.turn_count)
        state.conversation_history[:] = []

    def tearDown(self):
        hist, state.turn_count = self._saved
        state.conversation_history[:] = hist

    def _run_turns(self, routes):
        return _run_turns(routes)

    def test_top_level_effort_is_the_same_on_every_turn(self):
        sent = self._run_turns([SONNET_MEDIUM, SONNET_HIGH, HAIKU_LOW])
        self.assertEqual({r["output_config"]["effort"] for r in sent}, {llm.BASE_EFFORT})

    def test_routed_model_still_changes_per_turn(self):
        # Default. Keeping the first turn's model is opt-in, see TestRouteOnce.
        sent = self._run_turns([SONNET_MEDIUM, HAIKU_LOW])
        self.assertEqual([r["model"] for r in sent], [llm.MODEL_SONNET, llm.MODEL_HAIKU])

    def test_routed_effort_rides_in_an_effort_message(self):
        sent = self._run_turns([SONNET_MEDIUM])
        messages = sent[0]["messages"]
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-2], {"role": "system", "content": [],
                                        "output_config": {"effort": "medium"}})

    def test_no_effort_message_when_the_turn_runs_at_the_base_effort(self):
        sent = self._run_turns([SONNET_HIGH])
        self.assertEqual([m["role"] for m in sent[0]["messages"]], ["user"])

    def test_effort_beta_header_is_sent(self):
        sent = self._run_turns([SONNET_MEDIUM])
        betas = sent[0]["extra_headers"]["anthropic-beta"].split(",")
        self.assertIn(llm.EFFORT_BETA, betas)
        self.assertIn("extended-cache-ttl-2025-04-11", betas)

    def test_history_is_a_stable_prefix_across_effort_changes(self):
        # The effort message must never land inside the cached history.
        sent = self._run_turns([SONNET_MEDIUM, SONNET_HIGH, HAIKU_LOW, SONNET_MEDIUM])
        for earlier, later in zip(sent, sent[1:]):
            before = _history_part(earlier["messages"])
            after = _history_part(later["messages"])[:len(before)]
            self.assertEqual(_normalized(after), _normalized(before))

    def test_only_the_current_turn_has_an_effort_message(self):
        sent = self._run_turns([SONNET_MEDIUM, SONNET_HIGH, HAIKU_LOW])
        roles = [m["role"] for m in sent[-1]["messages"]]
        self.assertEqual(roles, ["user", "assistant"] * 2 + ["system", "user"])
        self.assertEqual([m["role"] for m in state.conversation_history],
                         ["user", "assistant"] * 3)

    def test_oauth_mode_sends_no_effort_messages(self):
        # The CLI path flattens messages into prompt text; an empty system
        # message would turn into a stray "--- System ---" line.
        with patch("claude_light.llm.AUTH_MODE", "OAUTH"), \
             patch("claude_light.llm._git_modified_snapshot", return_value=set()):
            sent = self._run_turns([SONNET_MEDIUM, HAIKU_LOW])
        self.assertNotIn("system", [m["role"] for m in sent[-1]["messages"]])

    def test_one_shot_uses_the_same_scheme(self):
        sent = []

        def fake_call(**kwargs):
            sent.append(kwargs)
            return "answer", _usage(), False

        with patch("claude_light.llm.route_query",
                   side_effect=lambda q: llm._route_result(*HAIKU_LOW)), \
             patch("claude_light.llm._make_streaming_api_call", side_effect=fake_call), \
             patch("claude_light.llm.retrieve", return_value=("", [])), \
             patch("claude_light.llm._print_reply"), \
             patch("claude_light.llm.print_stats"):
            llm._one_shot_turn("what is x", auto_apply=False, use_index=True)
        self.assertEqual(sent[0]["output_config"], {"effort": llm.BASE_EFFORT})
        self.assertEqual(sent[0]["messages"][0],
                         {"role": "system", "content": [], "output_config": {"effort": "low"}})


class TestWarmCacheMatchesTurns(unittest.TestCase):

    def test_warms_at_the_top_level_effort_turns_send(self):
        resp = MagicMock()
        resp.usage = _usage()
        with patch("claude_light.llm.client.messages.create", return_value=resp) as create:
            llm.warm_cache(quiet=True)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["output_config"], {"effort": llm.BASE_EFFORT})
        self.assertIn(llm.EFFORT_BETA, kwargs["extra_headers"]["anthropic-beta"])


class TestRouteOnce(unittest.TestCase):
    """CLAUDE_LIGHT_ROUTE_ONCE=1: the first turn's model runs the whole session."""

    def setUp(self):
        self._saved = (list(state.conversation_history), state.turn_count,
                       state.session_model, state.session_budget, state.session_cost)
        state.conversation_history[:] = []
        state.session_model = None
        state.session_budget = None
        self._flag = patch("claude_light.config.ROUTE_ONCE", True)
        self._flag.start()

    def tearDown(self):
        self._flag.stop()
        (hist, state.turn_count, state.session_model,
         state.session_budget, state.session_cost) = self._saved
        state.conversation_history[:] = hist

    def test_first_turn_on_haiku_keeps_haiku(self):
        sent = _run_turns([HAIKU_LOW, SONNET_MEDIUM, SONNET_HIGH])
        self.assertEqual([r["model"] for r in sent], [llm.MODEL_HAIKU] * 3)

    def test_first_turn_on_sonnet_keeps_sonnet(self):
        sent = _run_turns([SONNET_MEDIUM, HAIKU_LOW])
        self.assertEqual([r["model"] for r in sent], [llm.MODEL_SONNET] * 2)

    def test_effort_is_still_routed_per_turn(self):
        sent = _run_turns([HAIKU_LOW, SONNET_MEDIUM])
        self.assertEqual(sent[1]["messages"][-2],
                         {"role": "system", "content": [], "output_config": {"effort": "medium"}})

    def test_budget_cap_still_drops_to_haiku(self):
        state.session_budget = 1.0
        sent = _run_turns([SONNET_MEDIUM])
        state.session_cost = 2.0
        sent += _run_turns([SONNET_MEDIUM])
        self.assertEqual([r["model"] for r in sent], [llm.MODEL_SONNET, llm.MODEL_HAIKU])

    def test_warm_cache_warms_the_session_model_once_set(self):
        # Before the first turn the model is unknown, so MODEL is warmed; after
        # it, re-warms and the heartbeat must keep the model the turns read.
        resp = MagicMock()
        resp.usage = _usage()
        with patch("claude_light.llm.client.messages.create", return_value=resp) as create:
            llm.warm_cache(quiet=True)
            state.session_model = llm.MODEL_HAIKU
            llm.warm_cache(quiet=True)
        self.assertEqual([c.kwargs["model"] for c in create.call_args_list],
                         [llm.MODEL, llm.MODEL_HAIKU])


if __name__ == "__main__":
    unittest.main()
