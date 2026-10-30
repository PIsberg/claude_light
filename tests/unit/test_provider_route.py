"""
Tests for the route line, effort and budget under a provider without those
features (specs/001-gemini-codex-providers: T021).

Covers US3 scenarios 1-2, FR-008 (effort, top tier, dollar budget) and
FR-012 (route line shows the provider and "CLI default model"). Texts come
from contracts/provider.md.
"""

import io
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401
from claude_light import config, llm, state
from claude_light import main as main_mod
from claude_light.providers import base

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class BareProvider(base.Provider):
    """No capabilities at all; its tool picks the model."""
    name = "barecli"
    capabilities = frozenset()

    def model_label(self, routed_model):
        return "CLI default model"


@pytest.fixture
def bare(monkeypatch):
    monkeypatch.setattr(state, "provider", BareProvider(), raising=False)
    monkeypatch.setattr(state, "session_model", None)
    monkeypatch.setattr(state, "session_budget", None)
    monkeypatch.setattr(state, "budget_asked", False, raising=False)
    monkeypatch.setattr(config, "ROUTE_ONCE", True)
    choose = MagicMock(return_value="opus")
    monkeypatch.setattr(llm, "choose", choose)
    return choose


def _finalize(model, effort="medium", max_tokens=4096):
    err = io.StringIO()
    with redirect_stderr(err), redirect_stdout(io.StringIO()):
        result = llm._finalize_route(model, effort, max_tokens)
    return result, _ANSI.sub("", err.getvalue())


def test_fr012_us3_s1_route_line_names_provider_and_cli_default_model(bare):
    (model, _, _), line = _finalize(llm.MODEL_SONNET)
    assert model == "CLI default model"
    assert "◆  barecli  ·  CLI default model  ·  effort unavailable for barecli" in line
    assert "top tier unavailable" not in line


def test_fr008_us3_s2_top_tier_route_says_unavailable_and_never_prompts(bare):
    (model, _, _), line = _finalize(llm.MODEL_OPUS, "max")
    bare.assert_not_called()
    assert model == "CLI default model"
    assert ("◆  barecli  ·  CLI default model  ·  effort unavailable for barecli"
            "  ·  top tier unavailable for barecli") in line


def test_fr008_top_tier_route_does_not_pin_session_model(bare):
    _finalize(llm.MODEL_OPUS, "max")
    assert state.session_model is None
    _finalize(llm.MODEL_SONNET)
    assert state.session_model is None


def test_fr008_apply_effort_adds_no_system_message_without_effort_capability(bare, monkeypatch):
    monkeypatch.setattr(llm, "AUTH_MODE", "API_KEY")
    kwargs = {"max_tokens": 1000,
              "messages": [{"role": "user", "content": "earlier"},
                           {"role": "user", "content": "question"}]}
    non_base = "max" if llm.BASE_EFFORT != "max" else "low"
    llm._apply_effort(kwargs, non_base)
    assert [m["role"] for m in kwargs["messages"]] == ["user", "user"]


@pytest.mark.parametrize("arg", ["", "2.50", "$3", "off", "0", "junk"])
def test_fr008_budget_command_says_unavailable_and_sets_no_cap(bare, arg):
    msg = main_mod._budget_command(arg)
    assert msg == "Budget unavailable for barecli: it has no dollar pricing. No cap set."
    assert state.session_budget is None
