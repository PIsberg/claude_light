"""
Tests for the provider registry and the provider-neutral turn path
(specs/001-gemini-codex-providers: T002).

Covers FR-001, FR-002, FR-003, FR-005, FR-011, US1 scenarios 3-4, the
unknown-name edge case and US4 (Claude loads without the other providers).
"""

import copy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401  (loads the package before the submodules)
from claude_light import llm, providers, state
from claude_light.providers import base
from claude_light.providers.claude import ClaudeProvider

_NO_USAGE_FIGURES = types.SimpleNamespace(input_tokens=0, output_tokens=0)


class FakeProvider(base.Provider):
    """A provider with no capabilities that records what it is asked."""
    name = "fake"
    capabilities = frozenset()

    def __init__(self, error=None, reply="fake answer"):
        self.error = error
        self.reply = reply
        self.requests = []
        self.summaries = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.error is not None:
            raise self.error
        return self.reply, _NO_USAGE_FIGURES, False

    def summarize(self, instructions):
        self.summaries.append(instructions)
        return "a summary", _NO_USAGE_FIGURES


@pytest.fixture
def clean_state(monkeypatch):
    monkeypatch.setattr(state, "conversation_history", [])
    monkeypatch.setattr(state, "turn_count", 0)
    monkeypatch.setattr(state, "skeleton_context", "SKELETON")
    monkeypatch.setattr(state, "provider", None, raising=False)


@pytest.fixture
def neutral_turn(monkeypatch, clean_state):
    """Pin everything chat() decides before the provider call."""
    monkeypatch.setattr(llm, "retrieve", lambda *a, **k: ("CHUNK TEXT", []))
    monkeypatch.setattr(llm, "route_query", lambda q: ("some-model", "medium", 4096))
    monkeypatch.setattr(llm, "_finalize_route", lambda m, e, t: (m, e, t))
    monkeypatch.setattr(llm, "record_retrieval", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_maybe_compress_history", lambda: None)
    monkeypatch.setattr(llm, "print_stats", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_print_reply", lambda *a, **k: None)
    monkeypatch.setattr(llm.config, "LLMLINGUA_ENABLED", False)


# --- registry (FR-001, FR-005, unknown-name edge case) -----------------------

def test_fr001_names_and_default():
    assert providers.NAMES == ("claude", "gemini", "codex")
    assert providers.DEFAULT == "claude"


def test_fr001_get_claude_returns_its_provider():
    p = providers.get("claude")
    assert p.name == "claude"
    assert isinstance(p, ClaudeProvider)


def test_unknown_provider_name_raises_valueerror_naming_the_valid_names():
    with pytest.raises(ValueError) as exc:
        providers.get("nope")
    for name in providers.NAMES:
        assert name in str(exc.value)


def test_fr001_select_sets_state_provider(monkeypatch):
    monkeypatch.setattr(state, "provider", None, raising=False)
    chosen = providers.select("claude")
    assert state.provider is chosen
    assert providers.active() is chosen


def test_fr002_active_falls_back_to_default_when_nothing_selected(monkeypatch):
    monkeypatch.setattr(state, "provider", None, raising=False)
    assert providers.active().name == providers.DEFAULT


def test_us4_get_claude_works_without_the_other_provider_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_light.providers.gemini", None)
    monkeypatch.setitem(sys.modules, "claude_light.providers.codex", None)
    assert providers.get("claude").name == "claude"


def test_fr005_claude_capabilities_are_the_seven_constants():
    expected = {base.CACHING, base.EFFORT, base.TOP_TIER, base.USD_BUDGET,
                base.USAGE, base.AGENT, base.SHELL}
    p = providers.get("claude")
    assert set(p.capabilities) == expected
    assert all(p.has(c) for c in expected)
    assert not p.has("nonsense")


# --- main --provider (FR-001, FR-002, US1 scenario 3) ------------------------

def _run_main(monkeypatch, argv):
    from claude_light import main as main_mod
    fake = MagicMock()
    select = MagicMock(return_value=fake)
    monkeypatch.setattr(sys, "argv", ["claude_light"] + argv)
    monkeypatch.setattr(providers, "select", select)
    monkeypatch.setattr(main_mod, "one_shot", MagicMock())
    monkeypatch.setattr(main_mod, "_start_llmlingua_load", lambda: None)
    main_mod.main()
    return select, fake


def test_fr001_main_with_provider_flag_selects_it_and_checks_its_auth(monkeypatch):
    select, fake = _run_main(monkeypatch, ["--provider", "gemini", "what is this?"])
    select.assert_called_once_with("gemini")
    fake.require_auth.assert_called_once_with()


def test_fr002_us1s3_main_without_flag_selects_claude(monkeypatch):
    select, fake = _run_main(monkeypatch, ["what is this?"])
    select.assert_called_once_with("claude")
    fake.require_auth.assert_called_once_with()


def test_unknown_provider_flag_exits_2_listing_valid_names(monkeypatch, capsys):
    from claude_light import main as main_mod
    monkeypatch.setattr(sys, "argv", ["claude_light", "--provider", "x", "q"])
    monkeypatch.setattr(main_mod, "_start_llmlingua_load", lambda: None)
    # Nothing may reach a real API if the flag is wrongly accepted.
    monkeypatch.setattr(main_mod, "one_shot", MagicMock())
    monkeypatch.setattr(main_mod, "require_auth", MagicMock(), raising=False)
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'x'" in err
    # argparse quotes the choices on 3.11 and not on 3.12: check each name only.
    for name in providers.NAMES:
        assert name in err


# --- FR-003: same request for every provider ---------------------------------

def _chat_request(monkeypatch, provider):
    monkeypatch.setattr(state, "provider", provider, raising=False)
    state.conversation_history[:] = []
    llm.chat("What does OrderService do?")


def test_fr003_fake_provider_gets_the_request_claude_gets(monkeypatch, neutral_turn):
    seen = {}

    def claude_complete(self, request):
        seen["claude"] = copy.deepcopy(request)
        return "answer", _NO_USAGE_FIGURES, False

    with patch.object(ClaudeProvider, "complete", claude_complete):
        _chat_request(monkeypatch, ClaudeProvider())
    fake = FakeProvider()
    _chat_request(monkeypatch, fake)

    assert "claude" in seen, "Claude's complete() was never reached through llm.chat"
    assert len(fake.requests) == 1
    assert fake.requests[0]["system"] == seen["claude"]["system"]
    assert fake.requests[0]["messages"] == seen["claude"]["messages"]
    assert "CHUNK TEXT" in str(fake.requests[0]["messages"])
    assert "SKELETON" in str(fake.requests[0]["system"])


# --- US1 scenario 4: no Claude request under another provider ----------------

def test_us1s4_chat_makes_no_claude_request(monkeypatch, neutral_turn):
    fake = FakeProvider()
    monkeypatch.setattr(state, "provider", fake, raising=False)
    with patch.object(ClaudeProvider, "complete") as c, \
         patch.object(ClaudeProvider, "summarize") as s:
        llm.chat("What does OrderService do?")
    c.assert_not_called()
    s.assert_not_called()
    assert len(fake.requests) == 1
    assert state.conversation_history[-1] == {"role": "assistant", "content": "fake answer"}


def test_us1s4_one_shot_turn_makes_no_claude_request(monkeypatch, neutral_turn):
    fake = FakeProvider()
    monkeypatch.setattr(state, "provider", fake, raising=False)
    with patch.object(ClaudeProvider, "complete") as c, \
         patch.object(ClaudeProvider, "summarize") as s:
        llm._one_shot_turn("What does OrderService do?", False, True)
    c.assert_not_called()
    s.assert_not_called()
    assert len(fake.requests) == 1


def test_us1s4_compact_history_makes_no_claude_request(monkeypatch, clean_state):
    fake = FakeProvider()
    monkeypatch.setattr(state, "provider", fake, raising=False)
    state.conversation_history[:] = [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    with patch.object(ClaudeProvider, "complete") as c, \
         patch.object(ClaudeProvider, "summarize") as s:
        llm.compact_history()
    c.assert_not_called()
    s.assert_not_called()
    assert len(fake.summaries) == 1


# --- FR-011: a provider error leaves the session and git alone ---------------

def test_fr011_provider_error_leaves_history_turns_and_git_untouched(
        monkeypatch, neutral_turn, capsys):
    fake = FakeProvider(error=base.ProviderError("fake failed (exit 1): boom. Run fake on its own."))
    monkeypatch.setattr(state, "provider", fake, raising=False)
    apply = MagicMock()
    commit = MagicMock()
    monkeypatch.setattr(llm, "apply_edits", apply)
    monkeypatch.setattr(llm, "_commit_agent_edits", commit)

    llm.chat("What does OrderService do?")

    assert len(fake.requests) == 1
    assert state.conversation_history == []
    assert state.turn_count == 0
    apply.assert_not_called()
    commit.assert_not_called()
    assert "fake failed (exit 1)" in capsys.readouterr().out


# --- T017: warm_cache goes through the provider (constitution IV) -------------

def test_iv_warm_cache_under_a_provider_without_warm_is_silent(monkeypatch, clean_state, capsys):
    import claude_light.ui as ui
    monkeypatch.setattr(state, "provider", FakeProvider(), raising=False)
    with patch.object(ui, "notify") as notify, \
         patch.object(ClaudeProvider, "warm") as claude_warm:
        llm.warm_cache()
        llm.warm_cache(quiet=True)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    notify.assert_not_called()
    claude_warm.assert_not_called()


def test_iv_warm_cache_under_claude_calls_claude_warm_once_with_quiet(monkeypatch, clean_state):
    monkeypatch.setattr(state, "provider", ClaudeProvider(), raising=False)
    with patch.object(ClaudeProvider, "warm") as warm:
        llm.warm_cache(quiet=True)
    warm.assert_called_once_with(quiet=True)


def test_fr006_package_client_is_the_claude_provider_client():
    from claude_light.providers import claude as claude_mod
    assert claude_light.client is claude_mod.client


def test_fr006_package_client_stub_reaches_warm_cache(monkeypatch, clean_state):
    from claude_light.providers import claude as claude_mod
    stub = MagicMock()
    monkeypatch.setattr(state, "provider", None, raising=False)
    monkeypatch.setattr(state, "session_cost", 0.0, raising=False)
    monkeypatch.setattr(claude_mod, "AUTH_MODE", "API_KEY")
    monkeypatch.setattr(claude_light, "client", stub)
    try:
        claude_light.warm_cache(quiet=True)
    except Exception:
        pass  # the stub's response is not a real one; only the call matters here
    assert stub.messages.create.called, "warm_cache never reached the stub set on claude_light.client"


def test_us4_import_succeeds_with_the_claude_provider_module_blocked():
    import subprocess
    repo = Path(__file__).resolve().parent.parent.parent
    code = ("import sys; sys.modules['claude_light.providers.claude'] = None; "
            "import claude_light")
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
