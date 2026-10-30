"""
Tests for startup and background work under a provider without caching,
usage or shell (specs/001-gemini-codex-providers: T023).

Covers US1 scenario 4, US2 scenario 4, FR-003 and FR-010. Notice text from
contracts/provider.md.
"""

import io
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401
from claude_light import config, llm, providers, state
from claude_light import main as main_mod
from claude_light.providers import base

NOTICE = ("shell access is disabled for barecli: its agent cannot run commands. "
          "--allow-command only lets claude_light run the affected tests.")


class BareProvider(base.Provider):
    name = "barecli"
    capabilities = frozenset()

    def model_label(self, routed_model):
        return "CLI default model"

    def header(self):
        return ("CLI default model", "barecli CLI")

    def require_auth(self):
        return None

    def complete(self, request):
        return "fake answer", None, False

    def summarize(self, instructions):
        return "a summary", None


@pytest.fixture
def bare(monkeypatch):
    monkeypatch.setattr(state, "provider", BareProvider(), raising=False)
    return state.provider


def test_us2_s4_heartbeat_returns_without_waiting_and_prints_nothing(bare, monkeypatch):
    monkeypatch.setattr(main_mod, "AUTH_MODE", "API_KEY")
    warm = MagicMock()
    monkeypatch.setattr(main_mod, "warm_cache", warm)
    monkeypatch.setattr(main_mod, "HEARTBEAT_SECS", 0.01)
    monkeypatch.setattr(main_mod, "CACHE_TTL_SECS", -1)
    state.stop_event.clear()
    out, err = io.StringIO(), io.StringIO()

    def run():
        with redirect_stdout(out), redirect_stderr(err):
            main_mod.heartbeat()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(1.0)
    alive = t.is_alive()
    state.stop_event.set()
    t.join(2.0)
    state.stop_event.clear()
    assert not alive, "heartbeat kept running without the caching capability"
    warm.assert_not_called()
    assert out.getvalue() + err.getvalue() == ""


def test_us1_s4_index_ready_waits_for_index_even_under_oauth(bare, monkeypatch):
    monkeypatch.setattr(main_mod, "AUTH_MODE", "OAUTH")
    state.indexing_done.clear()
    result = []
    t = threading.Thread(
        target=lambda: result.append(main_mod._index_ready_for_query()), daemon=True)
    with redirect_stdout(io.StringIO()):
        t.start()
        t.join(0.5)
        waiting = t.is_alive()
        state.indexing_done.set()
        t.join(2.0)
    state.indexing_done.clear()
    assert waiting, "returned without waiting for the index"
    assert result == [True]


def test_fr003_one_shot_indexes_before_the_turn_under_oauth(bare, monkeypatch):
    monkeypatch.setattr(llm, "AUTH_MODE", "OAUTH")
    monkeypatch.setattr(llm, "_ONE_SHOT_INDEX_WAIT_SECS", 0.05)
    monkeypatch.setattr(llm, "_update_skeleton", lambda: None)
    done = threading.Event()
    seen = {}

    def slow_index(*a, **k):
        threading.Event().wait(0.4)
        done.set()

    def turn(prompt, auto_apply, use_index):
        seen["index_done_before_turn"] = done.is_set()
        seen["use_index"] = use_index

    monkeypatch.setattr(llm, "index_files", slow_index)
    monkeypatch.setattr(llm, "_one_shot_turn", turn)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        llm.one_shot("question")
    done.wait(2.0)
    assert seen["index_done_before_turn"] is True
    assert seen["use_index"] is True


def test_fr003_chat_takes_no_git_snapshot_under_oauth(bare, monkeypatch):
    monkeypatch.setattr(llm, "AUTH_MODE", "OAUTH")
    snapshot = MagicMock(return_value=set())
    monkeypatch.setattr(llm, "_git_modified_snapshot", snapshot)
    monkeypatch.setattr(state, "conversation_history", [])
    monkeypatch.setattr(state, "turn_count", 0)
    monkeypatch.setattr(state, "skeleton_context", "SKELETON")
    monkeypatch.setattr(state, "save_global_stats", MagicMock())
    monkeypatch.setattr(llm, "retrieve", lambda *a, **k: ("CHUNK", []))
    monkeypatch.setattr(llm, "route_query", lambda q: ("some-model", "medium", 4096))
    monkeypatch.setattr(llm, "_finalize_route", lambda m, e, t: (m, e, t))
    monkeypatch.setattr(llm, "record_retrieval", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_print_reply", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_maybe_compress_history", lambda: None)
    monkeypatch.setattr(config, "LLMLINGUA_ENABLED", False)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        llm.chat("What does OrderService do?")
    snapshot.assert_not_called()


class _Stop(Exception):
    pass


def test_us1_s4_start_chat_passes_provider_header(bare, monkeypatch):
    header = MagicMock(side_effect=_Stop)
    monkeypatch.setattr(main_mod, "print_startup_header", header)
    with pytest.raises(_Stop):
        main_mod.start_chat()
    header.assert_called_once_with("CLI default model", "barecli CLI")


def _run_main(monkeypatch, provider, argv):
    monkeypatch.setattr(sys, "argv", ["claude_light"] + argv)
    monkeypatch.setattr(config, "ALLOWED_COMMANDS", [])
    monkeypatch.setattr(providers, "select", lambda name: provider)
    monkeypatch.setattr(state, "provider", provider, raising=False)
    monkeypatch.setattr(main_mod, "_start_llmlingua_load", lambda: None)
    monkeypatch.setattr(main_mod, "start_chat", lambda **k: None)
    monkeypatch.setattr(main_mod, "one_shot", lambda *a, **k: None)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        main_mod.main()
    return out.getvalue() + err.getvalue()


def test_fr010_allow_command_prints_shell_notice_under_bare_provider(bare, monkeypatch):
    text = _run_main(monkeypatch, bare, ["--allow-command", "python -m pytest", "hello"])
    assert NOTICE in text


def test_fr010_no_notice_without_allow_command(bare, monkeypatch):
    text = _run_main(monkeypatch, bare, ["hello"])
    assert "shell access is disabled" not in text


def test_fr010_no_notice_under_claude(monkeypatch):
    claude = providers.get("claude")
    monkeypatch.setattr(claude, "require_auth", lambda: None)
    text = _run_main(monkeypatch, claude, ["--allow-command", "python -m pytest", "hello"])
    assert "shell access is disabled" not in text
