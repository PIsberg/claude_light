"""
Tests for claude_light.providers.codex (specs/001-gemini-codex-providers: T015).

Covers US1 scenario 2, US2 scenarios 1-2, US4, FR-007, FR-010, FR-011 and the
missing-tool and not-logged-in edge cases. argv and messages come from
contracts/provider.md; event shapes from research.md R1. The real codex CLI is
never started: run_cli is patched.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401  (loads the package before the submodules)
from claude_light import git_manager, llm, state
from claude_light.providers import codex
from claude_light.providers.base import ProviderError, transcript

_RUN = "claude_light.providers.codex.run_cli"
_WHICH = "claude_light.providers.codex.shutil.which"

_EXPECTED_ARGV_TAIL = [
    "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
    "--sandbox", "read-only",
    "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "apps", "-",
]


def _event(type_, **kw):
    return json.dumps({"type": type_, **kw})


def _msg(text):
    return _event("item.completed", item={"id": "i", "type": "agent_message", "text": text})


def _jsonl(*texts):
    lines = [_event("thread.started", thread_id="t"), _event("turn.started")]
    lines += [_msg(t) for t in texts]
    lines.append(_event("turn.completed", usage={"input_tokens": 1, "output_tokens": 1}))
    return "\n".join(lines) + "\n"


def _request():
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "system": [{"type": "text", "text": "SYS ONE"}, {"type": "text", "text": "SYS TWO"}],
        "messages": [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": [{"type": "text", "text": "the question"}]},
        ],
    }


def _complete(stdout=None, request=None):
    req = request or _request()
    out = _jsonl("an answer") if stdout is None else stdout
    with patch(_WHICH, return_value="/bin/codex"), patch(_RUN, return_value=out) as run:
        result = codex.PROVIDER.complete(req)
    return result, run, req


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# --- identity ----------------------------------------------------------------

def test_us1s2_codex_identity_and_label():
    p = codex.PROVIDER
    assert codex.BINARY == "codex"
    assert p.name == "codex"
    assert set(p.capabilities) == set()
    assert p.model_label("claude-opus-4") == "CLI default model"
    assert p.header() == ("CLI default model", "codex CLI")


# --- complete: argv, stdin, reply (US1 s2, FR-007, FR-010) --------------------

def test_us1s2_complete_returns_agent_message_with_no_usage_and_not_printed():
    (reply, usage, printed), _, _ = _complete(_jsonl("an answer"))
    assert reply == "an answer"
    assert usage is None
    assert printed is False


def test_us1s2_reply_is_the_last_agent_message_after_narration():
    (reply, _u, _p), _, _ = _complete(_jsonl("Let me look.", "Still looking.", "final answer"))
    assert reply == "final answer"


def test_us1s2_non_json_lines_are_skipped():
    out = "warning: something\n" + _jsonl("the answer") + "not json {\n"
    (reply, _u, _p), _, _ = _complete(out)
    assert reply == "the answer"


def test_fr011_output_without_agent_message_raises_provider_error():
    out = "\n".join([_event("thread.started", thread_id="t"),
                     _event("item.completed", item={"type": "reasoning", "text": "hmm"}),
                     _event("turn.completed", usage={})]) + "\n"
    with pytest.raises(ProviderError) as exc:
        _complete(out)
    assert "codex" in str(exc.value)
    assert "output could not be parsed" in str(exc.value)


def test_fr011_empty_stdout_raises_provider_error():
    with pytest.raises(ProviderError) as exc:
        _complete("")
    assert "output could not be parsed" in str(exc.value)


def test_us1s2_complete_stdin_is_the_transcript_joined_by_blank_line():
    _, run, req = _complete()
    system_text, prompt = transcript(req)
    args, _kw = run.call_args
    assert args[0] == "codex"
    assert args[2] == system_text + "\n\n" + prompt


def test_fr010_argv_is_the_contract_argv():
    _, run, _ = _complete()
    argv = run.call_args[0][1]
    assert argv == ["/bin/codex"] + _EXPECTED_ARGV_TAIL


def test_fr007_argv_has_no_model_flag_and_no_claude_model_id():
    _, run, _ = _complete()
    argv = run.call_args[0][1]
    assert "-m" not in argv and "--model" not in argv
    assert not any(str(a).startswith("claude-") for a in argv)
    assert "claude-sonnet-4-6" not in run.call_args[0][2]


def test_fr011_run_cli_error_passes_through_unchanged():
    err = ProviderError("codex failed (exit 1): boom. Run codex on its own to see the full error.")
    with patch(_WHICH, return_value="/bin/codex"), patch(_RUN, side_effect=err):
        with pytest.raises(ProviderError) as exc:
            codex.PROVIDER.complete(_request())
    assert exc.value is err


# --- summarize (contract table) ----------------------------------------------

def test_summarize_uses_same_argv_with_summary_prefix_and_no_usage():
    with patch(_WHICH, return_value="/bin/codex"), \
         patch(_RUN, return_value=_jsonl("the summary")) as run:
        text, usage = codex.PROVIDER.summarize("Summarize: X")
    assert text == "the summary"
    assert usage is None
    args, _kw = run.call_args
    assert args[1] == ["/bin/codex"] + _EXPECTED_ARGV_TAIL
    assert args[2] == "You summarize conversations.\n\nSummarize: X"


# --- require_auth (FR-011, edge cases) ---------------------------------------

def test_edge_missing_tool_exits_1_with_install_message(clean_env, capsys):
    with patch(_WHICH, return_value=None):
        with pytest.raises(SystemExit) as exc:
            codex.PROVIDER.require_auth()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "codex CLI not found. Install it: npm install -g @openai/codex" in out.err + out.out


def test_edge_not_logged_in_exits_1_with_login_message(clean_env, capsys):
    with patch(_WHICH, return_value="/bin/codex"):
        with pytest.raises(SystemExit) as exc:
            codex.PROVIDER.require_auth()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "codex is not logged in. Run: codex login" in out.err + out.out


def test_fr011_default_auth_file_under_home_passes_require_auth(clean_env):
    auth = clean_env / ".codex" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{}", encoding="utf-8")
    with patch(_WHICH, return_value="/bin/codex"):
        codex.PROVIDER.require_auth()


def test_fr011_codex_home_is_honoured(clean_env, monkeypatch):
    other = clean_env / "elsewhere"
    other.mkdir()
    (other / "auth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(other))
    with patch(_WHICH, return_value="/bin/codex"):
        codex.PROVIDER.require_auth()


def test_fr011_codex_home_without_auth_file_is_not_logged_in(clean_env, monkeypatch):
    # The default location has a login, but CODEX_HOME points somewhere without one.
    default = clean_env / ".codex" / "auth.json"
    default.parent.mkdir(parents=True)
    default.write_text("{}", encoding="utf-8")
    empty = clean_env / "empty_home"
    empty.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(empty))
    with patch(_WHICH, return_value="/bin/codex"):
        with pytest.raises(SystemExit) as exc:
            codex.PROVIDER.require_auth()
    assert exc.value.code == 1


# --- US2 / US4: an edit turn through llm.chat --------------------------------

@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run(*a):
        subprocess.run(a, capture_output=True, check=True)

    run("git", "init")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test User")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    run("git", "add", "app.py")
    run("git", "commit", "-m", "Initial commit")
    return tmp_path


@pytest.fixture
def codex_turn(monkeypatch, git_repo):
    monkeypatch.setattr(state, "conversation_history", [])
    monkeypatch.setattr(state, "turn_count", 0)
    monkeypatch.setattr(state, "skeleton_context", "SKELETON")
    monkeypatch.setattr(state, "provider", codex.PROVIDER, raising=False)
    monkeypatch.setattr(llm, "retrieve", lambda *a, **k: ("CHUNK TEXT", []))
    monkeypatch.setattr(llm, "route_query", lambda q: ("some-model", "medium", 4096))
    monkeypatch.setattr(llm, "_finalize_route", lambda m, e, t: (m, e, t))
    monkeypatch.setattr(llm, "record_retrieval", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_maybe_compress_history", lambda: None)
    monkeypatch.setattr(llm, "print_stats", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_print_reply", lambda *a, **k: None)
    monkeypatch.setattr(llm.config, "LLMLINGUA_ENABLED", False)
    return git_repo


_EDIT_REPLY = (
    "Changing x.\n"
    "```python:app.py\n"
    "<<<<<<< SEARCH\n"
    "x = 1\n"
    "=======\n"
    "x = 2\n"
    ">>>>>>> REPLACE\n"
    "```\n"
)


def _edit_turn():
    with patch(_WHICH, return_value="/bin/codex"), \
         patch(_RUN, return_value=_jsonl("Looking.", _EDIT_REPLY)):
        llm.chat("change x to 2", auto_apply=True)


def _head_message(repo):
    return subprocess.run(["git", "log", "-1", "--pretty=%s"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


def test_us2s1_edit_reply_changes_file_and_commits_with_prefix(codex_turn):
    _edit_turn()
    assert (codex_turn / "app.py").read_text(encoding="utf-8").strip() == "x = 2"
    assert _head_message(codex_turn).startswith(git_manager.COMMIT_PREFIX)


def test_us2s2_undo_restores_the_file(codex_turn):
    _edit_turn()
    assert git_manager.undo_last_commit() is True
    assert (codex_turn / "app.py").read_text(encoding="utf-8").strip() == "x = 1"


def test_us4_edit_turn_runs_without_the_claude_and_gemini_modules(codex_turn, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_light.providers.claude", None)
    monkeypatch.setitem(sys.modules, "claude_light.providers.gemini", None)
    _edit_turn()
    assert (codex_turn / "app.py").read_text(encoding="utf-8").strip() == "x = 2"
    assert _head_message(codex_turn).startswith(git_manager.COMMIT_PREFIX)


# FR-010 / constitution V: the CLI must not run in the user's project, where a
# repo-shipped .codex/ config could execute code. It runs in a fresh empty temp dir.
def _capture_cwd_run(seen, error=None, stdout=None):
    import os

    def fake(provider, argv, stdin_text, *a, **kw):
        cwd = kw.get("cwd")
        seen["cwd"] = cwd
        seen["stdin"] = stdin_text
        assert cwd is not None, "run_cli called without cwd"
        seen["exists"] = os.path.isdir(cwd)
        seen["listing"] = os.listdir(cwd)
        if error:
            raise error
        return stdout
    return fake


@pytest.mark.parametrize("method", ["complete", "summarize"])
@pytest.mark.parametrize("fail", [False, True])
def test_fr010_cli_runs_in_fresh_empty_temp_dir_removed_afterwards(method, fail):
    import os
    seen = {}
    out = _jsonl("x")
    err = ProviderError("boom") if fail else None
    fake = _capture_cwd_run(seen, error=err, stdout=out)
    with patch(_WHICH, return_value="/bin/codex"), patch(_RUN, side_effect=fake):
        call = (lambda: codex.PROVIDER.complete(_request())) if method == "complete"             else (lambda: codex.PROVIDER.summarize("Summarize: X"))
        if fail:
            with pytest.raises(ProviderError):
                call()
        else:
            call()
    cwd = seen["cwd"]
    assert seen["exists"] is True
    assert seen["listing"] == []
    assert os.path.realpath(cwd) != os.path.realpath(os.getcwd())
    assert not os.path.exists(cwd)
