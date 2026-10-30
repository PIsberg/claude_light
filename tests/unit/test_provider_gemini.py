"""
Tests for claude_light.providers.gemini (specs/001-gemini-codex-providers: T012).

Covers US1 scenario 1, US2 scenarios 1-2, US4, FR-007, FR-010, FR-011 and the
missing-tool and not-logged-in edge cases. argv and messages come from
contracts/provider.md. The real gemini CLI is never started: run_cli is patched.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401  (loads the package before the submodules)
from claude_light import git_manager, llm, state
from claude_light.providers import gemini
from claude_light.providers.base import ProviderError, transcript

_RUN = "claude_light.providers.gemini.run_cli"
_WHICH = "claude_light.providers.gemini.shutil.which"
_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")

_EXPECTED_ARGV_TAIL = [
    "-p", "Answer the request given on standard input.",
    "-o", "text", "--approval-mode", "plan",
    "--allowed-mcp-server-names", "__claude_light_none__",
]


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


def _complete(stdout="  an answer \n", request=None):
    req = request or _request()
    with patch(_WHICH, return_value="/bin/gemini"), patch(_RUN, return_value=stdout) as run:
        result = gemini.PROVIDER.complete(req)
    return result, run, req


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# --- identity ----------------------------------------------------------------

def test_us1s1_gemini_identity_and_label():
    p = gemini.PROVIDER
    assert gemini.BINARY == "gemini"
    assert p.name == "gemini"
    assert set(p.capabilities) == set()
    assert p.model_label("claude-opus-4") == "CLI default model"
    assert p.header() == ("CLI default model", "gemini CLI")


# --- complete: argv, stdin, reply (US1 s1, FR-007, FR-010) --------------------

def test_us1s1_complete_returns_stripped_stdout_with_no_usage_and_not_printed():
    (reply, usage, printed), _, _ = _complete("  an answer \n")
    assert reply == "an answer"
    assert usage is None
    assert printed is False


def test_us1s1_complete_stdin_is_the_transcript_joined_by_blank_line():
    _, run, req = _complete()
    system_text, prompt = transcript(req)
    args, _kw = run.call_args
    assert args[0] == "gemini"
    assert args[2] == system_text + "\n\n" + prompt


def test_fr010_argv_is_the_contract_argv():
    _, run, _ = _complete()
    argv = run.call_args[0][1]
    assert argv == ["/bin/gemini"] + _EXPECTED_ARGV_TAIL


def test_fr007_argv_has_no_model_flag_and_no_claude_model_id():
    _, run, _ = _complete()
    argv = run.call_args[0][1]
    assert "-m" not in argv and "--model" not in argv
    assert not any(str(a).startswith("claude-") for a in argv)
    assert "claude-sonnet-4-6" not in run.call_args[0][2]


def test_fr011_empty_stdout_raises_provider_error_naming_gemini():
    with pytest.raises(ProviderError) as exc:
        _complete("   \n")
    assert "gemini" in str(exc.value)
    assert "output could not be parsed" in str(exc.value)


def test_fr011_run_cli_error_passes_through_unchanged():
    err = ProviderError("gemini failed (exit 1): boom. Run gemini on its own to see the full error.")
    with patch(_WHICH, return_value="/bin/gemini"), patch(_RUN, side_effect=err):
        with pytest.raises(ProviderError) as exc:
            gemini.PROVIDER.complete(_request())
    assert exc.value is err


# --- summarize (contract table) ----------------------------------------------

def test_summarize_uses_same_argv_with_summary_prefix_and_no_usage():
    with patch(_WHICH, return_value="/bin/gemini"), \
         patch(_RUN, return_value=" the summary \n") as run:
        text, usage = gemini.PROVIDER.summarize("Summarize: X")
    assert text == "the summary"
    assert usage is None
    args, _kw = run.call_args
    assert args[1] == ["/bin/gemini"] + _EXPECTED_ARGV_TAIL
    assert args[2] == "You summarize conversations.\n\nSummarize: X"


# --- require_auth (FR-011, edge cases) ---------------------------------------

def test_edge_missing_tool_exits_1_with_install_message(clean_env, capsys):
    with patch(_WHICH, return_value=None):
        with pytest.raises(SystemExit) as exc:
            gemini.PROVIDER.require_auth()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "gemini CLI not found. Install it: npm install -g @google/gemini-cli" in out.err + out.out


def test_edge_not_logged_in_exits_1_with_login_message(clean_env, capsys):
    with patch(_WHICH, return_value="/bin/gemini"):
        with pytest.raises(SystemExit) as exc:
            gemini.PROVIDER.require_auth()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "gemini is not logged in. Run gemini once and sign in, or set GEMINI_API_KEY." in out.err + out.out


@pytest.mark.parametrize("var", _ENV_VARS)
def test_fr011_each_env_var_alone_passes_require_auth(clean_env, monkeypatch, var):
    monkeypatch.setenv(var, "1")
    with patch(_WHICH, return_value="/bin/gemini"):
        gemini.PROVIDER.require_auth()


def test_fr011_credentials_file_alone_passes_require_auth(clean_env):
    creds = clean_env / ".gemini" / "oauth_creds.json"
    creds.parent.mkdir(parents=True)
    creds.write_text("{}", encoding="utf-8")
    with patch(_WHICH, return_value="/bin/gemini"):
        gemini.PROVIDER.require_auth()


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
def gemini_turn(monkeypatch, git_repo):
    monkeypatch.setattr(state, "conversation_history", [])
    monkeypatch.setattr(state, "turn_count", 0)
    monkeypatch.setattr(state, "skeleton_context", "SKELETON")
    monkeypatch.setattr(state, "provider", gemini.PROVIDER, raising=False)
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
    with patch(_WHICH, return_value="/bin/gemini"), patch(_RUN, return_value=_EDIT_REPLY):
        llm.chat("change x to 2", auto_apply=True)


def _head_message(repo):
    return subprocess.run(["git", "log", "-1", "--pretty=%s"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()


def test_us2s1_edit_reply_changes_file_and_commits_with_prefix(gemini_turn):
    _edit_turn()
    assert (gemini_turn / "app.py").read_text(encoding="utf-8").strip() == "x = 2"
    assert _head_message(gemini_turn).startswith(git_manager.COMMIT_PREFIX)


def test_us2s2_undo_restores_the_file(gemini_turn):
    _edit_turn()
    assert git_manager.undo_last_commit() is True
    assert (gemini_turn / "app.py").read_text(encoding="utf-8").strip() == "x = 1"


def test_us4_edit_turn_runs_without_the_claude_and_codex_modules(gemini_turn, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_light.providers.claude", None)
    monkeypatch.setitem(sys.modules, "claude_light.providers.codex", None)
    _edit_turn()
    assert (gemini_turn / "app.py").read_text(encoding="utf-8").strip() == "x = 2"
    assert _head_message(gemini_turn).startswith(git_manager.COMMIT_PREFIX)
