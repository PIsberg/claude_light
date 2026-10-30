"""
Tests for claude_light.providers.gemini (specs/001-gemini-codex-providers: T012).

Covers US1 scenario 1, US2 scenarios 1-2, US4, FR-007, FR-010, FR-011 and the
missing-tool and not-logged-in edge cases. argv and messages come from
contracts/provider.md. The real agy CLI is never started: run_cli is patched.
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
from claude_light.providers import gemini
from claude_light.providers.base import ProviderError, transcript

_RUN = "claude_light.providers.gemini.run_cli"
_WHICH = "claude_light.providers.gemini.shutil.which"
_AGY = "/bin/agy"
_MISSING = "agy not found. Install the Antigravity CLI: https://antigravity.google"

_SETTINGS = {"permissions": {"allow": [], "deny": ["command", "mcp", "file"]},
             "allowNonWorkspaceAccess": False}
# T027 third pass: without it Gemini calls run_command, is denied, and returns "".
_PREAMBLE = ("You have no tools in this session: do not call any tool, run any command or read "
             "any file. Everything you can use is in this message. For an edit, write "
             "SEARCH/REPLACE blocks from the code shown here.\n\n")
_EXPECTED_ARGV_HEAD = [_AGY, "--input-format", "stream-json", "--output-format", "stream-json",
                       "--mode", "plan", "--settings"]


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


def _ndjson(*events, junk=False):
    lines = []
    for e in events:
        if junk:
            lines.append("agy: warming up")
        lines.append(json.dumps(e))
    return "\n".join(lines) + "\n"


def _result(response="an answer", status="SUCCESS", **extra):
    res = {"status": status, "response": response}
    res.update(extra)
    return {"event": "result", "result": res}


def _stdout(response="an answer"):
    return _ndjson({"event": "init"}, {"event": "step_update", "step": 1}, _result(response))


def _complete(stdout=None, request=None):
    req = request or _request()
    out = _stdout() if stdout is None else stdout
    with patch(_WHICH, return_value=_AGY), patch(_RUN, return_value=out) as run:
        result = gemini.PROVIDER.complete(req)
    return result, run, req


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# --- identity ----------------------------------------------------------------

def test_us1s1_gemini_identity_and_label():
    p = gemini.PROVIDER
    assert gemini.BINARY == "agy"
    assert p.name == "gemini"
    assert set(p.capabilities) == set()
    assert p.model_label("claude-opus-4") == "CLI default model"
    assert p.header() == ("CLI default model", "agy CLI")


# --- complete: argv, stdin, reply (US1 s1, FR-007, FR-010) --------------------

def test_us1s1_complete_returns_the_result_response_with_no_usage_and_not_printed():
    (reply, usage, printed), _, _ = _complete()
    assert reply == "an answer"
    assert usage is None
    assert printed is False


def test_us1s1_non_json_lines_between_events_are_skipped():
    out = _ndjson({"event": "init"}, {"event": "step_update"}, _result("fine"), junk=True)
    (reply, _u, _p), _, _ = _complete(out)
    assert reply == "fine"


def test_us1s1_stdin_is_one_user_event_line_with_the_transcript():
    _, run, req = _complete()
    system_text, prompt = transcript(req)
    stdin = run.call_args[0][2]
    assert stdin.endswith("\n") and stdin.count("\n") == 1
    assert json.loads(stdin) == {"event": "user", "message": {
        "role": "user", "content": _PREAMBLE + system_text + "\n\n" + prompt}}
    assert run.call_args[0][0] == "gemini"


def test_fr010_argv_is_the_contract_argv():
    _, run, _ = _complete()
    argv = run.call_args[0][1]
    assert argv[:-2] == _EXPECTED_ARGV_HEAD
    assert json.loads(argv[-2]) == _SETTINGS
    assert json.loads(gemini.SETTINGS) == _SETTINGS
    assert argv[-1] == "-p="
    assert len(argv) == len(_EXPECTED_ARGV_HEAD) + 2


def test_fr007_argv_has_no_model_flag_and_no_claude_model_id():
    _, run, _ = _complete()
    argv = run.call_args[0][1]
    assert "-m" not in argv and "--model" not in argv
    assert not any(str(a).startswith("claude-") for a in argv)
    assert "claude-sonnet-4-6" not in run.call_args[0][2]


def test_fr011_non_success_result_raises_with_error_and_sign_in_remedy():
    out = _ndjson({"event": "init"}, _result(None, status="ERROR", error="quota exhausted"))
    with pytest.raises(ProviderError) as exc:
        _complete(out)
    assert "quota exhausted" in str(exc.value)
    assert "run agy once and sign in" in str(exc.value)


def test_fr011_no_result_event_raises_provider_error():
    out = _ndjson({"event": "init"}, {"event": "step_update"})
    with pytest.raises(ProviderError) as exc:
        _complete(out)
    assert "output could not be parsed" in str(exc.value)


def test_fr011_empty_stdout_raises_provider_error():
    with pytest.raises(ProviderError) as exc:
        _complete("   \n")
    assert "output could not be parsed" in str(exc.value)


@pytest.mark.parametrize("response", [None, "", "   \n"])
def test_fr011_u2_success_with_missing_or_blank_response_raises(response):
    res = {"status": "SUCCESS"}
    if response is not None:
        res["response"] = response
    out = _ndjson({"event": "init"}, {"event": "result", "result": res})
    with pytest.raises(ProviderError) as exc:
        _complete(out)
    assert "output could not be parsed" in str(exc.value)


def test_fr011_run_cli_error_passes_through_unchanged():
    err = ProviderError("gemini failed (exit 3): boom. Run agy on its own to see the full error.")
    with patch(_WHICH, return_value=_AGY), patch(_RUN, side_effect=err):
        with pytest.raises(ProviderError) as exc:
            gemini.PROVIDER.complete(_request())
    assert exc.value is err


# --- summarize (contract table) ----------------------------------------------

def test_summarize_uses_same_argv_with_summary_prefix_and_no_usage():
    with patch(_WHICH, return_value=_AGY), \
         patch(_RUN, return_value=_stdout("the summary")) as run:
        text, usage = gemini.PROVIDER.summarize("Summarize: X")
    assert text == "the summary"
    assert usage is None
    args, _kw = run.call_args
    assert args[1][:-2] == _EXPECTED_ARGV_HEAD
    assert json.loads(args[1][-2]) == _SETTINGS
    assert args[1][-1] == "-p="
    assert json.loads(args[2]) == {"event": "user", "message": {
        "role": "user", "content": _PREAMBLE + "You summarize conversations.\n\nSummarize: X"}}


# --- require_auth (FR-011, edge cases) ---------------------------------------

def test_edge_missing_tool_exits_1_with_install_message(clean_env, capsys):
    with patch(_WHICH, return_value=None):
        with pytest.raises(SystemExit) as exc:
            gemini.PROVIDER.require_auth()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert _MISSING in out.err + out.out


def test_edge_missing_tool_at_complete_raises_the_same_message():
    with patch(_WHICH, return_value=None), patch(_RUN) as run:
        with pytest.raises(ProviderError) as exc:
            gemini.PROVIDER.complete(_request())
    assert _MISSING in str(exc.value)
    run.assert_not_called()


def test_fr011_agy_on_path_alone_passes_require_auth(clean_env):
    with patch(_WHICH, return_value=_AGY):
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
    with patch(_WHICH, return_value=_AGY), patch(_RUN, return_value=_stdout(_EDIT_REPLY)):
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


# FR-010 / constitution V: the CLI must not run in the user's project, where a
# repo-shipped .gemini/ config could execute code. It runs in a fresh empty temp dir.
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
    out = _stdout("x")
    err = ProviderError("boom") if fail else None
    fake = _capture_cwd_run(seen, error=err, stdout=out)
    with patch(_WHICH, return_value=_AGY), patch(_RUN, side_effect=fake):
        call = (lambda: gemini.PROVIDER.complete(_request())) if method == "complete"             else (lambda: gemini.PROVIDER.summarize("Summarize: X"))
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


# --- no-tools preamble, denied attempts, fail closed (T027 third pass) ---------

def test_t027_complete_content_starts_with_preamble_then_unchanged_transcript():
    _, run, req = _complete()
    system_text, prompt = transcript(req)
    content = json.loads(run.call_args[0][2])["message"]["content"]
    assert content.startswith(_PREAMBLE)
    assert content[len(_PREAMBLE):] == system_text + "\n\n" + prompt


def test_t027_summarize_content_starts_with_preamble():
    with patch(_WHICH, return_value=_AGY), patch(_RUN, return_value=_stdout("s")) as run:
        gemini.PROVIDER.summarize("Summarize: X")
    content = json.loads(run.call_args[0][2])["message"]["content"]
    assert content.startswith(_PREAMBLE)
    assert content[len(_PREAMBLE):] == "You summarize conversations.\n\nSummarize: X"


def _tool_step(tool, state="DONE"):
    # Real agy 1.4.11 shape: everything nests under "step_update".
    return {"event": "step_update",
            "step_update": {"conversation_id": "c1", "step_index": 2, "state": state,
                            "step_type": "tool", "tool_name": tool,
                            "tool_info": {"name": tool, "parameters": {}}}}


def _denied(kind, name):
    return [{"action": kind, "display_name": name}]


def test_t027_denied_attempt_with_empty_response_names_the_action():
    out = _ndjson({"event": "init"}, _tool_step("run_command"),
                  _result("", denied_actions=_denied("command", "RunCommand")))
    with pytest.raises(ProviderError) as exc:
        _complete(out)
    msg = str(exc.value)
    assert "RunCommand" in msg
    assert "denies" in msg
    assert "output could not be parsed" not in msg


_FORBIDDEN = [("run_command", "command"), ("send_command_input", "command"),
              ("call_mcp_tool", "mcp"), ("write_to_file", "file"),
              ("replace_file_content", "file"), ("multi_replace_file_content", "file"),
              ("sed_file", "file"), ("notebook_edit", "file")]


@pytest.mark.parametrize("tool,kind", _FORBIDDEN)
def test_t027_done_forbidden_tool_without_denial_raises_even_with_a_response(tool, kind):
    out = _ndjson({"event": "init"}, _tool_step(tool), _result("a fine answer"))
    with pytest.raises(ProviderError) as exc:
        _complete(out)
    assert f"agy ran {tool}, which claude_light forbids" in str(exc.value)


@pytest.mark.parametrize("tool,kind", _FORBIDDEN)
def test_t027_done_forbidden_tool_with_its_kind_denied_does_not_raise(tool, kind):
    out = _ndjson({"event": "init"}, _tool_step(tool),
                  _result("a fine answer", denied_actions=_denied(kind, "X")))
    (reply, _u, _p), _, _ = _complete(out)
    assert reply == "a fine answer"


@pytest.mark.parametrize("tool", ["search_web", "read_file"])
def test_t027_other_tools_do_not_raise(tool):
    out = _ndjson({"event": "init"}, _tool_step(tool), _result("a fine answer"))
    (reply, _u, _p), _, _ = _complete(out)
    assert reply == "a fine answer"
