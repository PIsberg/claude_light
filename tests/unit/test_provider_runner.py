"""
Tests for claude_light.providers.runner (specs/001-gemini-codex-providers: T005).

Covers FR-011 and the "tool fails or times out" edge case. Messages are the
ones in contracts/provider.md. subprocess.Popen is always faked: no real CLI runs.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.providers import base, runner

_POPEN = "claude_light.providers.runner.subprocess.Popen"
_KILL = "claude_light.providers.runner.kill_process_tree"


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0, communicate_error=None):
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.returncode = returncode
        self.communicate_error = communicate_error
        self.communicate_calls = []
        self.pid = 4242

    def communicate(self, input=None, timeout=None):
        self.communicate_calls.append((input, timeout))
        if self.communicate_error is not None:
            raise self.communicate_error
        return self.stdout_text, self.stderr_text

    def poll(self):
        return None

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def _run(proc, **kwargs):
    with patch(_POPEN, return_value=proc) as popen, patch(_KILL) as kill:
        try:
            out = runner.run_cli("gemini", ["gemini", "-p", "x"], "STDIN TEXT", **kwargs)
            error = None
        except BaseException as e:  # noqa: BLE001  (KeyboardInterrupt is a case)
            out, error = None, e
    return out, error, popen, kill


def test_fr011_returns_stdout_of_a_clean_run():
    proc = FakeProc(stdout="the answer\n")
    out, error, popen, _ = _run(proc)
    assert error is None
    assert out == "the answer\n"


def test_fr011_stdin_text_reaches_communicate_unchanged_with_default_timeout():
    proc = FakeProc(stdout="ok")
    _run(proc)
    assert proc.communicate_calls == [("STDIN TEXT", runner.TIMEOUT_SECS)]
    assert runner.TIMEOUT_SECS == 600


def test_fr011_popen_gets_pipes_text_mode_and_the_argv():
    proc = FakeProc(stdout="ok")
    _, _, popen, _ = _run(proc)
    args, kwargs = popen.call_args
    assert args[0] == ["gemini", "-p", "x"]
    assert kwargs["stdin"] == subprocess.PIPE
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_fr011_explicit_timeout_is_passed_to_communicate():
    proc = FakeProc(stdout="ok")
    _run(proc, timeout=7)
    assert proc.communicate_calls == [("STDIN TEXT", 7)]


def test_fr011_nonzero_exit_names_provider_exit_code_and_last_stderr_line():
    proc = FakeProc(stderr="warming up\nsome noise\n\nquota exceeded\n\n", returncode=3)
    _, error, _, _ = _run(proc)
    assert isinstance(error, base.ProviderError)
    assert str(error) == ("gemini failed (exit 3): quota exceeded. "
                          "Run gemini on its own to see the full error.")


def test_fr011_timeout_kills_the_tree_and_names_the_provider_and_seconds():
    proc = FakeProc(communicate_error=subprocess.TimeoutExpired(cmd="gemini", timeout=5))
    _, error, _, kill = _run(proc, timeout=5)
    assert isinstance(error, base.ProviderError)
    assert "gemini timed out after 5 s" in str(error)
    kill.assert_called_once_with(proc)


def test_fr011_keyboard_interrupt_kills_the_tree_and_is_reraised():
    proc = FakeProc(communicate_error=KeyboardInterrupt())
    _, error, _, kill = _run(proc)
    assert isinstance(error, KeyboardInterrupt)
    kill.assert_called_once_with(proc)


def test_fr011_binary_gone_mid_session_names_the_provider():
    with patch(_POPEN, side_effect=FileNotFoundError("gemini")), patch(_KILL):
        with pytest.raises(base.ProviderError) as exc:
            runner.run_cli("gemini", ["gemini"], "x")
    assert "gemini CLI not found" in str(exc.value)


def test_fr011_kill_process_tree_on_windows_runs_taskkill_on_the_pid():
    proc = FakeProc()
    with patch.object(runner.os, "name", "nt"), \
         patch("claude_light.providers.runner.subprocess.run") as run:
        runner.kill_process_tree(proc)
    assert run.call_args[0][0] == ["taskkill", "/F", "/T", "/PID", "4242"]


def test_fr011_kill_process_tree_on_posix_kills_the_process():
    proc = FakeProc()
    proc.kill = MagicMock()
    with patch.object(runner.os, "name", "posix"), \
         patch("claude_light.providers.runner.subprocess.run") as run:
        runner.kill_process_tree(proc)
    proc.kill.assert_called_once_with()
    run.assert_not_called()


def test_fr010_cwd_is_passed_to_popen(tmp_path):
    proc = FakeProc(stdout="ok")
    _, error, popen, _ = _run(proc, cwd=str(tmp_path))
    assert error is None
    assert popen.call_args.kwargs.get("cwd") == str(tmp_path)
