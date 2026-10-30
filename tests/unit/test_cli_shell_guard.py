"""
Tests for claude_light.cli_shell_guard, the PreToolUse hook that stops the
OAuth CLI agent chaining a second command after an allowlisted prefix
(issue #70). The CLI matched Bash(echo:*) against the first statement only,
so `echo step1 && touch marker` ran the touch too (verified live, claude
2.1.314). The hook runs exactly as the CLI runs it: a fresh interpreter fed
the hook JSON on stdin, exit code 2 to block.
"""

import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light import cli_shell_guard, tools


def _run_hook(payload):
    """Run the guard the way the CLI does. Returns (exit code, stderr)."""
    data = payload if isinstance(payload, str) else json.dumps(payload)
    proc = subprocess.run(
        [sys.executable, "-I", cli_shell_guard.__file__],
        input=data, capture_output=True, text=True, timeout=30,
    )
    return proc.returncode, proc.stderr


def _bash(command, tool="Bash"):
    return {"hook_event_name": "PreToolUse", "tool_name": tool,
            "tool_input": {"command": command}}


class TestCliShellGuard(unittest.TestCase):

    def test_chained_commands_are_blocked(self):
        print("\n  ▶ TestCliShellGuard.test_chained_commands_are_blocked")
        for command in ("echo step1 && touch marker_b.txt",
                        "echo a; rm -rf x",
                        "npm test | tee out",
                        "npm test > out",
                        "echo $(whoami)",
                        "echo `whoami`",
                        "echo a\ntouch b"):
            code, err = _run_hook(_bash(command))
            self.assertEqual(code, 2, command)
            self.assertIn("one simple command", err)

    def test_powershell_chaining_is_blocked(self):
        print("\n  ▶ TestCliShellGuard.test_powershell_chaining_is_blocked")
        code, _ = _run_hook(_bash("Get-ChildItem -Name . ; New-Item marker_b.txt", "PowerShell"))
        self.assertEqual(code, 2)

    def test_a_single_command_passes(self):
        print("\n  ▶ TestCliShellGuard.test_a_single_command_passes")
        code, err = _run_hook(_bash("python -m pytest tests/unit -q"))
        self.assertEqual((code, err), (0, ""))

    def test_unreadable_input_fails_closed(self):
        print("\n  ▶ TestCliShellGuard.test_unreadable_input_fails_closed")
        # Exit 1 is a non-blocking hook error to the CLI: the command would
        # run. Anything the guard cannot read must block instead.
        for payload in ("not json", "[]", {"tool_name": "Bash", "tool_input": {}},
                        {"tool_name": "Bash", "tool_input": {"command": 3}}):
            code, _ = _run_hook(payload)
            self.assertEqual(code, 2, payload)

    def test_same_characters_as_the_api_shell_tool(self):
        print("\n  ▶ TestCliShellGuard.test_same_characters_as_the_api_shell_tool")
        # Both shell paths promise the same rule; one set backs both.
        self.assertIs(tools._SHELL_FORBIDDEN, cli_shell_guard.SHELL_FORBIDDEN)


class TestCliSettingsInstallTheGuard(unittest.TestCase):

    def _settings_for(self, allowed, tools_on=True):
        from claude_light import llm, config
        seen = {}

        def fake_popen(cmd, **kwargs):
            with open(cmd[cmd.index("--settings") + 1], encoding="utf-8") as fh:
                seen.update(json.load(fh))
            proc = MagicMock()
            proc.stdout = iter(['{"type":"result","subtype":"success","result":"","usage":{}}\n'])
            proc.stderr = io.StringIO("")
            proc.wait.return_value = proc.poll.return_value = proc.returncode = 0
            return proc

        with patch.object(config, "ALLOWED_COMMANDS", allowed), \
             patch("claude_light.llm.subprocess.Popen", side_effect=fake_popen), \
             patch("shutil.which", return_value="/fake/bin/claude"), \
             patch("sys.stdout", io.StringIO()):
            llm._make_cli_subprocess_call("hi", tools=tools_on)
        return seen

    def test_allowlist_installs_a_pre_tool_use_guard_for_both_shells(self):
        print("\n  ▶ TestCliSettingsInstallTheGuard.test_allowlist_installs_a_pre_tool_use_guard_for_both_shells")
        settings = self._settings_for(["echo"])
        entries = settings["hooks"]["PreToolUse"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(set(entries[0]["matcher"].split("|")), {"Bash", "PowerShell"})
        hook = entries[0]["hooks"][0]
        self.assertEqual(hook["type"], "command")
        # Isolated mode, so a repo's own files or PYTHONPATH cannot replace
        # the guard; the script is named by absolute path, not run with -m.
        self.assertIn(" -I ", hook["command"])
        self.assertIn(Path(cli_shell_guard.__file__).as_posix(), hook["command"])

    def test_the_installed_command_blocks_a_chained_command(self):
        print("\n  ▶ TestCliSettingsInstallTheGuard.test_the_installed_command_blocks_a_chained_command")
        # Run the exact hook command line from the settings file through a
        # shell, as the CLI does, so a quoting mistake shows up here.
        hook = self._settings_for(["echo"])["hooks"]["PreToolUse"][0]["hooks"][0]
        proc = subprocess.run(hook["command"], shell=True, input=json.dumps(
            _bash("echo step1 && touch marker_b.txt")), capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 2, proc.stderr)

    def test_no_guard_without_an_allowlist(self):
        print("\n  ▶ TestCliSettingsInstallTheGuard.test_no_guard_without_an_allowlist")
        # No allowlist means both shells are disallowed outright; no hook.
        self.assertNotIn("hooks", self._settings_for([]))
        self.assertNotIn("hooks", self._settings_for(["echo"], tools_on=False))


if __name__ == "__main__":
    unittest.main()
