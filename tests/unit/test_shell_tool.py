"""Tests for the allowlist-gated `shell` tool in claude_light.tools."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light import config
from claude_light.tools import run_tool

_PY = f'"{sys.executable}"' if os.name == "nt" else sys.executable


class TestShellTool(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cl_shell_"))

    def run_shell(self, command, allowed=None):
        with mock.patch.object(config, "ALLOWED_COMMANDS", list(allowed or [])):
            return run_tool("shell", {"command": command}, root=self.root)

    def test_refused_when_nothing_allowed(self):
        out, is_error = self.run_shell("echo hi")
        self.assertTrue(is_error)
        self.assertIn("not allowed", out)

    def test_refused_when_not_on_allowlist(self):
        out, is_error = self.run_shell("rm -rf x", allowed=["echo"])
        self.assertTrue(is_error)
        self.assertIn("not allowed", out)

    def test_prefix_needs_word_boundary(self):
        out, is_error = self.run_shell("echoes hi", allowed=["echo"])
        self.assertTrue(is_error)

    def test_chaining_and_redirection_refused(self):
        for cmd in ("echo a; echo b", "echo a && echo b", "echo a | cat",
                    "echo a > f", "echo `id`", "echo $HOME", "echo a\necho b"):
            out, is_error = self.run_shell(cmd, allowed=["echo"])
            self.assertTrue(is_error, cmd)
            self.assertIn("may not contain", out)

    def test_allowed_command_runs_in_root(self):
        out, is_error = self.run_shell("echo hello", allowed=["echo"])
        self.assertFalse(is_error)
        self.assertIn("(exit 0)", out)
        self.assertIn("hello", out)

    def test_nonzero_exit_is_error(self):
        cmd = f"{_PY} -c \"raise SystemExit(3)\""
        out, is_error = self.run_shell(cmd, allowed=[_PY])
        self.assertTrue(is_error)
        self.assertIn("(exit 3)", out)

    def test_timeout_kills_command(self):
        cmd = f"{_PY} -c \"import time; time.sleep(30)\""
        with mock.patch.dict(os.environ, {"CLAUDE_LIGHT_RUN_TIMEOUT": "1"}):
            out, is_error = self.run_shell(cmd, allowed=[_PY])
        self.assertTrue(is_error)
        self.assertIn("TIMEOUT", out)

    def test_missing_command_arg(self):
        out, is_error = run_tool("shell", {}, root=self.root)
        self.assertTrue(is_error)


if __name__ == "__main__":
    unittest.main()
