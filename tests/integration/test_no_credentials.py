"""
With no credential anywhere, `--help` must still print the usage (#101): it
is the first thing a user runs after installing. A command that needs the
API must still stop with "No authentication found".
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _run(*args):
    with tempfile.TemporaryDirectory(prefix="cl_noauth_") as home:
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")}
        # Empty home and cwd: no ~/.anthropic, ./.env, automation token or
        # ~/.claude/.credentials.json to find.
        env.update(HOME=home, USERPROFILE=home, PYTHONIOENCODING="utf-8",
                   CLAUDE_LIGHT_LLMLINGUA="0")
        return subprocess.run(
            [sys.executable, str(ROOT / "claude_light.py"), *args],
            cwd=home, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=120)


class TestNoCredentials(unittest.TestCase):

    def test_help_works_without_a_credential(self):
        proc = _run("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertIn("usage:", proc.stdout)
        self.assertNotIn("No authentication found", proc.stderr)

    def test_a_question_without_a_credential_still_stops(self):
        proc = _run("What does main do?")
        self.assertEqual(proc.returncode, 1, proc.stdout[-1500:])
        self.assertIn("No authentication found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
