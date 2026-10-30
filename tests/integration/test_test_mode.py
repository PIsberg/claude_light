"""
`claude_light.py --test-mode` is the documented offline way to run the tool
(CLAUDE.md, Commands). It runs on MockManager (tests/utilities/test_mocks.py),
which patches the code it mocks by name, so it drifts silently when that code
changes: by 2026-10-30 it crashed on a missing st_mtime and then on a changed
print_stats signature (#100). This runs it end to end, as a user would.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


class TestTestMode(unittest.TestCase):

    def test_small_one_shot_answers_offline(self):
        env = dict(os.environ)
        env["CLAUDE_LIGHT_LLMLINGUA"] = "0"  # no model download in a test
        env["PYTHONIOENCODING"] = "utf-8"
        with tempfile.TemporaryDirectory(prefix="cl_test_mode_") as cwd:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "claude_light.py"), "--test-mode", "small",
                 "What does doTask1 in Service0 do?"],
                cwd=cwd, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=300,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("Simulated test response", proc.stdout)
        # Retrieval ran: the mock names the methods it found in the context.
        self.assertIn("doTask", proc.stdout)


if __name__ == "__main__":
    unittest.main()
