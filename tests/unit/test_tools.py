"""
Tests for claude_light.tools — the read-only workspace tools the model can call
in API-key mode (read_file, grep, list_files).
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.tools import TOOLS, run_tool


class TestWorkspaceTools(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cl_tools_"))
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "retry.py").write_text(
            "import time\n\n\ndef retry_with_backoff(fn):\n    return fn\n", encoding="utf-8")
        (self.root / "pkg" / "ui.py").write_text("def show():\n    print('hi')\n", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("retry_with_backoff in git internals\n", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "dep.js").write_text("retry_with_backoff()\n", encoding="utf-8")
        outside = self.root.parent / (self.root.name + "_outside.txt")
        outside.write_text("secret\n", encoding="utf-8")
        self.outside = outside

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        self.outside.unlink(missing_ok=True)

    def run_tool(self, name, tool_input):
        return run_tool(name, tool_input, root=self.root)

    def test_tool_definitions_are_read_only_and_well_formed(self):
        print("\n  ▶ TestWorkspaceTools.test_tool_definitions_are_read_only_and_well_formed")
        names = {t["name"] for t in TOOLS}
        self.assertEqual(names, {"read_file", "grep", "list_files"})
        for t in TOOLS:
            self.assertEqual(t["input_schema"]["type"], "object")
            self.assertTrue(t["description"])

    def test_read_file_returns_numbered_lines(self):
        print("\n  ▶ TestWorkspaceTools.test_read_file_returns_numbered_lines")
        out, is_error = self.run_tool("read_file", {"path": "pkg/retry.py"})
        self.assertFalse(is_error)
        self.assertIn("4\tdef retry_with_backoff(fn):", out)

    def test_read_file_line_range(self):
        print("\n  ▶ TestWorkspaceTools.test_read_file_line_range")
        out, is_error = self.run_tool("read_file", {"path": "pkg/retry.py", "start_line": 4, "end_line": 5})
        self.assertFalse(is_error)
        self.assertNotIn("import time", out)
        self.assertIn("5\t    return fn", out)

    def test_paths_outside_the_project_are_refused(self):
        print("\n  ▶ TestWorkspaceTools.test_paths_outside_the_project_are_refused")
        for path in ("../" + self.outside.name, str(self.outside)):
            out, is_error = self.run_tool("read_file", {"path": path})
            self.assertTrue(is_error, path)
            self.assertNotIn("secret", out)

    def test_missing_file_is_an_error_not_an_exception(self):
        print("\n  ▶ TestWorkspaceTools.test_missing_file_is_an_error_not_an_exception")
        out, is_error = self.run_tool("read_file", {"path": "pkg/nope.py"})
        self.assertTrue(is_error)
        self.assertIn("not found", out.lower())

    def test_grep_finds_matches_and_skips_vendored_and_vcs_dirs(self):
        print("\n  ▶ TestWorkspaceTools.test_grep_finds_matches_and_skips_vendored_and_vcs_dirs")
        out, is_error = self.run_tool("grep", {"pattern": r"def retry_\w+"})
        self.assertFalse(is_error)
        self.assertIn("pkg/retry.py:4:", out)
        self.assertNotIn(".git", out)
        self.assertNotIn("node_modules", out)

    def test_grep_glob_filter(self):
        print("\n  ▶ TestWorkspaceTools.test_grep_glob_filter")
        out, _ = self.run_tool("grep", {"pattern": "def ", "glob": "*ui.py"})
        self.assertIn("pkg/ui.py:1:", out)
        self.assertNotIn("retry.py", out)

    def test_invalid_regex_is_an_error(self):
        print("\n  ▶ TestWorkspaceTools.test_invalid_regex_is_an_error")
        out, is_error = self.run_tool("grep", {"pattern": "def ("})
        self.assertTrue(is_error)

    def test_list_files(self):
        print("\n  ▶ TestWorkspaceTools.test_list_files")
        out, is_error = self.run_tool("list_files", {"path": "pkg"})
        self.assertFalse(is_error)
        self.assertIn("pkg/retry.py", out)
        self.assertIn("pkg/ui.py", out)

    def test_invalid_input_and_unknown_tool_are_errors(self):
        print("\n  ▶ TestWorkspaceTools.test_invalid_input_and_unknown_tool_are_errors")
        self.assertTrue(self.run_tool("read_file", {})[1])
        self.assertTrue(self.run_tool("read_file", {"path": 42})[1])
        self.assertTrue(self.run_tool("read_file", {"path": "pkg/retry.py", "start_line": "x"})[1])
        self.assertTrue(self.run_tool("delete_everything", {"path": "."})[1])


if __name__ == "__main__":
    unittest.main()
