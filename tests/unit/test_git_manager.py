"""
Tests for claude_light.git_manager — isolated from the real repo via mocked
subprocess.run calls.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light import git_manager


class TestGetModifiedFiles(unittest.TestCase):
    """get_modified_files must parse `git status --porcelain` correctly even
    though _run_git.strip() eats the leading space on the first line."""

    def _mock_git_output(self, stdout: str, returncode: int = 0):
        mock = MagicMock()
        mock.stdout = stdout
        mock.returncode = returncode
        return mock

    def test_single_unstaged_modification_does_not_truncate_path(self):
        print("\n  ▶ TestGetModifiedFiles.test_single_unstaged_modification_does_not_truncate_path")
        # `git status --porcelain` for an unstaged edit emits " M path".
        # _run_git does .strip() on the whole stdout, so the first line's
        # leading space is removed -> "M path". A naive line[3:] would
        # return "ocs/architecture.md" instead of "docs/architecture.md".
        # Regression test for that bug.
        with patch("claude_light.git_manager.subprocess.run",
                   return_value=self._mock_git_output(" M docs/architecture.md\n")):
            files = git_manager.get_modified_files()
        self.assertEqual(files, ["docs/architecture.md"])

    def test_multiple_files_all_preserved(self):
        print("\n  ▶ TestGetModifiedFiles.test_multiple_files_all_preserved")
        stdout = " M a.py\n M b.py\nMM c.py\n"
        with patch("claude_light.git_manager.subprocess.run",
                   return_value=self._mock_git_output(stdout)):
            files = git_manager.get_modified_files()
        self.assertEqual(files, ["a.py", "b.py", "c.py"])

    def test_staged_new_file(self):
        print("\n  ▶ TestGetModifiedFiles.test_staged_new_file")
        with patch("claude_light.git_manager.subprocess.run",
                   return_value=self._mock_git_output("A  new_module.py\n")):
            files = git_manager.get_modified_files()
        self.assertEqual(files, ["new_module.py"])

    def test_empty_output_returns_empty_list(self):
        print("\n  ▶ TestGetModifiedFiles.test_empty_output_returns_empty_list")
        with patch("claude_light.git_manager.subprocess.run",
                   return_value=self._mock_git_output("")):
            files = git_manager.get_modified_files()
        self.assertEqual(files, [])

    def test_nonzero_returncode_returns_empty_list(self):
        print("\n  ▶ TestGetModifiedFiles.test_nonzero_returncode_returns_empty_list")
        with patch("claude_light.git_manager.subprocess.run",
                   return_value=self._mock_git_output("", returncode=128)):
            files = git_manager.get_modified_files()
        self.assertEqual(files, [])

    def test_path_with_subdirectories(self):
        print("\n  ▶ TestGetModifiedFiles.test_path_with_subdirectories")
        with patch("claude_light.git_manager.subprocess.run",
                   return_value=self._mock_git_output(" M a/b/c/d.py\n")):
            files = git_manager.get_modified_files()
        self.assertEqual(files, ["a/b/c/d.py"])


if __name__ == "__main__":
    unittest.main()
