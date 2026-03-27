"""Integration test: verify git_manager module imports and exposes all expected functions."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestGitManagerImport(unittest.TestCase):

    _functions = [
        "is_git_repo",
        "get_git_root",
        "get_modified_files",
        "get_last_commit_message",
        "auto_commit",
        "undo_last_commit",
        "get_commit_history",
    ]

    def test_git_manager_importable(self):
        print("\n  ▶ TestGitManagerImport.test_git_manager_importable")
        from claude_light import git_manager  # noqa: F401

    def test_all_functions_present(self):
        print("\n  ▶ TestGitManagerImport.test_all_functions_present")
        from claude_light import git_manager
        for func_name in self._functions:
            with self.subTest(func=func_name):
                self.assertTrue(
                    hasattr(git_manager, func_name),
                    f"Function '{func_name}' missing from git_manager",
                )


if __name__ == "__main__":
    unittest.main()
