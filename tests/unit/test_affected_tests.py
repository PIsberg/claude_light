"""Tests for claude_light.affected_tests: mapping edits to the tests to run."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light import affected_tests as at
from claude_light import config


class TestAffectedTests(unittest.TestCase):

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cl_aff_"))
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "billing.py").write_text("def total(): return 1\n", encoding="utf-8")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_billing.py").write_text("from pkg import billing\n", encoding="utf-8")
        (self.root / "tests" / "test_other.py").write_text("import os\n", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "test_billing.py").write_text("billing\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_module_edit_maps_to_tests_mentioning_it(self):
        self.assertEqual(at.find_affected_tests(["pkg/billing.py"], self.root),
                         ["tests/test_billing.py"])

    def test_edited_test_file_maps_to_itself(self):
        self.assertEqual(at.find_affected_tests(["tests/test_other.py"], self.root),
                         ["tests/test_other.py"])

    def test_helpers_under_tests_are_not_run_directly(self):
        (self.root / "tests" / "conftest.py").write_text("import billing\n", encoding="utf-8")
        (self.root / "tests" / "helpers.py").write_text("import billing\n", encoding="utf-8")
        self.assertEqual(at.find_affected_tests(["pkg/billing.py"], self.root),
                         ["tests/test_billing.py"])

    def test_non_python_edit_maps_to_nothing(self):
        self.assertEqual(at.find_affected_tests(["README.md"], self.root), [])

    def test_build_command_uses_allowlisted_pytest(self):
        cmd = at.build_test_command(["tests/test_a.py"], allowed=["python -m pytest"])
        self.assertEqual(cmd, "python -m pytest tests/test_a.py -q")

    def test_build_command_none_without_pytest_entry(self):
        self.assertIsNone(at.build_test_command(["tests/test_a.py"], allowed=["npm test"]))
        self.assertIsNone(at.build_test_command([], allowed=["python -m pytest"]))

    def test_check_is_noop_without_allowlist(self):
        with mock.patch.object(config, "ALLOWED_COMMANDS", []):
            self.assertIsNone(at.check_after_edit(["pkg/billing.py"], self.root))

    def test_check_reports_pass_and_fail(self):
        with mock.patch.object(config, "ALLOWED_COMMANDS", ["python -m pytest"]), \
             mock.patch("claude_light.tools.run_tool", return_value=("ok", False)):
            self.assertTrue(at.check_after_edit(["pkg/billing.py"], self.root))
        with mock.patch.object(config, "ALLOWED_COMMANDS", ["python -m pytest"]), \
             mock.patch("claude_light.tools.run_tool", return_value=("FAILED", True)), \
             mock.patch("claude_light.git_manager.is_git_repo", return_value=False):
            self.assertFalse(at.check_after_edit(["pkg/billing.py"], self.root))


if __name__ == "__main__":
    unittest.main()
