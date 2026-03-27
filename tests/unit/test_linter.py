"""
Tests for claude_light.linter — _lint_python_content, _lint_content dispatcher,
and tree-sitter linting for Java/JS/TS.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.linter import _lint_content, _lint_python_content


class TestLintPythonContent(unittest.TestCase):
    """Tests for Python syntax checking."""

    def test_valid_returns_none(self):
        print("\n  ▶ TestLintPythonContent.test_valid_returns_none")
        result = _lint_python_content("test.py", "x = 1 + 2\n")
        self.assertIsNone(result)

    def test_syntax_error_returns_string(self):
        print("\n  ▶ TestLintPythonContent.test_syntax_error_returns_string")
        result = _lint_python_content("test.py", "def foo(:\n    pass\n")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_unclosed_paren_returns_string(self):
        print("\n  ▶ TestLintPythonContent.test_unclosed_paren_returns_string")
        result = _lint_python_content("test.py", "x = (1 + 2\n")
        self.assertIsNotNone(result)

    def test_complex_valid_code(self):
        print("\n  ▶ TestLintPythonContent.test_complex_valid_code")
        code = "class Foo:\n    def bar(self):\n        return 42\n"
        result = _lint_python_content("foo.py", code)
        self.assertIsNone(result)


class TestLintContentDispatcher(unittest.TestCase):
    """Tests for the _lint_content extension dispatcher."""

    def test_unknown_extension_returns_none(self):
        print("\n  ▶ TestLintContentDispatcher.test_unknown_extension_returns_none")
        result = _lint_content("script.rb", "puts 'hello'")
        self.assertIsNone(result)

    def test_css_returns_none(self):
        print("\n  ▶ TestLintContentDispatcher.test_css_returns_none")
        result = _lint_content("style.css", "color: red;")
        self.assertIsNone(result)

    def test_python_valid(self):
        print("\n  ▶ TestLintContentDispatcher.test_python_valid")
        result = _lint_content("app.py", "x = 1\n")
        self.assertIsNone(result)

    def test_python_invalid(self):
        print("\n  ▶ TestLintContentDispatcher.test_python_invalid")
        result = _lint_content("app.py", "def broken(:\n    pass\n")
        self.assertIsNotNone(result)

    def test_skips_unknown_extensions(self):
        print("\n  ▶ TestLintContentDispatcher.test_skips_unknown_extensions")
        self.assertIsNone(_lint_content("script.sh", "this is { not valid anything"))
        self.assertIsNone(_lint_content("style.css", "color: ;{{{"))


class TestLintViaTreesitter(unittest.TestCase):
    """Tests that use tree-sitter for Java/JS/TS linting (skipped if unavailable)."""

    def test_valid_java_via_lint_content(self):
        print("\n  ▶ TestLintViaTreesitter.test_valid_java_via_lint_content")
        import claude_light as cl
        if not cl._TREESITTER_AVAILABLE:
            self.skipTest("tree-sitter not available")
        try:
            import tree_sitter_java  # noqa: F401
        except ImportError:
            self.skipTest("tree-sitter-java not installed")
        valid_java = (
            "public class Foo {\n"
            "    public void bar() {\n"
            "        int x = 1;\n"
            "    }\n"
            "}\n"
        )
        self.assertIsNone(_lint_content("Foo.java", valid_java))

    def test_broken_java_reports_error(self):
        print("\n  ▶ TestLintViaTreesitter.test_broken_java_reports_error")
        import claude_light as cl
        if not cl._TREESITTER_AVAILABLE:
            self.skipTest("tree-sitter not available")
        try:
            import tree_sitter_java  # noqa: F401
        except ImportError:
            self.skipTest("tree-sitter-java not installed")
        broken_java = (
            "public class Foo {\n"
            "    public void bar( {\n"  # missing closing paren
            "        int x = 1;\n"
            "    }\n"
            "}\n"
        )
        err = _lint_content("Foo.java", broken_java)
        self.assertIsNotNone(err)
        self.assertIn("SyntaxError", err)

    def test_valid_javascript_via_lint_content(self):
        print("\n  ▶ TestLintViaTreesitter.test_valid_javascript_via_lint_content")
        import claude_light as cl
        if not cl._TREESITTER_AVAILABLE:
            self.skipTest("tree-sitter not available")
        try:
            import tree_sitter_javascript  # noqa: F401
        except ImportError:
            self.skipTest("tree-sitter-javascript not installed")
        valid_js = "function greet(name) {\n  return 'Hello, ' + name;\n}\n"
        self.assertIsNone(_lint_content("app.js", valid_js))

    def test_broken_javascript_reports_error(self):
        print("\n  ▶ TestLintViaTreesitter.test_broken_javascript_reports_error")
        import claude_light as cl
        if not cl._TREESITTER_AVAILABLE:
            self.skipTest("tree-sitter not available")
        try:
            import tree_sitter_javascript  # noqa: F401
        except ImportError:
            self.skipTest("tree-sitter-javascript not installed")
        broken_js = "function greet(name {\n  return 'Hello';\n}\n"
        err = _lint_content("app.js", broken_js)
        self.assertIsNotNone(err)
        self.assertIn("SyntaxError", err)


class TestLintLanguagesNoTreesitter(unittest.TestCase):
    """Tests for graceful degradation when tree-sitter unavailable."""

    def test_java_linter_no_treesitter(self):
        print("\n  ▶ TestLintLanguagesNoTreesitter.test_java_linter_no_treesitter")
        from claude_light.linter import _lint_java_content
        import claude_light as cl
        with patch.object(cl, "_TREESITTER_AVAILABLE", False):
            result = _lint_java_content("Foo.java", "class Foo {}")
        self.assertIsNone(result)

    def test_js_linter_no_treesitter(self):
        print("\n  ▶ TestLintLanguagesNoTreesitter.test_js_linter_no_treesitter")
        from claude_light.linter import _lint_javascript_content
        import claude_light as cl
        with patch.object(cl, "_TREESITTER_AVAILABLE", False):
            result = _lint_javascript_content("app.js", "function foo() {}")
        self.assertIsNone(result)

    def test_ts_linter_no_treesitter(self):
        print("\n  ▶ TestLintLanguagesNoTreesitter.test_ts_linter_no_treesitter")
        from claude_light.linter import _lint_typescript_content
        import claude_light as cl
        with patch.object(cl, "_TREESITTER_AVAILABLE", False):
            result = _lint_typescript_content("app.ts", "const x: number = 1;")
        self.assertIsNone(result)

    def test_java_linter_import_error(self):
        print("\n  ▶ TestLintLanguagesNoTreesitter.test_java_linter_import_error")
        from claude_light.linter import _lint_java_content
        import claude_light as cl
        with patch.object(cl, "_TREESITTER_AVAILABLE", True):
            with patch.dict("sys.modules", {"tree_sitter_java": None}):
                result = _lint_java_content("Foo.java", "class Foo {}")
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
