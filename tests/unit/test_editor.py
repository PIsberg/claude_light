"""
Tests for claude_light.editor — parse_edit_blocks, _resolve_new_content, apply_edits.
Also tests for claude_light.ui — show_diff, _colorize_diff.
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.editor import parse_edit_blocks, apply_edits, _resolve_new_content
from claude_light.ui import show_diff, _colorize_diff


# ---------------------------------------------------------------------------
# parse_edit_blocks
# ---------------------------------------------------------------------------

class TestParseEditBlocks(unittest.TestCase):

    def test_search_replace_block(self):
        print("\n  ▶ TestParseEditBlocks.test_search_replace_block")
        text = '''Here is the change:
```python:src/main.py
<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE
```
'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["type"], "edit")
        self.assertEqual(edits[0]["path"], "src/main.py")
        self.assertEqual(edits[0]["search"].strip(), "old text")
        self.assertEqual(edits[0]["replace"].strip(), "new text")

    def test_new_file_block(self):
        print("\n  ▶ TestParseEditBlocks.test_new_file_block")
        text = '''Here is the new file:
```python:src/utils.py
def util():
    pass
```
'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["type"], "new")
        self.assertEqual(edits[0]["path"], "src/utils.py")
        self.assertEqual(edits[0]["content"].strip(), "def util():\n    pass")

    def test_missing_tag_comment_path(self):
        print("\n  ▶ TestParseEditBlocks.test_missing_tag_comment_path")
        text = '''Here is the new file:
```python
# src/feature.py
def feature(): pass
```'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["type"], "new")
        self.assertEqual(edits[0]["path"], "src/feature.py")

    def test_conversational_block_ignored(self):
        print("\n  ▶ TestParseEditBlocks.test_conversational_block_ignored")
        text = '''You can just call it like this:
```python
feature()
```'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 0)

    def test_absolute_path_normalized(self):
        print("\n  ▶ TestParseEditBlocks.test_absolute_path_normalized")
        text = '```python:/home/user/workspace/src/test.py\ndef feature(): pass\n```'
        edits = parse_edit_blocks(text)
        self.assertEqual(edits[0]["path"], "home/user/workspace/src/test.py")

    def test_multiple_edits_in_one_response(self):
        print("\n  ▶ TestParseEditBlocks.test_multiple_edits_in_one_response")
        text = '''
```python:a.py
<<<<<<< SEARCH
old_a
=======
new_a
>>>>>>> REPLACE
```

```python:b.py
<<<<<<< SEARCH
old_b
=======
new_b
>>>>>>> REPLACE
```
'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 2)
        paths = [e["path"] for e in edits]
        self.assertIn("a.py", paths)
        self.assertIn("b.py", paths)

    def test_relative_path_stripped(self):
        print("\n  ▶ TestParseEditBlocks.test_relative_path_stripped")
        text = '```python:./src/foo.py\ndef foo(): pass\n```'
        edits = parse_edit_blocks(text)
        self.assertEqual(edits[0]["path"], "src/foo.py")

    def test_c_style_comment_path_extraction(self):
        print("\n  ▶ TestParseEditBlocks.test_c_style_comment_path_extraction")
        text = '```javascript\n// src/util.js\nfunction util() {}\n```'
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["path"], "src/util.js")

    def test_html_comment_path(self):
        print("\n  ▶ TestParseEditBlocks.test_html_comment_path")
        text = '```html\n<!-- index.html -->\n<h1>Hello</h1>\n```'
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["path"], "index.html")

    def test_go_new_file(self):
        print("\n  ▶ TestParseEditBlocks.test_go_new_file")
        text = '```go:cmd/main.go\npackage main\n\nfunc main() {}\n```'
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["type"], "new")
        self.assertEqual(edits[0]["path"], "cmd/main.go")

    def test_no_path_snippet_ignored(self):
        print("\n  ▶ TestParseEditBlocks.test_no_path_snippet_ignored")
        text = '```\nsome random snippet\n```'
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 0)


# ---------------------------------------------------------------------------
# _resolve_new_content
# ---------------------------------------------------------------------------

class TestResolveNewContent(unittest.TestCase):

    def setUp(self):
        self.original_code = "def foo():\n    try:\n        a = 1\n        b = 2\n    except:\n        pass\n"
        Path("dummy.py").write_text(self.original_code, encoding="utf-8")

    def tearDown(self):
        Path("dummy.py").unlink(missing_ok=True)

    def test_resolve_exact(self):
        print("\n  ▶ TestResolveNewContent.test_resolve_exact")
        search = "        a = 1\n        b = 2"
        replace = "        a = 3\n        b = 4"
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("a = 3", result)

    def test_resolve_fuzzy_indentation(self):
        print("\n  ▶ TestResolveNewContent.test_resolve_fuzzy_indentation")
        search = "a = 1\nb = 2\n"
        replace = "a = 5\nb = 5\n"
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("        a = 5", result)
        self.assertIn("        b = 5", result)

    def test_resolve_fuzzy_hallucination(self):
        print("\n  ▶ TestResolveNewContent.test_resolve_fuzzy_hallucination")
        search = "    try:\n        a = 1, b = 2"
        replace = "    try:\n        a = 9\n        b = 9"
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("        a = 9", result)

    def test_resolve_fuzzy_blank_line_in_replacement(self):
        print("\n  ▶ TestResolveNewContent.test_resolve_fuzzy_blank_line_in_replacement")
        search = "    try:\n        a = 1, b = 2"
        replace = "    try:\n        a = 9\n\n        b = 9"
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("a = 9", result)
        self.assertIn("b = 9", result)
        self.assertIn("\n\n", result)

    def test_new_file_no_existing(self):
        print("\n  ▶ TestResolveNewContent.test_new_file_no_existing")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                edit = {"type": "new", "path": "brand_new.py", "content": "x = 1\n"}
                old, new = _resolve_new_content(edit)
                self.assertEqual(old, "")
                self.assertEqual(new, "x = 1\n")
            finally:
                os.chdir(orig)

    def test_new_file_overwrites_existing(self):
        print("\n  ▶ TestResolveNewContent.test_new_file_overwrites_existing")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("existing.py").write_text("old content\n", encoding="utf-8")
                edit = {"type": "new", "path": "existing.py", "content": "new content\n"}
                old, new = _resolve_new_content(edit)
                self.assertEqual(old, "old content\n")
                self.assertEqual(new, "new content\n")
            finally:
                os.chdir(orig)

    def test_edit_missing_file_raises(self):
        print("\n  ▶ TestResolveNewContent.test_edit_missing_file_raises")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                edit = {"type": "edit", "path": "ghost.py", "search": "x", "replace": "y"}
                with self.assertRaises(FileNotFoundError):
                    _resolve_new_content(edit)
            finally:
                os.chdir(orig)

    def test_edit_search_not_found_raises(self):
        print("\n  ▶ TestResolveNewContent.test_edit_search_not_found_raises")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("real.py").write_text("def foo(): pass\n", encoding="utf-8")
                edit = {"type": "edit", "path": "real.py", "search": "DEFINITELY_NOT_IN_FILE", "replace": "y"}
                with self.assertRaises(ValueError):
                    _resolve_new_content(edit)
            finally:
                os.chdir(orig)

    def test_windows_line_endings(self):
        print("\n  ▶ TestResolveNewContent.test_windows_line_endings")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("win.py").write_bytes(b"def foo():\r\n    x = 1\r\n    return x\r\n")
                edit = {"type": "edit", "path": "win.py", "search": "x = 1\n    return x", "replace": "x = 99\n    return x"}
                old, new = _resolve_new_content(edit)
                self.assertIn("99", new)
            finally:
                os.chdir(orig)


# ---------------------------------------------------------------------------
# apply_edits (check_only mode)
# ---------------------------------------------------------------------------

class TestApplyEditsCheckOnly(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_edits_returns_empty_list(self):
        print("\n  ▶ TestApplyEditsCheckOnly.test_empty_edits_returns_empty_list")
        result = apply_edits([], check_only=True)
        self.assertEqual(result, [])

    def test_valid_python_no_errors(self):
        print("\n  ▶ TestApplyEditsCheckOnly.test_valid_python_no_errors")
        Path("foo.py").write_text("x = 1\n", encoding="utf-8")
        edits = [{"type": "edit", "path": "foo.py", "search": "x = 1", "replace": "x = 2"}]
        errs = apply_edits(edits, check_only=True)
        self.assertEqual(errs, [])

    def test_invalid_python_returns_error(self):
        print("\n  ▶ TestApplyEditsCheckOnly.test_invalid_python_returns_error")
        Path("bar.py").write_text("y = 1\n", encoding="utf-8")
        edits = [{"type": "edit", "path": "bar.py", "search": "y = 1", "replace": "y = (unclosed"}]
        errs = apply_edits(edits, check_only=True)
        self.assertEqual(len(errs), 1)
        self.assertIn("SyntaxError", errs[0])

    def test_new_file_type_no_error(self):
        print("\n  ▶ TestApplyEditsCheckOnly.test_new_file_type_no_error")
        edits = [{"type": "new", "path": "new_file.py", "content": "z = 42\n"}]
        errs = apply_edits(edits, check_only=True)
        self.assertEqual(errs, [])

    def test_missing_file_returns_error(self):
        print("\n  ▶ TestApplyEditsCheckOnly.test_missing_file_returns_error")
        edits = [{"type": "edit", "path": "missing.py", "search": "x", "replace": "y"}]
        errs = apply_edits(edits, check_only=True)
        self.assertEqual(len(errs), 1)
        self.assertIn("missing.py", errs[0])

    def test_syntax_error_check_only(self):
        print("\n  ▶ TestApplyEditsCheckOnly.test_syntax_error_check_only")
        Path("dummy.py").write_text("def foo():\n    try:\n        a = 1\n        b = 2\n    except:\n        pass\n", encoding="utf-8")
        search = "        a = 1\n        b = 2"
        replace = "        a = 3\n        print('missing parenthesis"
        errs = apply_edits([{"path": "dummy.py", "type": "edit", "search": search, "replace": replace}], check_only=True)
        self.assertEqual(len(errs), 1)
        self.assertIn("SyntaxError", errs[0])
        self.assertIn("dummy.py", errs[0])


# ---------------------------------------------------------------------------
# apply_edits (write mode)
# ---------------------------------------------------------------------------

class TestApplyEditsWrite(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.orig)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_stdout(self):
        buf = io.StringIO()
        return patch("sys.stdout", buf), buf

    def test_writes_new_file_in_noninteractive(self):
        print("\n  ▶ TestApplyEditsWrite.test_writes_new_file_in_noninteractive")
        edits = [{"type": "new", "path": "output.py", "content": "x = 99\n"}]
        p_stdout, buf = self._patch_stdout()
        with patch("sys.stdin") as mock_stdin, p_stdout:
            mock_stdin.isatty.return_value = False
            apply_edits(edits, check_only=False)
        self.assertTrue(Path("output.py").exists())
        self.assertIn("99", Path("output.py").read_text(encoding="utf-8"))

    def test_user_declines_edit(self):
        print("\n  ▶ TestApplyEditsWrite.test_user_declines_edit")
        Path("existing.py").write_text("x = 1\n", encoding="utf-8")
        edits = [{"type": "edit", "path": "existing.py", "search": "x = 1", "replace": "x = 2"}]
        p_stdout, buf = self._patch_stdout()
        with patch("sys.stdin") as mock_stdin, \
             patch("builtins.input", return_value="n"), p_stdout:
            mock_stdin.isatty.return_value = True
            apply_edits(edits, check_only=False)
        content = Path("existing.py").read_text(encoding="utf-8")
        self.assertIn("x = 1", content)

    def test_user_accepts_edit(self):
        print("\n  ▶ TestApplyEditsWrite.test_user_accepts_edit")
        Path("mod.py").write_text("x = 1\n", encoding="utf-8")
        edits = [{"type": "edit", "path": "mod.py", "search": "x = 1", "replace": "x = 99"}]
        p_stdout, buf = self._patch_stdout()
        with patch("sys.stdin") as mock_stdin, \
             patch("builtins.input", return_value="y"), p_stdout:
            mock_stdin.isatty.return_value = True
            apply_edits(edits, check_only=False)
        content = Path("mod.py").read_text(encoding="utf-8")
        self.assertIn("99", content)

    def test_empty_edits_prints_message(self):
        print("\n  ▶ TestApplyEditsWrite.test_empty_edits_prints_message")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            apply_edits([], check_only=False)
        self.assertIn("No file blocks found", captured.getvalue())

    def test_keyboard_interrupt_cancels(self):
        print("\n  ▶ TestApplyEditsWrite.test_keyboard_interrupt_cancels")
        Path("foo.py").write_text("x = 1\n", encoding="utf-8")
        edits = [{"type": "edit", "path": "foo.py", "search": "x = 1", "replace": "x = 2"}]
        with patch("sys.stdin") as mock_stdin, \
             patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("sys.stdout", io.StringIO()):
            mock_stdin.isatty.return_value = True
            apply_edits(edits, check_only=False)
        content = Path("foo.py").read_text(encoding="utf-8")
        self.assertIn("x = 1", content)


# ---------------------------------------------------------------------------
# show_diff
# ---------------------------------------------------------------------------

class TestShowDiff(unittest.TestCase):

    def test_shows_new_file(self):
        print("\n  ▶ TestShowDiff.test_shows_new_file")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            show_diff("some/path.py", "def foo(): pass\n", old_content="")
        self.assertIn("NEW FILE", captured.getvalue())

    def test_shows_diff_for_change(self):
        print("\n  ▶ TestShowDiff.test_shows_diff_for_change")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            show_diff("some/path.py", "def bar(): pass\n", old_content="def foo(): pass\n")
        self.assertIn("bar", captured.getvalue())

    def test_no_change_message(self):
        print("\n  ▶ TestShowDiff.test_no_change_message")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            show_diff("some/path.py", "same content", old_content="same content")
        self.assertIn("no changes detected", captured.getvalue())

    def test_reads_existing_file(self):
        print("\n  ▶ TestShowDiff.test_reads_existing_file")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("existing.py").write_text("old content\n", encoding="utf-8")
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    show_diff("existing.py", "new content\n")
                self.assertIn("new", captured.getvalue())
            finally:
                os.chdir(orig)


# ---------------------------------------------------------------------------
# _colorize_diff
# ---------------------------------------------------------------------------

class TestColorizeDiff(unittest.TestCase):

    def test_plus_lines_green(self):
        print("\n  ▶ TestColorizeDiff.test_plus_lines_green")
        lines = ["+added line"]
        result = _colorize_diff(lines)
        self.assertIn("\033[32m", result[0])

    def test_minus_lines_red(self):
        print("\n  ▶ TestColorizeDiff.test_minus_lines_red")
        lines = ["-removed line"]
        result = _colorize_diff(lines)
        self.assertIn("\033[31m", result[0])

    def test_context_lines_unchanged(self):
        print("\n  ▶ TestColorizeDiff.test_context_lines_unchanged")
        lines = [" context line"]
        result = _colorize_diff(lines)
        self.assertEqual(result[0], " context line")

    def test_hunk_header_cyan(self):
        print("\n  ▶ TestColorizeDiff.test_hunk_header_cyan")
        lines = ["@@ -1,3 +1,4 @@"]
        result = _colorize_diff(lines)
        self.assertIn("\033[36m", result[0])

    def test_plus_plus_not_colored_green(self):
        print("\n  ▶ TestColorizeDiff.test_plus_plus_not_colored_green")
        lines = ["--- a/file.py", "+++ b/file.py"]
        result = _colorize_diff(lines)
        self.assertNotIn("\033[32m", result[1])


if __name__ == "__main__":
    unittest.main()
