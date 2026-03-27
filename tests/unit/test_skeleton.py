"""
Tests for claude_light.skeleton — build_skeleton, _assemble_skeleton, _render_md_file,
_refresh_single_md, _refresh_tree_only, and related helpers.
Also tests for claude_light.indexer — _file_hash.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.skeleton import (
    build_skeleton, _assemble_skeleton, _render_md_file,
    _refresh_single_md, _refresh_tree_only, _is_skipped,
)
from claude_light.indexer import _file_hash


# ---------------------------------------------------------------------------
# _is_skipped
# ---------------------------------------------------------------------------

class TestIsSkipped(unittest.TestCase):

    def test_skips_git(self):
        print("\n  ▶ TestIsSkipped.test_skips_git")
        self.assertTrue(_is_skipped(Path(".git/config")))

    def test_skips_node_modules(self):
        print("\n  ▶ TestIsSkipped.test_skips_node_modules")
        self.assertTrue(_is_skipped(Path("src/node_modules/foo/bar.js")))

    def test_skips_dotted_dirs(self):
        print("\n  ▶ TestIsSkipped.test_skips_dotted_dirs")
        self.assertTrue(_is_skipped(Path(".hidden/file.py")))

    def test_does_not_skip_normal(self):
        print("\n  ▶ TestIsSkipped.test_does_not_skip_normal")
        self.assertFalse(_is_skipped(Path("src/main/java/Foo.java")))

    def test_does_not_skip_root_file(self):
        print("\n  ▶ TestIsSkipped.test_does_not_skip_root_file")
        self.assertFalse(_is_skipped(Path("README.md")))

    def test_skips_pycache(self):
        print("\n  ▶ TestIsSkipped.test_skips_pycache")
        self.assertTrue(_is_skipped(Path("src/__pycache__/foo.pyc")))

    def test_skips_build_dir(self):
        print("\n  ▶ TestIsSkipped.test_skips_build_dir")
        self.assertTrue(_is_skipped(Path("build/classes/Foo.class")))


# ---------------------------------------------------------------------------
# _file_hash
# ---------------------------------------------------------------------------

class TestFileHash(unittest.TestCase):

    def test_produces_md5(self):
        print("\n  ▶ TestFileHash.test_produces_md5")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.txt"
            p.write_text("hello", encoding="utf-8")
            h = _file_hash(p)
            self.assertIsInstance(h, str)
            self.assertEqual(len(h), 32)

    def test_different_content_different_hash(self):
        print("\n  ▶ TestFileHash.test_different_content_different_hash")
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "a.txt"
            p2 = Path(tmpdir) / "b.txt"
            p1.write_text("hello", encoding="utf-8")
            p2.write_text("world", encoding="utf-8")
            self.assertNotEqual(_file_hash(p1), _file_hash(p2))

    def test_same_content_same_hash(self):
        print("\n  ▶ TestFileHash.test_same_content_same_hash")
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "a.txt"
            p2 = Path(tmpdir) / "b.txt"
            p1.write_text("same content", encoding="utf-8")
            p2.write_text("same content", encoding="utf-8")
            self.assertEqual(_file_hash(p1), _file_hash(p2))


# ---------------------------------------------------------------------------
# build_skeleton
# ---------------------------------------------------------------------------

class TestBuildSkeleton(unittest.TestCase):

    def test_returns_directory_structure_header(self):
        print("\n  ▶ TestBuildSkeleton.test_returns_directory_structure_header")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("README.md").write_text("# test", encoding="utf-8")
                result = build_skeleton()
                self.assertIn("Directory structure:", result)
            finally:
                os.chdir(orig)

    def test_includes_md_content(self):
        print("\n  ▶ TestBuildSkeleton.test_includes_md_content")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("NOTES.md").write_text("# My Notes\nSome content here", encoding="utf-8")
                result = build_skeleton()
                self.assertIn("My Notes", result)
            finally:
                os.chdir(orig)

    def test_skips_dotted_dirs(self):
        print("\n  ▶ TestBuildSkeleton.test_skips_dotted_dirs")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                hidden = Path(".hidden")
                hidden.mkdir()
                (hidden / "secret.py").write_text("# secret", encoding="utf-8")
                result = build_skeleton()
                self.assertNotIn(".hidden", result)
            finally:
                os.chdir(orig)

    def test_truncates_long_md_files(self):
        print("\n  ▶ TestBuildSkeleton.test_truncates_long_md_files")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                big_content = "x" * 10000
                Path("BIG.md").write_text(big_content, encoding="utf-8")
                result = build_skeleton()
                self.assertIn("TRUNCATED", result)
            finally:
                os.chdir(orig)

    def test_does_not_truncate_claude_md(self):
        print("\n  ▶ TestBuildSkeleton.test_does_not_truncate_claude_md")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                big_content = "y" * 6000
                Path("CLAUDE.md").write_text(big_content, encoding="utf-8")
                result = build_skeleton()
                self.assertNotIn("TRUNCATED", result)
            finally:
                os.chdir(orig)

    def test_reuses_cached_md_when_unchanged(self):
        print("\n  ▶ TestBuildSkeleton.test_reuses_cached_md_when_unchanged")
        import claude_light as cl
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                p = Path("guide.md")
                p.write_text("# Guide\nContent", encoding="utf-8")
                h = _file_hash(p)
                cl._skeleton_md_hashes = {str(p): h}
                cl._skeleton_md_parts = {str(p): "cached_render"}
                result = build_skeleton()
                self.assertIn("cached_render", result)
            finally:
                os.chdir(orig)
                cl._skeleton_md_hashes = {}
                cl._skeleton_md_parts = {}


# ---------------------------------------------------------------------------
# _assemble_skeleton / _refresh_single_md
# ---------------------------------------------------------------------------

class TestSkeletonHelpers(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        self._orig_tree = cl._skeleton_tree
        self._orig_hashes = dict(cl._skeleton_md_hashes)
        self._orig_parts = dict(cl._skeleton_md_parts)

    def tearDown(self):
        import claude_light as cl
        cl._skeleton_tree = self._orig_tree
        cl._skeleton_md_hashes = self._orig_hashes
        cl._skeleton_md_parts = self._orig_parts

    def test_assemble_skeleton_combines_tree_and_docs(self):
        print("\n  ▶ TestSkeletonHelpers.test_assemble_skeleton_combines_tree_and_docs")
        import claude_light as cl
        cl._skeleton_tree = "Directory structure:\n  foo/"
        cl._skeleton_md_parts = {"README.md": "<!-- README.md -->\nHello"}
        result = _assemble_skeleton()
        self.assertIn("Directory structure:", result)
        self.assertIn("Hello", result)

    def test_assemble_skeleton_empty_parts(self):
        print("\n  ▶ TestSkeletonHelpers.test_assemble_skeleton_empty_parts")
        import claude_light as cl
        cl._skeleton_tree = "Directory structure:\n"
        cl._skeleton_md_parts = {}
        result = _assemble_skeleton()
        self.assertNotIn("None", result)

    def test_refresh_single_md_nonexistent_cleans_up(self):
        print("\n  ▶ TestSkeletonHelpers.test_refresh_single_md_nonexistent_cleans_up")
        import claude_light as cl
        cl._skeleton_md_hashes = {"ghost.md": "abc123"}
        cl._skeleton_md_parts = {"ghost.md": "old content"}
        changed = _refresh_single_md("ghost.md")
        self.assertTrue(changed)
        self.assertNotIn("ghost.md", cl._skeleton_md_parts)

    def test_refresh_single_md_unchanged(self):
        print("\n  ▶ TestSkeletonHelpers.test_refresh_single_md_unchanged")
        import claude_light as cl
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                p = Path("notes.md")
                p.write_text("# Notes", encoding="utf-8")
                h = _file_hash(p)
                cl._skeleton_md_hashes = {"notes.md": h}
                cl._skeleton_md_parts = {"notes.md": "cached content"}
                changed = _refresh_single_md("notes.md")
                self.assertFalse(changed)
            finally:
                os.chdir(orig)


# ---------------------------------------------------------------------------
# _render_md_file
# ---------------------------------------------------------------------------

class TestRenderMdFile(unittest.TestCase):

    def test_reads_and_renders(self):
        print("\n  ▶ TestRenderMdFile.test_reads_and_renders")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.md"
            p.write_text("# Hello\nSome content", encoding="utf-8")
            result = _render_md_file(p)
            self.assertIn("Hello", result)
            self.assertIn("test.md", result)

    def test_truncates_long_files(self):
        print("\n  ▶ TestRenderMdFile.test_truncates_long_files")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "big.md"
            p.write_text("x" * 10000, encoding="utf-8")
            result = _render_md_file(p)
            self.assertIn("TRUNCATED", result)

    def test_does_not_truncate_claude_md(self):
        print("\n  ▶ TestRenderMdFile.test_does_not_truncate_claude_md")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "CLAUDE.md"
            p.write_text("y" * 8000, encoding="utf-8")
            result = _render_md_file(p)
            self.assertNotIn("TRUNCATED", result)

    def test_missing_file_returns_empty(self):
        print("\n  ▶ TestRenderMdFile.test_missing_file_returns_empty")
        result = _render_md_file(Path("/nonexistent/path/file.md"))
        self.assertEqual(result, "")

    def test_empty_file_returns_empty(self):
        print("\n  ▶ TestRenderMdFile.test_empty_file_returns_empty")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "empty.md"
            p.write_text("", encoding="utf-8")
            result = _render_md_file(p)
            self.assertEqual(result, "")

    def test_large_non_core_markdown_is_compacted(self):
        print("\n  ▶ TestRenderMdFile.test_large_non_core_markdown_is_compacted")
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "guide.md"
            p.write_text(
                "# Guide\n\n"
                + "Intro paragraph.\n\n"
                + "\n".join(f"- item {i}" for i in range(80))
                + "\n\n## Details\n\n"
                + ("Long body line.\n" * 200),
                encoding="utf-8",
            )
            rendered = _render_md_file(p)
        self.assertIn("<!--", rendered)
        self.assertIn("# Guide", rendered)


# ---------------------------------------------------------------------------
# _refresh_tree_only
# ---------------------------------------------------------------------------

class TestRefreshTreeOnly(unittest.TestCase):

    def test_refresh_tree_only(self):
        print("\n  ▶ TestRefreshTreeOnly.test_refresh_tree_only")
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("foo.py").write_text("x=1", encoding="utf-8")
                result = _refresh_tree_only()
                self.assertIn("Directory structure:", result)
            finally:
                os.chdir(orig)


if __name__ == "__main__":
    unittest.main()
