"""
Tests for claude_light.indexer — _chunks_for_file, _debounce, _remove_file_from_index,
_load_cache, reindex_file, index_files.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.indexer import _chunks_for_file, _debounce


class TestChunksForFile(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        self._orig_store = dict(cl.chunk_store)
        cl.chunk_store.clear()
        cl.chunk_store.update({
            "src/Foo.java": {"text": "whole", "emb": None},
            "src/Foo.java::methodA": {"text": "a", "emb": None},
            "src/Foo.java::methodB": {"text": "b", "emb": None},
            "src/Bar.java": {"text": "whole2", "emb": None},
            "src/Bar.java::doThing": {"text": "c", "emb": None},
            "Foo.java": {"text": "class Foo {}", "emb": None},
            "FooBar.java": {"text": "class FooBar {}", "emb": None},
            "Foo.java::method1": {"text": "void m1() {}", "emb": None},
        })

    def tearDown(self):
        import claude_light as cl
        cl.chunk_store.clear()
        cl.chunk_store.update(self._orig_store)

    def test_returns_whole_file_and_methods(self):
        print("\n  ▶ TestChunksForFile.test_returns_whole_file_and_methods")
        result = _chunks_for_file("src/Foo.java")
        self.assertIn("src/Foo.java", result)
        self.assertIn("src/Foo.java::methodA", result)
        self.assertIn("src/Foo.java::methodB", result)
        self.assertNotIn("src/Bar.java", result)

    def test_returns_only_matching_file(self):
        print("\n  ▶ TestChunksForFile.test_returns_only_matching_file")
        result = _chunks_for_file("src/Bar.java")
        self.assertIn("src/Bar.java", result)
        self.assertIn("src/Bar.java::doThing", result)
        self.assertNotIn("src/Foo.java", result)

    def test_returns_empty_for_unknown_file(self):
        print("\n  ▶ TestChunksForFile.test_returns_empty_for_unknown_file")
        result = _chunks_for_file("src/Unknown.java")
        self.assertEqual(result, [])

    def test_does_not_match_prefix_overlap(self):
        print("\n  ▶ TestChunksForFile.test_does_not_match_prefix_overlap")
        result = _chunks_for_file("Foo.java")
        self.assertNotIn("FooBar.java", result)
        self.assertIn("Foo.java", result)
        self.assertIn("Foo.java::method1", result)


class TestDebounce(unittest.TestCase):

    def test_debounce_cancels_previous_and_schedules_new(self):
        print("\n  ▶ TestDebounce.test_debounce_cancels_previous_and_schedules_new")
        import claude_light as cl
        calls = []
        fn = lambda: calls.append(1)
        _debounce("test_key", fn, delay=0.001)
        self.assertIn("test_key", cl._file_timers)
        cl._file_timers.pop("test_key").cancel()

    def test_debounce_replaces_existing(self):
        print("\n  ▶ TestDebounce.test_debounce_replaces_existing")
        import claude_light as cl
        fn1 = lambda: None
        fn2 = lambda: None
        _debounce("replace_key", fn1, delay=60)
        timer1 = cl._file_timers.get("replace_key")
        _debounce("replace_key", fn2, delay=60)
        timer2 = cl._file_timers.get("replace_key")
        self.assertIsNot(timer1, timer2)
        cl._file_timers.pop("replace_key").cancel()


class TestRemoveFileFromIndex(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        self._orig_store = dict(cl.chunk_store)
        self._orig_hashes = dict(cl._file_hashes)
        cl.chunk_store.clear()
        cl.chunk_store.update({
            "src/X.java": {"text": "class X {}", "emb": None},
            "src/X.java::m1": {"text": "void m1() {}", "emb": None},
        })
        cl._file_hashes["src/X.java"] = "abc"

    def tearDown(self):
        import claude_light as cl
        cl.chunk_store.clear()
        cl.chunk_store.update(self._orig_store)
        cl._file_hashes.clear()
        cl._file_hashes.update(self._orig_hashes)

    def test_removes_all_chunks_for_file(self):
        print("\n  ▶ TestRemoveFileFromIndex.test_removes_all_chunks_for_file")
        from claude_light import _remove_file_from_index
        import claude_light as cl
        # Patch the indexer's own _save_cache binding — see the twin fix in
        # test_claude_light.py. Patching claude_light._save_cache (the alias
        # in __init__.py) is a no-op because _remove_file_from_index resolves
        # the symbol against its own module, so the real project cache would
        # be written on every pytest run.
        with patch("claude_light.indexer._save_cache"):
            _remove_file_from_index("src/X.java")
        self.assertNotIn("src/X.java", cl.chunk_store)
        self.assertNotIn("src/X.java::m1", cl.chunk_store)
        self.assertNotIn("src/X.java", cl._file_hashes)


class TestLoadCache(unittest.TestCase):

    def test_load_cache_missing_files(self):
        print("\n  ▶ TestLoadCache.test_load_cache_missing_files")
        from claude_light import _load_cache
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                mock_files = [MagicMock(spec=Path)]
                mock_files[0].__str__ = lambda self: "fake.py"
                cached, stale = _load_cache(mock_files, "all-MiniLM-L6-v2", {}, quiet=True)
                self.assertEqual(cached, {})
                self.assertEqual(stale, mock_files)
            finally:
                os.chdir(orig)

    def test_load_cache_embed_model_mismatch(self):
        print("\n  ▶ TestLoadCache.test_load_cache_embed_model_mismatch")
        from claude_light import _load_cache
        import claude_light as cl
        import json
        import pickle
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                cl.CACHE_DIR.mkdir(exist_ok=True)
                cl.CACHE_MANIFEST.write_text(
                    json.dumps({"embed_model": "old-model", "files": {}}),
                    encoding="utf-8"
                )
                cl.CACHE_INDEX.write_bytes(pickle.dumps({}))
                cached, stale = _load_cache([], "new-model", {}, quiet=True)
                self.assertEqual(cached, {})
            finally:
                os.chdir(orig)
                # NOTE: do NOT unlink CACHE_MANIFEST/INDEX here — those are
                # relative paths and os.chdir(orig) already ran, so unlinking
                # would nuke the real project's .claude_light_cache/ files.
                # TemporaryDirectory() cleans up the in-tmpdir copies on exit.


class TestReindexFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = os.getcwd()
        os.chdir(self.tmpdir)
        import claude_light as cl
        import numpy as np
        self._orig_store = dict(cl.chunk_store)
        self._orig_embed = cl.EMBED_MODEL
        self._orig_embedder = cl.embedder
        self._orig_hashes = dict(cl._file_hashes)
        cl.chunk_store.clear()
        cl.EMBED_MODEL = "all-MiniLM-L6-v2"
        cl.chunk_store["module.py"] = {"text": "old code", "emb": np.zeros(3)}

    def tearDown(self):
        os.chdir(self.orig)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import claude_light as cl
        cl.chunk_store.clear()
        cl.chunk_store.update(self._orig_store)
        cl.EMBED_MODEL = self._orig_embed
        cl.embedder = self._orig_embedder
        cl._file_hashes.clear()
        cl._file_hashes.update(self._orig_hashes)

    def test_reindex_file_no_embedder(self):
        print("\n  ▶ TestReindexFile.test_reindex_file_no_embedder")
        from claude_light import reindex_file
        import claude_light as cl
        cl.embedder = None
        reindex_file("module.py")  # Should return silently

    def test_reindex_file_with_embedder(self):
        print("\n  ▶ TestReindexFile.test_reindex_file_with_embedder")
        from claude_light import reindex_file
        import claude_light as cl
        import numpy as np
        Path("module.py").write_text("def foo(): return 1\n", encoding="utf-8")
        mock_emb = MagicMock()
        mock_emb.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        cl.embedder = mock_emb
        with patch("claude_light._save_cache"), patch("builtins.print"):
            reindex_file("module.py")
        keys = list(cl.chunk_store.keys())
        self.assertTrue(any("module.py" in k for k in keys))

    def test_reindex_file_exception_no_crash(self):
        print("\n  ▶ TestReindexFile.test_reindex_file_exception_no_crash")
        from claude_light import reindex_file
        import claude_light as cl
        cl.embedder = MagicMock()
        with patch("builtins.print"):
            reindex_file("nonexistent.py")


class TestWatcherPaths(unittest.TestCase):
    """Watchdog reports './a.py' (Windows: '.\\\\a.py'); the index keys files as
    'a.py'. Live, a re-index after an edit left the old chunks in place under
    'a.py::f' and added new ones under '.\\\\a.py::f', so retrieval could return
    the stale code, and an edited .md file was added to the skeleton twice."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = os.getcwd()
        os.chdir(self.tmpdir)
        import claude_light.state as st
        self._store = dict(st.chunk_store)
        self._embedder = st.embedder
        self._hashes = dict(st._file_hashes)
        st.chunk_store.clear()

    def tearDown(self):
        os.chdir(self.orig)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import claude_light.state as st
        st.chunk_store.clear()
        st.chunk_store.update(self._store)
        st.embedder = self._embedder
        st._file_hashes.clear()
        st._file_hashes.update(self._hashes)

    def test_reindex_through_a_dot_path_replaces_the_old_chunks(self):
        print("\n  ▶ TestWatcherPaths.test_reindex_through_a_dot_path_replaces_the_old_chunks")
        import numpy as np
        import claude_light.state as st
        from claude_light.indexer import reindex_file
        Path("calc.py").write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
        st.chunk_store["calc.py::divide"] = {"text": "old", "emb": np.zeros(3)}
        st.embedder = MagicMock()
        st.embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        with patch("claude_light.indexer._save_cache"):
            reindex_file(os.path.join(".", "calc.py"))
        self.assertEqual(sorted(st.chunk_store), ["calc.py::divide"])
        self.assertNotEqual(st.chunk_store["calc.py::divide"]["text"], "old")

    def test_handler_passes_project_relative_paths(self):
        print("\n  ▶ TestWatcherPaths.test_handler_passes_project_relative_paths")
        from claude_light import indexer
        seen = []
        event = MagicMock(is_directory=False, src_path=os.path.join(".", "pkg", "mod.py"))
        with patch.object(indexer, "_debounce", side_effect=lambda src, fn: seen.append(src)):
            indexer.SourceHandler().on_modified(event)
        self.assertEqual(seen, [os.path.join("pkg", "mod.py")])

    def test_handler_ignores_skipped_directories(self):
        print("\n  ▶ TestWatcherPaths.test_handler_ignores_skipped_directories")
        # A pip install into .venv or a build under node_modules re-embedded
        # those files into the project index.
        from claude_light import indexer
        seen = []
        with patch.object(indexer, "_debounce", side_effect=lambda src, fn: seen.append(src)), \
             patch.object(indexer, "_remove_file_from_index") as removed:
            for parts in ((".venv", "lib", "x.py"), ("node_modules", "a", "b.js"),
                          (".claude_light_cache", "y.py"), (".git", "z.py")):
                ev = MagicMock(is_directory=False, src_path=os.path.join(".", *parts))
                indexer.SourceHandler().on_modified(ev)
                indexer.SourceHandler().on_deleted(ev)
        self.assertEqual(seen, [])
        removed.assert_not_called()


class TestIndexFilesLight(unittest.TestCase):

    def test_no_source_files_no_crash(self):
        print("\n  ▶ TestIndexFilesLight.test_no_source_files_no_crash")
        from claude_light import index_files
        import claude_light as cl
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                orig_store = dict(cl.chunk_store)
                with patch("builtins.print"):
                    index_files(quiet=False)
                cl.chunk_store.clear()
                cl.chunk_store.update(orig_store)
            finally:
                os.chdir(orig)

    def test_indexes_python_file(self):
        print("\n  ▶ TestIndexFilesLight.test_indexes_python_file")
        from claude_light import index_files
        import claude_light as cl
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            os.chdir(tmpdir)
            try:
                Path("module.py").write_text("def hello(): return 'hi'\n", encoding="utf-8")
                mock_embedder = MagicMock()
                mock_embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                orig_embedder = cl.embedder
                orig_model = cl.EMBED_MODEL
                orig_store = dict(cl.chunk_store)
                with patch("claude_light.SentenceTransformer", return_value=mock_embedder), \
                     patch("builtins.print"):
                    cl.EMBED_MODEL = None
                    cl.embedder = None
                    index_files(quiet=True)
                self.assertGreater(len(cl.chunk_store), 0)
                cl.chunk_store.clear()
                cl.chunk_store.update(orig_store)
                cl.EMBED_MODEL = orig_model
                cl.embedder = orig_embedder
            finally:
                os.chdir(orig)


if __name__ == "__main__":
    unittest.main()


class TestBackgroundNotices(unittest.TestCase):
    """Background threads (the file watcher, the cache heartbeat) used to
    print: live, 'Re-indexed calc.py' landed between the tool lines and the
    reply, and a print while the prompt is up draws over what is being typed.
    They post a notice for the toolbar instead."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig = os.getcwd()
        os.chdir(self.tmpdir)
        import claude_light.state as st
        self._store = dict(st.chunk_store)
        self._embedder = st.embedder
        st.notice = None

    def tearDown(self):
        os.chdir(self.orig)
        import claude_light.state as st
        st.chunk_store.clear()
        st.chunk_store.update(self._store)
        st.embedder = self._embedder
        st.notice = None

    def test_reindex_posts_a_notice_and_prints_nothing(self):
        print("\n  ▶ TestBackgroundNotices.test_reindex_posts_a_notice_and_prints_nothing")
        import io
        import numpy as np
        import claude_light.state as st
        from claude_light.indexer import reindex_file
        from claude_light.ui import current_notice
        Path("calc.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        st.embedder = MagicMock()
        st.embedder.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        out = io.StringIO()
        with patch("claude_light.indexer._save_cache"), patch("sys.stdout", out):
            reindex_file(os.path.join(".", "calc.py"))
        self.assertEqual(out.getvalue(), "")
        self.assertIn("calc.py", current_notice())

    def test_notice_expires(self):
        print("\n  ▶ TestBackgroundNotices.test_notice_expires")
        from claude_light import ui
        with patch("claude_light.ui.time.monotonic", return_value=100.0):
            ui.notify("re-indexed a.py")
            self.assertEqual(ui.current_notice(), "re-indexed a.py")
        with patch("claude_light.ui.time.monotonic", return_value=100.0 + ui.NOTICE_SECS + 1):
            self.assertEqual(ui.current_notice(), "")

    def test_toolbar_shows_the_notice(self):
        print("\n  ▶ TestBackgroundNotices.test_toolbar_shows_the_notice")
        import claude_light.main as m
        from claude_light import ui
        ui.notify("re-indexed calc.py (2 chunks)")
        self.assertIn("re-indexed calc.py", m._status_text(columns=200))

    def test_toolbar_fits_a_narrow_terminal(self):
        print("\n  ▶ TestBackgroundNotices.test_toolbar_fits_a_narrow_terminal")
        # Live at 1300 px the hints were cut mid-word ("alt+enter").
        import re
        import claude_light.main as m
        from claude_light import ui
        import claude_light.state as st
        # A long session: the status alone is wider than 50 columns.
        saved_tokens, saved_cost = dict(st.session_tokens), st.session_cost
        self.addCleanup(lambda: (st.session_tokens.update(saved_tokens), setattr(st, "session_cost", saved_cost)))
        st.session_tokens.update(input=1_234_567, cache_write=2_345_678, cache_read=3_456_789)
        st.session_cost = 12.3456
        if not st.indexing_done.is_set():
            self.addCleanup(st.indexing_done.clear)
        st.indexing_done.set()
        ui.notify("re-indexed claude_light/indexer.py (31 chunks)")
        plain = lambda html: re.sub(r"<[^>]+>", "", html).replace("&lt;", "<").replace("&amp;", "&")
        wide = plain(m._status_text(columns=200))
        self.assertIn("alt+enter new line", wide)
        self.assertIn("re-indexed", wide)
        for columns in (100, 70, 50):
            narrow = plain(m._status_text(columns=columns))
            self.assertLessEqual(len(narrow), columns, narrow)
            self.assertIn(os.path.basename(os.getcwd())[:10], narrow)
            self.assertNotIn("alt+ent", narrow.replace("alt+enter new line", ""))  # never cut mid-hint

    def test_background_cache_warms_are_quiet(self):
        print("\n  ▶ TestBackgroundNotices.test_background_cache_warms_are_quiet")
        from claude_light import llm
        with patch.object(llm, "warm_cache") as warm, \
             patch.object(llm, "_refresh_tree_only", return_value="tree"), \
             patch.object(llm, "_refresh_single_md", return_value=True), \
             patch.object(llm, "_assemble_skeleton", return_value="sk"):
            llm.refresh_skeleton_only()
            llm.refresh_md_file("README.md")
        self.assertEqual([c.kwargs.get("quiet") for c in warm.call_args_list], [True, True])
