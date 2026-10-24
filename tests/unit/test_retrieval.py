"""
Tests for claude_light.retrieval — _dedup_retrieved_context, retrieve, adaptive selection.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.retrieval import _dedup_retrieved_context, record_retrieval, explain_last_retrieval
import claude_light.state as _state


class TestExplainLastRetrieval(unittest.TestCase):

    def setUp(self):
        self._orig = _state.last_retrieval

    def tearDown(self):
        _state.last_retrieval = self._orig

    def test_no_query_yet(self):
        print("\n  ▶ TestExplainLastRetrieval.test_no_query_yet")
        _state.last_retrieval = None
        self.assertIn("No query", explain_last_retrieval())

    def test_records_and_explains_hits(self):
        print("\n  ▶ TestExplainLastRetrieval.test_records_and_explains_hits")
        record_retrieval("what does X do?", "claude-sonnet-4-6", "high", 8192,
                         [("src/A.py::foo", 0.80), ("src/B.py::bar", 0.50)])
        out = explain_last_retrieval()
        self.assertIn("what does X do?", out)
        self.assertIn("claude-sonnet-4-6", out)
        self.assertIn("effort=high", out)
        self.assertIn("A.py::foo", out)
        self.assertIn("0.800", out)

    def test_skip_reason_shown(self):
        print("\n  ▶ TestExplainLastRetrieval.test_skip_reason_shown")
        record_retrieval("and that?", "claude-haiku-4-5", "low", 1500, [],
                         skipped_reason="clarification follow-up")
        out = explain_last_retrieval()
        self.assertIn("skipped", out.lower())
        self.assertIn("clarification follow-up", out)

    def test_compression_line_shown(self):
        print("\n  ▶ TestExplainLastRetrieval.test_compression_line_shown")
        record_retrieval("q", "m", "medium", 3000, [("f.py::g", 0.7)],
                         compression={"skipped": False, "tokens_before": 900,
                                      "tokens_after": 450, "ratio": 0.5})
        out = explain_last_retrieval()
        self.assertIn("900", out)
        self.assertIn("450", out)


class TestDedupRetrievedContext(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        self._orig_store = dict(cl.chunk_store)
        cl.chunk_store.clear()

    def tearDown(self):
        import claude_light as cl
        cl.chunk_store.clear()
        cl.chunk_store.update(self._orig_store)

    def test_empty_pairs(self):
        print("\n  ▶ TestDedupRetrievedContext.test_empty_pairs")
        result = _dedup_retrieved_context([])
        self.assertEqual(result, "")

    def test_whole_file_chunk(self):
        print("\n  ▶ TestDedupRetrievedContext.test_whole_file_chunk")
        import claude_light as cl
        import numpy as np
        cl.chunk_store["src/Foo.py"] = {"text": "def foo(): pass", "emb": np.zeros(10)}
        result = _dedup_retrieved_context([("src/Foo.py", 0.9)])
        self.assertIn("def foo(): pass", result)

    def test_method_chunk_deduplication(self):
        print("\n  ▶ TestDedupRetrievedContext.test_method_chunk_deduplication")
        import claude_light as cl
        import numpy as np
        preamble = "// src/Foo.java\npackage com.example;\npublic class Foo {"
        sep = "\n    // ...\n"
        method_a = "public void methodA() {}"
        method_b = "public void methodB() {}"
        cl.chunk_store["src/Foo.java::methodA"] = {
            "text": preamble + sep + method_a + "\n",
            "emb": np.zeros(10),
        }
        cl.chunk_store["src/Foo.java::methodB"] = {
            "text": preamble + sep + method_b + "\n",
            "emb": np.zeros(10),
        }
        result = _dedup_retrieved_context([
            ("src/Foo.java::methodA", 0.9),
            ("src/Foo.java::methodB", 0.85),
        ])
        self.assertEqual(result.count("package com.example;"), 1)
        self.assertIn("methodA", result)
        self.assertIn("methodB", result)

    def test_missing_chunk_id_skipped(self):
        print("\n  ▶ TestDedupRetrievedContext.test_missing_chunk_id_skipped")
        result = _dedup_retrieved_context([("nonexistent::method", 0.9)])
        self.assertEqual(result, "")

    def test_multiple_files_all_included(self):
        print("\n  ▶ TestDedupRetrievedContext.test_multiple_files_all_included")
        import claude_light as cl
        import numpy as np
        preamble_a = "// A.java\npackage a;\npublic class A {"
        preamble_b = "// B.java\npackage b;\npublic class B {"
        sep = "\n    // ...\n"
        cl.chunk_store["A.java::mA"] = {"text": preamble_a + sep + "void mA() {}", "emb": np.zeros(3)}
        cl.chunk_store["B.java::mB"] = {"text": preamble_b + sep + "void mB() {}", "emb": np.zeros(3)}
        result = _dedup_retrieved_context([("A.java::mA", 0.9), ("B.java::mB", 0.8)])
        self.assertIn("package a;", result)
        self.assertIn("package b;", result)
        self.assertIn("mA", result)
        self.assertIn("mB", result)

    def test_three_methods_same_file_single_preamble(self):
        print("\n  ▶ TestDedupRetrievedContext.test_three_methods_same_file_single_preamble")
        import claude_light as cl
        import numpy as np
        preamble = "// Svc.java\npackage svc;\npublic class Svc {"
        sep = "\n    // ...\n"
        for name in ["alpha", "beta", "gamma"]:
            cl.chunk_store[f"Svc.java::{name}"] = {
                "text": preamble + sep + f"void {name}() {{}}",
                "emb": np.zeros(3),
            }
        result = _dedup_retrieved_context([
            ("Svc.java::alpha", 0.9),
            ("Svc.java::beta", 0.8),
            ("Svc.java::gamma", 0.7),
        ])
        self.assertEqual(result.count("package svc;"), 1)
        for name in ["alpha", "beta", "gamma"]:
            self.assertIn(name, result)


class TestDefaultScoreFloor(unittest.TestCase):
    """With all-mpnet-base-v2, a natural question scores lower against code
    than a keyword query: live, "In which file and function is the
    exponential backoff implemented?" peaked at 0.38 on retry.py, while
    "exponential backoff retry" scored 0.70. The old 0.45 floor dropped it,
    and 13 of 19 natural questions about this repo got no context at all."""

    def setUp(self):
        import claude_light as cl
        import claude_light.config as config
        import claude_light.state as _st
        self._saved = (dict(cl.chunk_store), cl.TOP_K, cl.EMBED_MODEL, cl.embedder,
                       cl.MIN_SCORE, cl.RELATIVE_SCORE_FLOOR)
        cl.MIN_SCORE = config.MIN_SCORE              # the shipped defaults
        cl.RELATIVE_SCORE_FLOOR = config.RELATIVE_SCORE_FLOOR
        cl.TOP_K = 5
        cl.EMBED_MODEL = "all-mpnet-base-v2"
        self._was_ready = _st.embedder_ready.is_set()
        _st.embedder_ready.set()

    def tearDown(self):
        import claude_light as cl
        import claude_light.state as _st
        store, cl.TOP_K, cl.EMBED_MODEL, cl.embedder, cl.MIN_SCORE, cl.RELATIVE_SCORE_FLOOR = self._saved
        cl.chunk_store.clear()
        cl.chunk_store.update(store)
        (_st.embedder_ready.set if self._was_ready else _st.embedder_ready.clear)()

    def test_question_scoring_038_on_the_right_chunk_is_retrieved(self):
        print("\n  ▶ TestDefaultScoreFloor.test_question_scoring_038_on_the_right_chunk_is_retrieved")
        import math
        import claude_light as cl
        import numpy as np
        right = np.array([0.38, math.sqrt(1 - 0.38 ** 2), 0.0])
        wrong = np.array([0.10, 0.0, math.sqrt(1 - 0.10 ** 2)])
        cl.chunk_store.clear()
        cl.chunk_store.update({
            "claude_light/retry.py::retry_with_backoff": {"text": "def retry_with_backoff(): pass", "emb": right},
            "claude_light/ui.py::print_stats": {"text": "def print_stats(): pass", "emb": wrong},
        })
        cl.embedder = MagicMock()
        cl.embedder.encode.return_value = np.array([1.0, 0.0, 0.0])
        _, hits = cl.retrieve("In which file and function is the exponential backoff implemented?")
        self.assertEqual([h[0] for h in hits], ["claude_light/retry.py::retry_with_backoff"])


class TestRetrieve(unittest.TestCase):

    def setUp(self):
        import claude_light as cl
        import numpy as np
        self._orig_store = dict(cl.chunk_store)
        self._orig_top_k = cl.TOP_K
        self._orig_embed = cl.EMBED_MODEL
        self._orig_embedder = cl.embedder
        self._orig_min_score = cl.MIN_SCORE
        self._orig_rel_floor = cl.RELATIVE_SCORE_FLOOR

        emb_a = np.array([1.0, 0.0, 0.0])
        emb_b = np.array([0.0, 1.0, 0.0])
        emb_c = np.array([0.5, 0.5, 0.0])
        cl.chunk_store.clear()
        cl.chunk_store.update({
            "src/A.py::func_a": {"text": "def func_a(): pass", "emb": emb_a / 1.0},
            "src/B.py::func_b": {"text": "def func_b(): pass", "emb": emb_b / 1.0},
            "src/C.py::func_c": {"text": "def func_c(): pass", "emb": emb_c / 1.0},
        })
        cl.TOP_K = 5
        cl.EMBED_MODEL = "all-MiniLM-L6-v2"
        cl.MIN_SCORE = 0.0
        cl.RELATIVE_SCORE_FLOOR = 0.0
        import claude_light.state as _st
        _st.embedder_ready.set()

    def tearDown(self):
        import claude_light as cl
        import claude_light.state as _st
        cl.chunk_store.clear()
        cl.chunk_store.update(self._orig_store)
        cl.TOP_K = self._orig_top_k
        cl.EMBED_MODEL = self._orig_embed
        cl.embedder = self._orig_embedder
        cl.MIN_SCORE = self._orig_min_score
        cl.RELATIVE_SCORE_FLOOR = self._orig_rel_floor
        _st.embedder_ready.clear()

    def test_retrieve_returns_context_and_hits(self):
        print("\n  ▶ TestRetrieve.test_retrieve_returns_context_and_hits")
        import claude_light as cl
        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([1.0, 0.0, 0.0])
        cl.embedder = mock_embedder
        ctx, hits = cl.retrieve("some query")
        self.assertIsInstance(ctx, str)
        self.assertIsInstance(hits, list)

    def test_retrieve_empty_store_returns_empty(self):
        print("\n  ▶ TestRetrieve.test_retrieve_empty_store_returns_empty")
        import claude_light as cl
        cl.chunk_store.clear()
        mock_embedder = MagicMock()
        cl.embedder = mock_embedder
        ctx, hits = cl.retrieve("some query")
        self.assertEqual(ctx, "")
        self.assertEqual(hits, [])

    def test_retrieve_filters_low_scores(self):
        print("\n  ▶ TestRetrieve.test_retrieve_filters_low_scores")
        import claude_light as cl
        import numpy as np
        cl.MIN_SCORE = 0.5
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([0.0, 0.0, 1.0])
        cl.embedder = mock_embedder
        ctx, hits = cl.retrieve("unrelated query")
        self.assertEqual(hits, [])
        self.assertEqual(ctx, "")

    def test_relative_floor_drops_low_chunks(self):
        print("\n  ▶ TestRetrieve.test_relative_floor_drops_low_chunks")
        import claude_light as cl
        import numpy as np
        cl.chunk_store.clear()
        cl.chunk_store["A.py"] = {"text": "a", "emb": np.array([0.9, 0.1]) / 1.0}
        cl.chunk_store["B.py"] = {"text": "b", "emb": np.array([0.1, 0.9]) / 1.0}
        mock_emb = MagicMock()
        mock_emb.encode.return_value = np.array([1.0, 0.0])
        cl.embedder = mock_emb
        cl.MIN_SCORE = 0.0
        cl.RELATIVE_SCORE_FLOOR = 0.9
        ctx, hits = cl.retrieve("test query")
        hit_ids = [cid for cid, _ in hits]
        self.assertNotIn("B.py", hit_ids)

    def test_low_effort_uses_summary_only_context(self):
        print("\n  ▶ TestRetrieve.test_low_effort_uses_summary_only_context")
        import claude_light as cl
        import numpy as np
        cl.chunk_store.clear()
        cl.chunk_store["src/a.py::alpha"] = {
            "text": "// src/a.py\nfrom a import b\n\n    // ...\ndef alpha(x):\n    return x + 1\n",
            "emb": np.array([1.0, 0.0, 0.0]),
        }
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([1.0, 0.0, 0.0])
        cl.embedder = mock_embedder
        ctx, hits = cl.retrieve("where is alpha", token_budget=2000, effort="low")
        self.assertTrue(hits)
        self.assertIn("Relevant Files:", ctx)

    def test_adaptive_selection_diverse_hits(self):
        print("\n  ▶ TestRetrieve.test_adaptive_selection_diverse_hits")
        import numpy as np
        import claude_light as cl
        from claude_light.retrieval import _adaptive_select_pairs

        orig_store = dict(cl.chunk_store)
        try:
            cl.chunk_store.clear()
            cl.chunk_store.update({
                "src/a.py::one": {"text": "x" * 400, "emb": np.array([1.0, 0.0])},
                "src/a.py::two": {"text": "y" * 400, "emb": np.array([0.99, 0.0])},
                "src/b.py::three": {"text": "z" * 120, "emb": np.array([0.97, 0.0])},
            })
            ids = list(cl.chunk_store.keys())
            scores = np.array([1.0, 0.99, 0.97])
            selected = _adaptive_select_pairs(ids, scores, budget=220, k=5)
            selected_files = {cid.split("::", 1)[0] for cid, _ in selected}
            self.assertIn("src/a.py", selected_files)
            self.assertIn("src/b.py", selected_files)
        finally:
            cl.chunk_store.clear()
            cl.chunk_store.update(orig_store)


if __name__ == "__main__":
    unittest.main()
