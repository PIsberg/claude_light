"""
Tests for claude_light.callgraph, the fetch_chunk tool, and call-graph expansion
in retrieval.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light.state as state
from claude_light import callgraph
from claude_light.retrieval import _related_pairs, _finish
from claude_light.tools import run_tool

_SEP = "\n    // ...\n"


def _chunk(path, name, body):
    return {"id": f"{path}::{name}", "text": f"// {path}\nimport os{_SEP}{body}\n"}


FIXTURE = [
    _chunk("app/api.py", "handle", "def handle(req):\n    user = load_user(req)\n    return render(user)"),
    _chunk("app/api.py", "load_user", "def load_user(req):\n    return validate(req.id)"),
    _chunk("app/api.py", "render", "def render(user):\n    return str(user)"),
    _chunk("app/auth.py", "validate", "def validate(uid):\n    return uid > 0"),
    _chunk("app/cli.py", "main", "def main():\n    validate(1)\n    get()"),
    # 'get' is defined in four files: too generic to follow
    _chunk("a.py", "get", "def get():\n    pass"),
    _chunk("b.py", "get", "def get():\n    pass"),
    _chunk("c.py", "get", "def get():\n    pass"),
    _chunk("d.py", "get", "def get():\n    pass"),
]


class _StoreCase(unittest.TestCase):
    def setUp(self):
        with state.lock:
            self._orig = dict(state.chunk_store)
            state.chunk_store.clear()
            for c in FIXTURE:
                state.chunk_store[c["id"]] = {"text": c["text"], "emb": np.zeros(2)}
        callgraph._names_cache.clear()

    def tearDown(self):
        with state.lock:
            state.chunk_store.clear()
            state.chunk_store.update(self._orig)
        callgraph._names_cache.clear()


class TestCallGraph(_StoreCase):

    def test_called_names_skips_keywords_and_header(self):
        print("\n  ▶ TestCallGraph.test_called_names_skips_keywords_and_header")
        names = callgraph.called_names(FIXTURE[0]["text"])
        self.assertEqual(names, {"handle", "load_user", "render"})  # def line counts; `if` etc. do not
        self.assertNotIn("import", names)

    def test_callees_and_callers(self):
        print("\n  ▶ TestCallGraph.test_callees_and_callers")
        callees, callers = callgraph.related_chunks("app/api.py::handle")
        self.assertEqual(callees, ["app/api.py::load_user", "app/api.py::render"])
        self.assertEqual(callers, [])
        callees, callers = callgraph.related_chunks("app/auth.py::validate")
        self.assertEqual(callees, [])
        self.assertEqual(callers, ["app/api.py::load_user", "app/cli.py::main"])

    def test_generic_names_are_not_followed(self):
        print("\n  ▶ TestCallGraph.test_generic_names_are_not_followed")
        callees, _ = callgraph.related_chunks("app/cli.py::main")
        self.assertEqual(callees, ["app/auth.py::validate"])

    def test_unknown_or_whole_file_id(self):
        print("\n  ▶ TestCallGraph.test_unknown_or_whole_file_id")
        self.assertEqual(callgraph.related_chunks("nope::x"), ([], []))
        self.assertEqual(callgraph.related_chunks("app/api.py"), ([], []))

    def test_edit_invalidates_cache(self):
        print("\n  ▶ TestCallGraph.test_edit_invalidates_cache")
        self.assertEqual(callgraph.related_chunks("app/api.py::render")[0], [])
        with state.lock:
            state.chunk_store["app/api.py::render"]["text"] = \
                _chunk("app/api.py", "render", "def render(u):\n    return load_user(u)")["text"]
        self.assertEqual(callgraph.related_chunks("app/api.py::render")[0], ["app/api.py::load_user"])

    def test_interleaved_alternates_callee_first(self):
        print("\n  ▶ TestCallGraph.test_interleaved_alternates_callee_first")
        with state.lock:
            state.chunk_store["app/api.py::load_user"]["text"] = \
                _chunk("app/api.py", "load_user", "def load_user(r):\n    return render(r)\n    validate(r)")["text"]
        out = callgraph.interleaved("app/api.py::load_user")
        self.assertEqual(out[0], "app/api.py::render")           # callee
        self.assertIn("app/api.py::handle", out)                  # caller
        self.assertEqual(len(out), len(set(out)))


class TestFetchChunk(_StoreCase):

    def fetch(self, ids):
        return run_tool("fetch_chunk", {"ids": ids})

    def test_exact_id_returns_body_without_preamble(self):
        print("\n  ▶ TestFetchChunk.test_exact_id_returns_body_without_preamble")
        out, is_error = self.fetch(["app/auth.py::validate"])
        self.assertFalse(is_error)
        self.assertIn("def validate(uid)", out)
        self.assertNotIn("import os", out)
        self.assertIn("Called by: app/api.py::load_user, app/cli.py::main", out)

    def test_bare_symbol_and_backslash_path(self):
        print("\n  ▶ TestFetchChunk.test_bare_symbol_and_backslash_path")
        self.assertIn("def render(user)", self.fetch(["render"])[0])
        with state.lock:
            cid = "app\\auth.py::validate"
            state.chunk_store[cid] = state.chunk_store.pop("app/auth.py::validate")
        self.assertIn("def validate(uid)", self.fetch(["app/auth.py::validate"])[0])

    def test_ambiguous_and_missing(self):
        print("\n  ▶ TestFetchChunk.test_ambiguous_and_missing")
        out, is_error = self.fetch(["get", "zzz"])
        self.assertFalse(is_error)
        self.assertIn("Ambiguous", out)
        self.assertIn("a.py::get", out)
        self.assertIn("No such chunk", out)

    def test_bad_input_is_an_error(self):
        print("\n  ▶ TestFetchChunk.test_bad_input_is_an_error")
        self.assertTrue(run_tool("fetch_chunk", {})[1])
        self.assertTrue(run_tool("fetch_chunk", {"ids": []})[1])
        self.assertTrue(run_tool("fetch_chunk", {"ids": [3]})[1])

    def test_empty_index_is_an_error(self):
        print("\n  ▶ TestFetchChunk.test_empty_index_is_an_error")
        with state.lock:
            state.chunk_store.clear()
        out, is_error = self.fetch(["x"])
        self.assertTrue(is_error)
        self.assertIn("read_file", out)

    def test_caps_number_of_ids(self):
        print("\n  ▶ TestFetchChunk.test_caps_number_of_ids")
        out, _ = self.fetch(["render"] * 9)
        self.assertEqual(out.count("## "), 5)


class TestRelatedExpansion(_StoreCase):

    def test_adds_neighbours_within_budget(self):
        print("\n  ▶ TestRelatedExpansion.test_adds_neighbours_within_budget")
        top = [("app/api.py::handle", 0.9)]
        related = _related_pairs(top, 10_000, "medium")
        self.assertEqual([c for c, _ in related], ["app/api.py::load_user", "app/api.py::render"])

    def test_respects_budget(self):
        print("\n  ▶ TestRelatedExpansion.test_respects_budget")
        top = [("app/api.py::handle", 0.9)]
        self.assertEqual(_related_pairs(top, 1, "medium"), [])

    def test_skipped_at_low_effort_and_when_disabled(self):
        print("\n  ▶ TestRelatedExpansion.test_skipped_at_low_effort_and_when_disabled")
        top = [("app/api.py::handle", 0.9)]
        self.assertEqual(_related_pairs(top, 10_000, "low"), [])
        with patch("claude_light.config.CALL_GRAPH", False):
            self.assertEqual(_related_pairs(top, 10_000, "high"), [])

    def test_does_not_repeat_selected_chunks(self):
        print("\n  ▶ TestRelatedExpansion.test_does_not_repeat_selected_chunks")
        top = [("app/api.py::handle", 0.9), ("app/api.py::render", 0.8)]
        related = _related_pairs(top, 10_000, "medium")
        self.assertEqual([c for c, _ in related], ["app/api.py::load_user"])

    def test_related_bodies_are_rendered(self):
        print("\n  ▶ TestRelatedExpansion.test_related_bodies_are_rendered")
        ctx, pairs = _finish([("app/api.py::handle", 0.9)], 10_000, "medium")
        self.assertIn("def load_user(req)", ctx)
        self.assertIn("def render(user)", ctx)
        self.assertEqual(len(pairs), 3)


if __name__ == "__main__":
    unittest.main()
