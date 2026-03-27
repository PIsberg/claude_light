"""
Tests for claude_light.parsing — _strip_comments, _walk, _extract_symbol_name, chunk_file.
Also tests from claude_light.skeleton — _build_compressed_tree, _render_compressed_node.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.parsing import _strip_comments, _walk, _extract_symbol_name, chunk_file
from claude_light.skeleton import _build_compressed_tree, _render_compressed_node
from claude_light.indexer import _chunk_label


# ---------------------------------------------------------------------------
# _chunk_label
# ---------------------------------------------------------------------------

class TestChunkLabel(unittest.TestCase):

    def test_strips_directory_prefix(self):
        print("\n  ▶ TestChunkLabel.test_strips_directory_prefix")
        self.assertEqual(_chunk_label("src/Foo.java::doThing"), "Foo.java::doThing")
        self.assertEqual(_chunk_label("src/Foo.java"), "Foo.java")

    def test_nested_path_with_method(self):
        print("\n  ▶ TestChunkLabel.test_nested_path_with_method")
        result = _chunk_label("com/example/service/OrderService.java::placeOrder")
        self.assertEqual(result, "OrderService.java::placeOrder")

    def test_path_only(self):
        print("\n  ▶ TestChunkLabel.test_path_only")
        result = _chunk_label("path/to/file.py")
        self.assertEqual(result, "file.py")

    def test_double_colon_in_method(self):
        print("\n  ▶ TestChunkLabel.test_double_colon_in_method")
        result = _chunk_label("src/Foo.java::method")
        self.assertEqual(result, "Foo.java::method")


# ---------------------------------------------------------------------------
# _strip_comments
# ---------------------------------------------------------------------------

class TestStripComments(unittest.TestCase):

    def test_strip_python_comments(self):
        print("\n  ▶ TestStripComments.test_strip_python_comments")
        code = "def foo():\n    # this is a comment\n    return 42\n"
        stripped = _strip_comments(code, ".py")
        self.assertNotIn("# this is a comment", stripped)
        self.assertIn("def foo():", stripped)
        self.assertIn("return 42", stripped)

    def test_strip_c_style_comments(self):
        print("\n  ▶ TestStripComments.test_strip_c_style_comments")
        code = "void doThing() {\n    // line comment\n    /* block \n       comment */\n    println();\n}"
        stripped = _strip_comments(code, ".java")
        self.assertNotIn("line comment", stripped)
        self.assertNotIn("block", stripped)
        self.assertIn("void doThing() {", stripped)
        self.assertIn("println();", stripped)

    def test_rust_comments(self):
        print("\n  ▶ TestStripComments.test_rust_comments")
        code = "fn main() {\n    // comment\n    println!(\"hi\"); /* block */\n}"
        result = _strip_comments(code, ".rs")
        self.assertNotIn("comment", result)
        self.assertNotIn("block", result)
        self.assertIn("println!", result)

    def test_go_comments(self):
        print("\n  ▶ TestStripComments.test_go_comments")
        code = "func main() {\n    // go comment\n    fmt.Println()\n}"
        result = _strip_comments(code, ".go")
        self.assertNotIn("go comment", result)
        self.assertIn("fmt.Println()", result)

    def test_unknown_extension_unchanged(self):
        print("\n  ▶ TestStripComments.test_unknown_extension_unchanged")
        code = "# This is some text\nwith content"
        result = _strip_comments(code, ".txt")
        self.assertIn("# This is some text", result)

    def test_collapses_excessive_blank_lines(self):
        print("\n  ▶ TestStripComments.test_collapses_excessive_blank_lines")
        code = "a = 1\n\n\n\n\nb = 2"
        result = _strip_comments(code, ".py")
        self.assertNotIn("\n\n\n", result)

    def test_ts_comments(self):
        print("\n  ▶ TestStripComments.test_ts_comments")
        code = "const x: number = 1; // ts comment\n/* block comment */\nconst y = 2;"
        result = _strip_comments(code, ".ts")
        self.assertNotIn("ts comment", result)
        self.assertNotIn("block comment", result)
        self.assertIn("const x", result)

    def test_tsx_comments(self):
        print("\n  ▶ TestStripComments.test_tsx_comments")
        code = "// tsx comment\nconst App = () => <div/>;"
        result = _strip_comments(code, ".tsx")
        self.assertNotIn("tsx comment", result)
        self.assertIn("const App", result)


# ---------------------------------------------------------------------------
# _walk
# ---------------------------------------------------------------------------

class TestWalk(unittest.TestCase):

    def test_walk_collects_matching_nodes(self):
        print("\n  ▶ TestWalk.test_walk_collects_matching_nodes")
        root = MagicMock()
        child1 = MagicMock()
        child1.type = "function_definition"
        child1.children = []
        child2 = MagicMock()
        child2.type = "class_definition"
        child2.children = [child1]
        root.type = "module"
        root.children = [child2]

        results = []
        _walk(root, ["function_definition"], results)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0], child1)

    def test_walk_does_not_recurse_into_match(self):
        print("\n  ▶ TestWalk.test_walk_does_not_recurse_into_match")
        outer = MagicMock()
        outer.type = "function_definition"
        inner = MagicMock()
        inner.type = "function_definition"
        inner.children = []
        outer.children = [inner]

        results = []
        _walk(outer, ["function_definition"], results)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0], outer)


# ---------------------------------------------------------------------------
# _extract_symbol_name
# ---------------------------------------------------------------------------

class TestExtractSymbolName(unittest.TestCase):

    def test_identifier_child(self):
        print("\n  ▶ TestExtractSymbolName.test_identifier_child")
        node = MagicMock()
        node.type = "function_definition"
        child = MagicMock()
        child.type = "identifier"
        child.text = b"my_func"
        child2 = MagicMock()
        child2.type = "parameters"
        node.children = [child, child2]
        self.assertEqual(_extract_symbol_name(node), "my_func")

    def test_fallback_uses_line_number(self):
        print("\n  ▶ TestExtractSymbolName.test_fallback_uses_line_number")
        node = MagicMock()
        node.type = "unknown_node"
        node.start_point = (10, 0)
        node.children = []
        result = _extract_symbol_name(node)
        self.assertIn("10", result)

    def test_decorated_definition_delegates_to_inner(self):
        print("\n  ▶ TestExtractSymbolName.test_decorated_definition_delegates_to_inner")
        inner_func = MagicMock()
        inner_func.type = "function_definition"
        id_child = MagicMock()
        id_child.type = "identifier"
        id_child.text = b"decorated_fn"
        inner_func.children = [id_child]

        outer = MagicMock()
        outer.type = "decorated_definition"
        outer.children = [inner_func]

        result = _extract_symbol_name(outer)
        self.assertEqual(result, "decorated_fn")


# ---------------------------------------------------------------------------
# chunk_file
# ---------------------------------------------------------------------------

class TestChunkFile(unittest.TestCase):

    def test_unknown_extension_whole_file(self):
        print("\n  ▶ TestChunkFile.test_unknown_extension_whole_file")
        result = chunk_file("script.sh", "#!/bin/bash\necho hello\n")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "script.sh")
        self.assertIn("echo hello", result[0]["text"])

    def test_python_with_treesitter_or_fallback(self):
        print("\n  ▶ TestChunkFile.test_python_with_treesitter_or_fallback")
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        result = chunk_file("main.py", code)
        self.assertGreaterEqual(len(result), 1)
        for chunk in result:
            self.assertIn("id", chunk)
            self.assertIn("text", chunk)

    def test_python_file_produces_chunks(self):
        print("\n  ▶ TestChunkFile.test_python_file_produces_chunks")
        import claude_light as cl
        if not cl._TREESITTER_AVAILABLE:
            self.skipTest("tree-sitter not available")
        if ".py" not in cl._LANG_CONFIG or cl._LANG_CONFIG[".py"] is None:
            self.skipTest("tree-sitter-python not available")
        code = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        chunks = chunk_file("module.py", code)
        ids = [c["id"] for c in chunks]
        self.assertTrue(any("::" in cid for cid in ids))
        self.assertTrue(any("alpha" in cid for cid in ids))

    def test_no_duplicate_ids(self):
        print("\n  ▶ TestChunkFile.test_no_duplicate_ids")
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        chunks = chunk_file("util.py", code)
        ids = [c["id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# _build_compressed_tree
# ---------------------------------------------------------------------------

class TestBuildCompressedTree(unittest.TestCase):

    def test_typical_project_structure(self):
        print("\n  ▶ TestBuildCompressedTree.test_typical_project_structure")
        paths = [
            Path("src/main/java/com/example/UserService.java"),
            Path("src/main/java/com/example/OrderService.java"),
            Path("src/main/java/com/example/util/Helper.java"),
            Path("README.md"),
        ]
        tree = _build_compressed_tree(paths)
        self.assertIn("README.md", tree)
        self.assertIn("src/main/java/com/example/", tree)
        self.assertIn("{OrderService,UserService}.java", tree)
        self.assertIn("util/", tree)
        self.assertIn("Helper.java", tree)

    def test_empty_paths(self):
        print("\n  ▶ TestBuildCompressedTree.test_empty_paths")
        result = _build_compressed_tree([])
        self.assertIn("Directory structure:", result)

    def test_single_file(self):
        print("\n  ▶ TestBuildCompressedTree.test_single_file")
        paths = [Path("README.md")]
        result = _build_compressed_tree(paths)
        self.assertIn("README.md", result)

    def test_chain_collapse(self):
        print("\n  ▶ TestBuildCompressedTree.test_chain_collapse")
        paths = [Path("a/b/c/file.py")]
        result = _build_compressed_tree(paths)
        self.assertIn("a/b/c/", result)

    def test_brace_grouping_multiple_extensions(self):
        print("\n  ▶ TestBuildCompressedTree.test_brace_grouping_multiple_extensions")
        paths = [
            Path("src/Alpha.java"),
            Path("src/Beta.java"),
            Path("src/README.md"),
        ]
        result = _build_compressed_tree(paths)
        self.assertIn("{Alpha,Beta}.java", result)
        self.assertIn("README.md", result)

    def test_single_file_no_brace(self):
        print("\n  ▶ TestBuildCompressedTree.test_single_file_no_brace")
        paths = [Path("src/Only.java")]
        result = _build_compressed_tree(paths)
        self.assertIn("Only.java", result)
        self.assertNotIn("{", result)


# ---------------------------------------------------------------------------
# _render_compressed_node
# ---------------------------------------------------------------------------

class TestRenderCompressedNode(unittest.TestCase):

    def test_files_only(self):
        print("\n  ▶ TestRenderCompressedNode.test_files_only")
        node = {"alpha.py": None, "beta.py": None, "gamma.py": None}
        lines = []
        _render_compressed_node(node, lines, "")
        joined = "\n".join(lines)
        self.assertIn("{alpha,beta,gamma}.py", joined)

    def test_dir_with_single_child_collapses(self):
        print("\n  ▶ TestRenderCompressedNode.test_dir_with_single_child_collapses")
        node = {"src": {"main": {"App.java": None}}}
        lines = []
        _render_compressed_node(node, lines, "")
        joined = "\n".join(lines)
        self.assertIn("src/main/", joined)

    def test_no_ext_files_listed_separately(self):
        print("\n  ▶ TestRenderCompressedNode.test_no_ext_files_listed_separately")
        node = {"Makefile": None, "Dockerfile": None}
        lines = []
        _render_compressed_node(node, lines, "")
        joined = "\n".join(lines)
        self.assertIn("Dockerfile", joined)
        self.assertIn("Makefile", joined)


if __name__ == "__main__":
    unittest.main()
