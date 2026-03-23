import os
import sys
import unittest
from pathlib import Path

# Inject dummy API key to prevent sys.exit when importing claude_light
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy-test-key"

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from claude_light import (
    route_query, 
    parse_edit_blocks, 
    _strip_comments, 
    _build_compressed_tree, 
    _chunk_label
)


class TestClaudeLight(unittest.TestCase):

    def test_chunk_label(self):
        self.assertEqual(_chunk_label("src/Foo.java::doThing"), "Foo.java::doThing")
        self.assertEqual(_chunk_label("src/Foo.java"), "Foo.java")

    def test_route_query(self):
        # low effort (simple lookups)
        model, effort, tokens = route_query("list all files")
        self.assertEqual(effort, "low")
        
        # medium effort (default fallback)
        model, effort, tokens = route_query("what does this function do")
        self.assertEqual(effort, "medium")
        
        # high effort (code modifications)
        model, effort, tokens = route_query("refactor the parse method to handle exceptions")
        self.assertEqual(effort, "high")
        
        # max effort (complex architectural reasoning)
        model, effort, tokens = route_query("evaluate the scalability trade-offs of the current microservices architecture deeply")
        self.assertEqual(effort, "max")

    def test_strip_comments_python(self):
        code = "def foo():\n    # this is a comment\n    return 42\n"
        stripped = _strip_comments(code, ".py")
        self.assertNotIn("# this is a comment", stripped)
        self.assertIn("def foo():", stripped)
        self.assertIn("return 42", stripped)

    def test_strip_comments_c_style(self):
        code = "void doThing() {\n    // line comment\n    /* block \n       comment */\n    println();\n}"
        stripped = _strip_comments(code, ".java")
        self.assertNotIn("line comment", stripped)
        self.assertNotIn("block", stripped)
        self.assertNotIn("comment */", stripped)
        self.assertIn("void doThing() {", stripped)
        self.assertIn("println();", stripped)

    def test_parse_edit_blocks_search_replace(self):
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

    def test_parse_edit_blocks_new_file(self):
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

    def test_parse_edit_blocks_missing_tag(self):
        text = '''Here is the new file:
```python
# src/feature.py
def feature(): pass
```'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["type"], "new")
        self.assertEqual(edits[0]["path"], "src/feature.py")
        
    def test_parse_edit_blocks_conversational(self):
        text = '''You can just call it like this:
```python
feature()
```'''
        edits = parse_edit_blocks(text)
        self.assertEqual(len(edits), 0)
        
    def test_parse_edit_blocks_absolute_normalized(self):
        text = '''```python:/home/user/workspace/src/test.py\ndef feature(): pass\n```'''
        edits = parse_edit_blocks(text)
        self.assertEqual(edits[0]["path"], "home/user/workspace/src/test.py")

    def test_build_compressed_tree(self):
        paths = [
            Path("src/main/java/com/example/UserService.java"),
            Path("src/main/java/com/example/OrderService.java"),
            Path("src/main/java/com/example/util/Helper.java"),
            Path("README.md")
        ]
        tree = _build_compressed_tree(paths)
        self.assertIn("README.md", tree)
        self.assertIn("src/main/java/com/example/", tree)
        self.assertIn("{OrderService,UserService}.java", tree)
        self.assertIn("util/", tree)
        self.assertIn("Helper.java", tree)

class TestResolveNewContent(unittest.TestCase):
    
    def setUp(self):
        self.original_code = "def foo():\n    try:\n        a = 1\n        b = 2\n    except:\n        pass\n"
        Path("dummy.py").write_text(self.original_code, encoding="utf-8")

    def tearDown(self):
        Path("dummy.py").unlink(missing_ok=True)

    def test_resolve_exact(self):
        # Exact match
        from claude_light import _resolve_new_content
        search = "        a = 1\n        b = 2"
        replace = "        a = 3\n        b = 4"
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("a = 3", result)
        
    def test_resolve_fuzzy_indentation(self):
        # LLM forgets the 8 leading spaces!
        from claude_light import _resolve_new_content
        search = "a = 1\nb = 2\n"
        replace = "a = 5\nb = 5\n"
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("        a = 5", result)
        self.assertIn("        b = 5", result)

    def test_resolve_fuzzy_hallucination(self):
        # LLM hallucinates a typo combining two lines!
        from claude_light import _resolve_new_content
        search = "    try:\n        a = 1, b = 2"
        replace = "    try:\n        a = 9\n        b = 9"
        # Should catch the sequence block due to high string similarity
        _, result = _resolve_new_content({"path": "dummy.py", "type": "edit", "search": search, "replace": replace})
        self.assertIn("        a = 9", result)

    def test_autonomous_linting(self):
        # LLM generated a totally busted syntax
        from claude_light import apply_edits
        search = "        a = 1\n        b = 2"
        replace = "        a = 3\n        print('missing parenthesis"

        # Test that check_only=True silently evaluates and returns the SyntaxError
        errs = apply_edits([{"path": "dummy.py", "type": "edit", "search": search, "replace": replace}], check_only=True)
        self.assertEqual(len(errs), 1)
        self.assertIn("SyntaxError", errs[0])
        self.assertIn("dummy.py", errs[0])

    def test_autonomous_linting_java(self):
        # LLM generated Java with an unbalanced brace
        from claude_light import _lint_content
        import claude_light as cl
        if not cl._TREESITTER_AVAILABLE:
            self.skipTest("tree-sitter not installed")
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

        broken_java = (
            "public class Foo {\n"
            "    public void bar( {\n"   # missing closing paren
            "        int x = 1;\n"
            "    }\n"
            "}\n"
        )
        err = _lint_content("Foo.java", broken_java)
        self.assertIsNotNone(err)
        self.assertIn("SyntaxError", err)

    def test_lint_content_skips_unknown_extensions(self):
        from claude_light import _lint_content
        # Non-.py / non-.java files should return None (no linter registered)
        self.assertIsNone(_lint_content("script.sh", "this is { not valid anything"))
        self.assertIsNone(_lint_content("style.css", "color: ;{{{"))

if __name__ == "__main__":
    unittest.main()
