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

if __name__ == "__main__":
    unittest.main()
