import tree_sitter_java as tsj
from tree_sitter import Language, Parser

java_lang = Language(tsj.language())
parser = Parser(java_lang)

broken_java = b'''
public class Test {
    public void run() {
        System.out.println("missing semicolon")
    }
}
'''
tree = parser.parse(broken_java)

errors = []
def walk(n):
    if n.type == 'ERROR' or n.is_missing:
        errors.append(n)
    for c in n.children:
        walk(c)

walk(tree.root_node)

for e in errors:
    print(f"Line {e.start_point[0]+1}: Syntax error around `{broken_java[e.start_byte:e.end_byte].decode('utf-8')}`")
print(f"Errors found: {len(errors)}")
