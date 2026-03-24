from pathlib import Path
from claude_light.config import _TREESITTER_AVAILABLE

def _lint_python_content(filepath: str, new_content: str) -> str | None:
    try:
        import ast
        ast.parse(new_content, filename=filepath)
        return None
    except SyntaxError as e:
        import traceback
        return "".join(traceback.format_exception_only(type(e), e)).strip()


def _lint_via_treesitter(new_content: str, ts_language) -> str | None:
    """Parse new_content with a tree-sitter Language and return an error string or None."""
    from tree_sitter import Parser as TSParser
    parser = TSParser(ts_language)
    tree = parser.parse(bytes(new_content, "utf-8"))
    if not tree.root_node.has_error:
        return None
    errors = []
    stack = [tree.root_node]
    src_lines = new_content.splitlines()
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            line = node.start_point[0] + 1
            col  = node.start_point[1] + 1
            snippet = src_lines[node.start_point[0]] if node.start_point[0] < len(src_lines) else ""
            errors.append(f"  line {line}, col {col}: syntax error near '{snippet.strip()[:60]}'")
        else:
            stack.extend(node.children)
    detail = "\n".join(errors[:5]) or "  (unknown location)"
    return f"SyntaxError:\n{detail}"


def _lint_java_content(filepath: str, new_content: str) -> str | None:
    if not _TREESITTER_AVAILABLE:
        return None
    try:
        import tree_sitter_java as tsjava
        from tree_sitter import Language
        return _lint_via_treesitter(new_content, Language(tsjava.language()))
    except Exception:
        return None


def _lint_javascript_content(filepath: str, new_content: str) -> str | None:
    if not _TREESITTER_AVAILABLE:
        return None
    try:
        import tree_sitter_javascript as tsjs
        from tree_sitter import Language
        return _lint_via_treesitter(new_content, Language(tsjs.language()))
    except Exception:
        return None


def _lint_typescript_content(filepath: str, new_content: str) -> str | None:
    if not _TREESITTER_AVAILABLE:
        return None
    try:
        import tree_sitter_typescript as tsts
        from tree_sitter import Language
        ext = Path(filepath).suffix.lower()
        lang = Language(tsts.language_tsx() if ext == ".tsx" else tsts.language_typescript())
        return _lint_via_treesitter(new_content, lang)
    except Exception:
        return None


# Dispatch table: extension -> linter function (filepath, content) -> error str | None
_LINTERS: dict = {
    ".py":   _lint_python_content,
    ".java": _lint_java_content,
    ".js":   _lint_javascript_content,
    ".ts":   _lint_typescript_content,
    ".tsx":  _lint_typescript_content,
}


def _lint_content(filepath: str, new_content: str) -> str | None:
    """Run the appropriate syntax linter for filepath. Returns an error string or None."""
    ext = Path(filepath).suffix.lower()
    linter = _LINTERS.get(ext)
    return linter(filepath, new_content) if linter else None
