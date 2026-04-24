import re
from pathlib import Path
from claude_light.config import _LANG_CONFIG, _NAME_CHILD_TYPES

try:
    from tree_sitter import Parser as TSParser
except ImportError:
    pass

_COMMENT_EXTS_C = frozenset({".java", ".js", ".ts", ".tsx", ".go", ".rs"})


def _walk(node, node_types, results):
    """DFS: collect nodes whose type is in node_types; don't recurse into matches."""
    if node.type in node_types:
        results.append(node)
        return
    for child in node.children:
        _walk(child, node_types, results)


def _extract_symbol_name(node):
    """Return the identifier name for an AST symbol node."""
    # Python decorated_definition wraps a function — dig inside for the name
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in {"function_definition", "async_function_definition"}:
                return _extract_symbol_name(child)
    for child in node.children:
        if child.type in _NAME_CHILD_TYPES:
            return child.text.decode("utf-8", errors="replace")
    return f"{node.type}_{node.start_point[0]}"


def _chunk_with_treesitter(filepath, source, language, node_types):
    """
    Parse source with tree-sitter and emit one chunk per matched symbol node.
    Preamble = source lines before the first symbol (imports, class header, etc.).
    Falls back to whole-file if parsing yields no symbols.
    """
    src_bytes = bytes(source, "utf-8")
    parser    = TSParser(language)
    tree      = parser.parse(src_bytes)

    symbols = []
    _walk(tree.root_node, node_types, symbols)

    if not symbols:
        return [{"id": filepath, "text": source}]

    lines          = source.splitlines(keepends=True)
    first_sym_line = symbols[0].start_point[0]   # 0-indexed row
    preamble       = "".join(lines[:first_sym_line]).rstrip()

    chunks = []
    seen = {}
    for node in symbols:
        name = _extract_symbol_name(node)
        # Deduplicate overloaded names with a numeric suffix
        if name in seen:
            seen[name] += 1
            uid = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
            uid = name

        node_text  = source[node.start_byte:node.end_byte]
        chunk_text = (
            f"// {filepath}\n"
            + preamble + "\n"
            + "    // ...\n"
            + node_text.strip()
            + "\n"
        )
        chunks.append({"id": f"{filepath}::{uid}", "text": chunk_text})

    return chunks


def chunk_file(filepath, source):
    """
    Split a source file into symbol-level chunks using tree-sitter AST parsing.
    Dispatches by file extension; falls back to whole-file for unsupported or
    missing grammar packages.
    Chunk ID format: 'filepath::symbolName' or 'filepath' (whole-file fallback).
    """
    ext = Path(filepath).suffix.lower()
    cfg = _LANG_CONFIG.get(ext)
    if cfg is None:
        return [{"id": filepath, "text": source}]
    return _chunk_with_treesitter(filepath, source, cfg["lang"], cfg["node_types"])


def _strip_comments(text: str, ext: str) -> str:
    """Remove comments from a code chunk to cut retrieved-context token usage.

    Strips only code comments — not the file-path header line or preamble —
    so the structural context (package, imports, class declaration) is preserved.
    """
    if ext in _COMMENT_EXTS_C:
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)   # /* block */
        text = re.sub(r'//[^\n]*',  '', text)                    # // line
    elif ext == ".py":
        text = re.sub(r'#.*', '', text)                          # # line
        text = re.sub(r'"""[\s\S]*?"""', '', text)               # """ docstring """
        text = re.sub(r"'''[\s\S]*?'''", '', text)               # ''' docstring '''
    # Collapse runs of 3+ blank lines left behind by removal
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text
