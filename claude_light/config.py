import os
import sys
from pathlib import Path

try:
    from tree_sitter import Language, Parser as TSParser
    _TREESITTER_AVAILABLE = True
except ImportError:
    _TREESITTER_AVAILABLE = False
    print(
        "[Warning] tree-sitter not installed. Falling back to whole-file chunking.\n"
        "  pip install tree-sitter tree-sitter-java tree-sitter-python "
        "tree-sitter-go tree-sitter-rust tree-sitter-javascript tree-sitter-typescript"
    )

is_test_mode = "--test-mode" in sys.argv

def _resolve_api_key() -> str:
    """Return the API key from env, then ~/.anthropic, then .env in cwd."""
    if is_test_mode:
        return "sk-ant-test-mock-key"
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    for dotfile in (Path.home() / ".anthropic", Path(".env")):
        try:
            for line in dotfile.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""

API_KEY = _resolve_api_key()
if not API_KEY and not is_test_mode:
    raise SystemExit(
        "[Error] No API key found. Set ANTHROPIC_API_KEY in your environment,\n"
        "        or add ANTHROPIC_API_KEY=sk-ant-... to ~/.anthropic or ./.env"
    )

MODEL_HAIKU             = "claude-haiku-4-5-20251001"
MODEL_SONNET            = "claude-sonnet-4-6"
MODEL_OPUS              = "claude-opus-4-6"
MODEL                   = MODEL_SONNET              # default; overridden per-turn by router
SUMMARY_MODEL           = MODEL_HAIKU               # cheap model for history compression
HEARTBEAT_SECS          = 30
CACHE_TTL_SECS          = 240
TARGET_RETRIEVED_TOKENS = 6_000   # desired context size per query in tokens
MIN_SCORE               = 0.45    # discard retrieved files below this similarity (absolute floor)
RELATIVE_SCORE_FLOOR    = 0.60    # drop chunks below this fraction of the top chunk's score
# Per-effort retrieval token budgets (scaled from TARGET_RETRIEVED_TOKENS)
_RETRIEVAL_BUDGET = {"low": 1_500, "medium": 3_000, "high": 6_000, "max": 9_000}
MAX_HISTORY_TURNS       = 6       # compress+cap when stored turns exceed this
SUMMARIZE_BATCH         = 3       # how many old turns to collapse into a summary at once

SKIP_DIRS      = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

# Disk cache — stored in a hidden dir so _is_skipped() ignores it automatically
CACHE_DIR      = Path(".claude_light_cache")
CACHE_INDEX    = CACHE_DIR / "index.pkl"
CACHE_MANIFEST = CACHE_DIR / "manifest.json"

_DOC_PREFIX   = {"nomic-ai/nomic-embed-text-v1.5": "search_document: "}
_QUERY_PREFIX = {"nomic-ai/nomic-embed-text-v1.5": "search_query: "}

# ---------------------------------------------------------------------------
# Tree-sitter language configuration
# ---------------------------------------------------------------------------

_LANG_CONFIG: dict = {}          # ext → {"lang": Language, "node_types": [...]} | None
INDEXABLE_EXTENSIONS: set = set()  # all file extensions we want to index
_NAME_CHILD_TYPES = {"identifier", "name", "field_identifier", "type_identifier"}

# Extensions to index and their preferred tree-sitter node types (loaded lazily)
_WANTED_LANGS = {
    ".java": (lambda: Language(__import__("tree_sitter_java").language()),
              ["method_declaration", "constructor_declaration"]),
    ".py":   (lambda: Language(__import__("tree_sitter_python").language()),
              ["function_definition", "async_function_definition", "decorated_definition"]),
    ".js":   (lambda: Language(__import__("tree_sitter_javascript").language()),
              ["function_declaration", "method_definition"]),
    ".go":   (lambda: Language(__import__("tree_sitter_go").language()),
              ["function_declaration", "method_declaration"]),
    ".rs":   (lambda: Language(__import__("tree_sitter_rust").language()),
              ["function_item"]),
}

def _load_languages():
    """Populate _LANG_CONFIG and INDEXABLE_EXTENSIONS at import time."""
    if not _TREESITTER_AVAILABLE:
        for ext in _WANTED_LANGS:
            _LANG_CONFIG[ext] = None
        INDEXABLE_EXTENSIONS.update(_WANTED_LANGS)
        return

    for ext, (get_lang, node_types) in _WANTED_LANGS.items():
        try:
            _LANG_CONFIG[ext] = {"lang": get_lang(), "node_types": node_types}
        except Exception:
            _LANG_CONFIG[ext] = None

    # TypeScript exposes two separate callables
    try:
        import tree_sitter_typescript as _tspy
        ts_nodes = ["function_declaration", "method_definition", "arrow_function"]
        _LANG_CONFIG[".ts"]  = {"lang": Language(_tspy.language_typescript()), "node_types": ts_nodes}
        _LANG_CONFIG[".tsx"] = {"lang": Language(_tspy.language_tsx()),         "node_types": ts_nodes}
    except Exception:
        _LANG_CONFIG[".ts"] = _LANG_CONFIG[".tsx"] = None

    INDEXABLE_EXTENSIONS.update(_LANG_CONFIG)

_load_languages()

SYSTEM_PROMPT = """\
You are an expert code assistant. Answer questions about the codebase provided below. Be concise and precise.

To edit an existing file, use one or more SEARCH/REPLACE blocks — one per contiguous change:

```python:path/to/file.py
<<<<<<< SEARCH
    exact lines to replace (must match the file verbatim, including explicit indentation)
=======
    replacement lines (must include the exact same indentation as the original)
>>>>>>> REPLACE
```

To create a new file, use a plain block with no SEARCH/REPLACE markers:

```python:path/to/newfile.py
full file content here
```

Rules:
1. CRITICAL: SEARCH must match the file exactly (character for character). Include all leading whitespace, indentation, and trailing blank lines. If modifying an indented block, DO NOT strip the leading spaces!
2. CRITICAL: REPLACE must contain the exact same leading indentation as the SEARCH block. If you output a flat REPLACE block, it will cause SyntaxErrors in Python!
3. Use multiple blocks for multiple changes, even within the same file.
4. Use the correct path relative to the project root.
5. After all blocks, write a short plain-English summary of what changed.
"""

# 2026 PRICING (USD per 1 Million Tokens)
PRICE_INPUT  = 3.00
PRICE_WRITE  = PRICE_INPUT * 1.25   # $3.75
PRICE_READ   = PRICE_INPUT * 0.10   # $0.30
PRICE_OUTPUT = 15.00
