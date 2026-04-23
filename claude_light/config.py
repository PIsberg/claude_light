import os
import sys
from pathlib import Path

# Configure UTF-8 output as early as possible (config is the first module
# imported by __init__.py) so that ui.py's _UNICODE detection sees UTF-8
# rather than the cp1252 default on Windows.
if os.name == 'nt':
    for _s in (sys.stdout, sys.stderr):
        try:
            if hasattr(_s, 'reconfigure'):
                _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

from claude_light.testing import is_test_mode_enabled, get_test_api_key

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

is_test_mode = is_test_mode_enabled()

def _get_pro_token() -> str:
    """Read Claude CLI credentials. Prioritize the automation token if it exists."""
    # 1. Check for the local automation token (most reliable for subprocesses on Windows)
    auto_token_path = Path.home() / ".claude_light_automation_token"
    if auto_token_path.exists():
        try:
            return auto_token_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    # 2. Fallback to the official Claude CLI credentials (from ~/.claude/.credentials.json)
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.exists():
        try:
            import json
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            return data.get("claudeAiOauth", {}).get("accessToken", "")
        except Exception:
            pass
    return ""

def _resolve_api_key() -> tuple[str, str, str, str]:
    """Return (api_key, auth_mode, source, auth_token)."""
    if is_test_mode:
        return get_test_api_key(), "API_KEY", "Test", get_test_api_key()
    
    # 1. Check environment variable
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key, "API_KEY", "Environment", key
    
    # 2. Check local dotfiles
    for dotfile in (Path.home() / ".anthropic", Path(".env")):
        try:
            for line in dotfile.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY=") and not line.startswith("#"):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if k: return k, "API_KEY", f"Dotfile ({dotfile.name})", k
        except OSError:
            pass
            
    # 3. Fallback to Claude CLI Pro subscription (via CLI subprocess)
    pro_token = _get_pro_token()
    if pro_token:
        # We don't actually use the token directly anymore in llm.py, 
        # but presence of the token confirms the user is logged in to the CLI.
        return "", "OAUTH", "Claude CLI (Pro Subscription)", pro_token
        
    return "", "OAUTH", "None", ""

API_KEY, AUTH_MODE, API_KEY_SOURCE, AUTH_TOKEN = _resolve_api_key()
ECONOMY_MODE = "USD" if AUTH_MODE == "API_KEY" else "TOKENS"

MODEL_HAIKU             = "claude-haiku-4-5-20251001"
MODEL_SONNET            = "claude-sonnet-4-6"
MODEL_OPUS              = "claude-opus-4-6"
MODEL                   = MODEL_SONNET              # default; overridden per-turn by router
SUMMARY_MODEL           = MODEL_HAIKU               # cheap model for history compression
HEARTBEAT_SECS          = 30
CACHE_TTL_SECS          = 240
ENABLE_STREAMING        = True         # enable real-time token streaming
TARGET_RETRIEVED_TOKENS = 6_000   # desired context size per query in tokens
MIN_SCORE               = 0.45    # discard retrieved files below this similarity (absolute floor)
RELATIVE_SCORE_FLOOR    = 0.60    # drop chunks below this fraction of the top chunk's score
# Per-effort retrieval token budgets (scaled from TARGET_RETRIEVED_TOKENS)
_RETRIEVAL_BUDGET = {"low": 1_500, "medium": 3_000, "high": 6_000, "max": 9_000}
MAX_HISTORY_TURNS       = 6       # compress+cap when stored turns exceed this
SUMMARIZE_BATCH         = 3       # how many old turns to collapse into a summary at once

# LLMLingua-2 prompt compression (optional; falls back silently if not installed).
# See docs/llmlingua_plan.md. ON by default — opt out with CLAUDE_LIGHT_LLMLINGUA=0.
# If `llmlingua` isn't installed, compress_context() is a silent no-op, so this
# default is safe on fresh machines: it only activates once the user runs
# `pip install llmlingua`.
LLMLINGUA_ENABLED      = os.environ.get("CLAUDE_LIGHT_LLMLINGUA", "1") != "0"
LLMLINGUA_MODEL        = os.environ.get(
    "CLAUDE_LIGHT_LLMLINGUA_MODEL",
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
)
LLMLINGUA_TARGET_RATE  = float(os.environ.get("CLAUDE_LIGHT_LLMLINGUA_RATE", "0.5"))
LLMLINGUA_MIN_TOKENS   = 800      # below this, skip — compression overhead > savings
LLMLINGUA_FORCE_TOKENS = ["\n", "```", "::", "//"]

SKIP_DIRS      = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

# Global Stats - stored in home directory to persist across all repositories
GLOBAL_STATS_FILE = Path.home() / ".claude_light_stats.json"

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
