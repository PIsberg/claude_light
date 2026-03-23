import os
import re
import sys
import time
import json
import pickle
import hashlib
import difflib
import threading
import concurrent.futures
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import anthropic

try:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "[Error] sentence-transformers not installed.\n"
        "  pip install sentence-transformers"
    )

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

try:
    from rich.console import Console as _RichConsole
    from rich.markdown import Markdown as _RichMarkdown
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    print("[Warning] rich not installed. Claude responses won't be formatted.\n  pip install rich")

try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.history import FileHistory as _FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _AutoSuggest
    from prompt_toolkit.completion import WordCompleter as _WordCompleter
    _PROMPTTK_AVAILABLE = True
except ImportError:
    _PROMPTTK_AVAILABLE = False
    print("[Warning] prompt_toolkit not installed. No command history/completion.\n  pip install prompt_toolkit")

# Prerequisites:
#   pip install sentence-transformers numpy
#   pip install tree-sitter tree-sitter-java tree-sitter-python \
#               tree-sitter-go tree-sitter-rust tree-sitter-javascript tree-sitter-typescript
#   apt install python3-watchdog python3-anthropic
#   Note: sentence-transformers pulls in PyTorch (~1.5 GB on first install)

# --- CONFIG ---
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

# Auto-tuned at runtime — do not set manually
EMBED_MODEL = None
TOP_K       = None
SKIP_DIRS      = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

# Disk cache — stored in a hidden dir so _is_skipped() ignores it automatically
CACHE_DIR      = Path(".claude_light_cache")
CACHE_INDEX    = CACHE_DIR / "index.pkl"
CACHE_MANIFEST = CACHE_DIR / "manifest.json"

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
exact lines to replace (must match the file verbatim, including indentation)
=======
replacement lines
>>>>>>> REPLACE
```

To create a new file, use a plain block with no SEARCH/REPLACE markers:

```python:path/to/newfile.py
full file content here
```

Rules:
- SEARCH must match the file exactly (whitespace, indentation, blank lines).
- Use multiple blocks for multiple changes, even within the same file.
- Use the correct path relative to the project root.
- After all blocks, write a short plain-English summary of what changed.
"""

# Matches any fenced block with a path tag:  ```[lang]:path\n...\n```
_ANY_BLOCK = re.compile(r"```[a-zA-Z]*:([^\n`]+)\n(.*?)```", re.DOTALL)
# Kept for stripping edit blocks from history (matches both formats)
_EDIT_BLOCK = _ANY_BLOCK

client   = anthropic.Anthropic(api_key=API_KEY)
embedder = None   # initialised by auto_tune() before first use
console  = _RichConsole() if _RICH_AVAILABLE else None

# nomic-embed requires explicit prefixes for docs vs queries; MiniLM does not
_DOC_PREFIX   = {"nomic-ai/nomic-embed-text-v1.5": "search_document: "}
_QUERY_PREFIX = {"nomic-ai/nomic-embed-text-v1.5": "search_query: "}

# 2026 PRICING (USD per 1 Million Tokens)
PRICE_INPUT  = 3.00
PRICE_WRITE  = PRICE_INPUT * 1.25   # $3.75
PRICE_READ   = PRICE_INPUT * 0.10   # $0.30
PRICE_OUTPUT = 15.00

# Shared state
skeleton_context     = ""   # cached: dir tree + .md files (assembled from parts below)
_skeleton_tree       = ""   # directory tree portion — rebuilt on structural changes
_skeleton_md_hashes: dict = {}  # {path_str: md5_hex} — skip re-read when unchanged
_skeleton_md_parts:  dict = {}  # {path_str: rendered_str} — one entry per .md file
chunk_store          = {}   # {chunk_id → {"text": str, "emb": np.ndarray}}
                             # chunk_id = "filepath" (whole-file) or "filepath::method"
conversation_history = []   # clean turns; retrieved context is NOT stored here
session_cost         = 0.0
session_tokens       = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
last_interaction     = time.time()
lock                 = threading.Lock()
stop_event           = threading.Event()
_source_files: list  = []   # set by index_files(); used by _save_cache()
_file_hashes: dict   = {}   # {str(path): md5_hex} — kept in sync by index_files + reindex_file


# ---------------------------------------------------------------------------
# UI helpers — ANSI colours, spinner
# ---------------------------------------------------------------------------

_ANSI_RED     = "\033[31m"
_ANSI_GREEN   = "\033[32m"
_ANSI_YELLOW  = "\033[33m"
_ANSI_BLUE    = "\033[34m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_CYAN    = "\033[36m"
_ANSI_BOLD    = "\033[1m"
_ANSI_DIM     = "\033[2m"
_ANSI_RESET   = "\033[0m"

_T_RAG    = f"{_ANSI_CYAN}[RAG]{_ANSI_RESET}"
_T_CACHE  = f"{_ANSI_YELLOW}[Cache]{_ANSI_RESET}"
_T_SYS    = f"{_ANSI_MAGENTA}[System]{_ANSI_RESET}"
_T_EDIT   = f"{_ANSI_CYAN}[Edit]{_ANSI_RESET}"
_T_ERR    = f"{_ANSI_RED}[Error]{_ANSI_RESET}"
_T_TEST   = f"{_ANSI_BLUE}[Test Mode]{_ANSI_RESET}"
_T_ROUTE  = f"{_ANSI_MAGENTA}[Router]{_ANSI_RESET}"


class _Spinner:
    """Animated braille spinner for long-running operations."""
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            print(f"\r\033[K{_ANSI_CYAN}{frame}{_ANSI_RESET} {self.label}...", end="", flush=True)
            i += 1
            time.sleep(0.1)
        print("\r\033[K", end="", flush=True)

    def update(self, label: str):
        self.label = label

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------------------
# Cost / stats
# ---------------------------------------------------------------------------

def calculate_cost(usage):
    write = (getattr(usage, "cache_creation_input_tokens", 0) / 1_000_000) * PRICE_WRITE
    read  = (getattr(usage, "cache_read_input_tokens",      0) / 1_000_000) * PRICE_READ
    inp   = (usage.input_tokens  / 1_000_000) * PRICE_INPUT
    out   = (usage.output_tokens / 1_000_000) * PRICE_OUTPUT
    return write + read + inp + out


def print_stats(usage, label="Stats", file=sys.stdout):
    write_tokens = getattr(usage, "cache_creation_input_tokens", 0)
    read_tokens  = getattr(usage, "cache_read_input_tokens",      0)
    total_input  = usage.input_tokens + write_tokens + read_tokens

    actual_cost  = calculate_cost(usage)
    nocache_cost = (total_input         / 1_000_000) * PRICE_INPUT \
                 + (usage.output_tokens / 1_000_000) * PRICE_OUTPUT

    savings     = nocache_cost - actual_cost
    savings_pct = (savings / nocache_cost * 100) if nocache_cost > 0 else 0.0
    hit_pct     = (read_tokens / total_input * 100) if total_input > 0 else 0.0

    print(
        f"{_ANSI_DIM}[{label}]{_ANSI_RESET}  {total_input:,} tokens  |  "
        f"cached {_ANSI_GREEN}{read_tokens:,}{_ANSI_RESET} ({hit_pct:.1f}%)  |  "
        f"new {write_tokens:,}",
        file=file,
    )
    print(
        f"{_ANSI_DIM}[Cost]{_ANSI_RESET}   "
        f"{_ANSI_YELLOW}${actual_cost:.4f}{_ANSI_RESET}  |  "
        f"saved {_ANSI_GREEN}${savings:.4f}{_ANSI_RESET} ({savings_pct:.1f}% vs no-cache)  |  "
        f"session {_ANSI_YELLOW}${session_cost:.4f}{_ANSI_RESET}",
        file=file,
    )


def route_query(query: str) -> tuple[str, str, int]:
    """Classify a query and return (model_id, effort_label, max_tokens).

    Effort levels:
      low    → Haiku   — simple lookups, listings, one-liners
      medium → Sonnet  — explanations, moderate analysis
      high   → Sonnet  — code generation, multi-step tasks (default)
      max    → Opus    — deep architecture, cross-cutting analysis, extended thinking

    Prints the routing decision to stderr before returning.
    """
    q = query.lower()
    words = q.split()
    word_count = len(words)

    # --- signal sets ---
    _LOW_SIGNALS = {
        "list", "show", "where", "what is", "what are", "how many",
        "print", "display", "tell me", "which file", "which files",
        "find", "locate", "count",
    }
    _HIGH_SIGNALS = {
        "implement", "write", "create", "add", "refactor", "fix", "debug",
        "build", "develop", "generate", "update", "change", "modify",
        "migrate", "convert", "extend", "integrate",
    }
    _MAX_SIGNALS = {
        "architect", "architecture", "design system", "deeply", "deep analysis",
        "performance", "optimize", "security", "scalability", "trade-off",
        "trade off", "tradeoff", "cross-cutting", "evaluate", "compare",
        "strategy", "reasoning", "step by step", "comprehensive",
    }

    # Score each tier
    low_hits  = sum(1 for s in _LOW_SIGNALS  if s in q)
    high_hits = sum(1 for s in _HIGH_SIGNALS if s in q)
    max_hits  = sum(1 for s in _MAX_SIGNALS  if s in q)

    # Determine effort
    if max_hits >= 2 or (max_hits >= 1 and word_count > 35):
        effort, model, max_tokens = "max",    MODEL_OPUS,   16_000
    elif high_hits >= 1 or word_count > 30:
        effort, model, max_tokens = "high",   MODEL_SONNET,  8_192
    elif low_hits >= 1 and high_hits == 0 and word_count <= 20:
        effort, model, max_tokens = "low",    MODEL_HAIKU,   2_048
    else:
        effort, model, max_tokens = "medium", MODEL_SONNET,  4_096

    _effort_color = {
        "low":    _ANSI_GREEN,
        "medium": _ANSI_CYAN,
        "high":   _ANSI_YELLOW,
        "max":    _ANSI_MAGENTA,
    }
    color = _effort_color[effort]
    model_short = model.split("-")[1]   # "haiku" / "sonnet" / "opus"
    print(
        f"{_T_ROUTE} effort={color}{effort}{_ANSI_RESET}  "
        f"model={_ANSI_BOLD}{model_short}{_ANSI_RESET}",
        file=sys.stderr,
    )
    return model, effort, max_tokens


def _extract_text(content_blocks) -> str:
    """Return joined text from content blocks, skipping thinking blocks."""
    return "".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )


def _accumulate_usage(usage):
    """Add per-call token counts to session totals (call under no lock needed — GIL is enough)."""
    session_tokens["input"]       += usage.input_tokens
    session_tokens["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0)
    session_tokens["cache_read"]  += getattr(usage, "cache_read_input_tokens", 0)
    session_tokens["output"]      += usage.output_tokens


def print_session_summary():
    inp   = session_tokens["input"]
    cw    = session_tokens["cache_write"]
    cr    = session_tokens["cache_read"]
    out   = session_tokens["output"]
    total = inp + cw + cr + out

    cost_inp = (inp / 1_000_000) * PRICE_INPUT
    cost_cw  = (cw  / 1_000_000) * PRICE_WRITE
    cost_cr  = (cr  / 1_000_000) * PRICE_READ
    cost_out = (out / 1_000_000) * PRICE_OUTPUT
    cost_tot = cost_inp + cost_cw + cost_cr + cost_out

    def pct(n):
        return f"{n / total * 100:.1f}%" if total else "—"

    col_w = [22, 12, 8, 9]   # label | tokens | % | cost

    _B = _ANSI_CYAN
    _R = _ANSI_RESET
    _H = _ANSI_BOLD

    def row(label, tokens, cost, color=""):
        cost_str = f"${cost:.4f}"
        return (f"│ {color}{label:<{col_w[0]}}{_R} │ {tokens:>{col_w[1]},} │"
                f" {pct(tokens):>{col_w[2]}} │ {_ANSI_YELLOW}{cost_str:>{col_w[3]}}{_R} │")

    turns      = len(conversation_history) // 2
    input_base = inp + cw + cr
    hit_str    = f"{_ANSI_GREEN}{cr / input_base * 100:.1f}%{_R}" if input_base else "—"

    print(f"\n{_B}┌{'─'*62}┐{_R}")
    print(f"{_B}│{_R}{_H}{'Session Token Summary':^62}{_R}{_B}│{_R}")
    print(f"{_B}├{'─'*24}┬{'─'*14}┬{'─'*10}┬{'─'*11}┤{_R}")
    print(f"{_B}│{_R} {'Type':<{col_w[0]}} {_B}│{_R} {'Tokens':>{col_w[1]}} {_B}│{_R} {'%':>{col_w[2]}} {_B}│{_R} {'Cost':>{col_w[3]}} {_B}│{_R}")
    print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*11}┤{_R}")
    print(row("Input (uncached)",  inp,  cost_inp))
    print(row("Cache write",       cw,   cost_cw))
    print(row("Cache read",        cr,   cost_cr,  _ANSI_GREEN))
    print(row("Output",            out,  cost_out))
    print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*11}┤{_R}")
    print(row("TOTAL",             total, cost_tot, _ANSI_BOLD))
    print(f"{_B}└{'─'*24}┴{'─'*14}┴{'─'*10}┴{'─'*11}┘{_R}")
    print(f"  Turns: {_ANSI_BOLD}{turns}{_R}  |  Cache hit rate: {hit_str}")


# ---------------------------------------------------------------------------
# Skeleton — cached system prompt (directory tree + markdown docs)
# ---------------------------------------------------------------------------

def _is_skipped(path):
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)


def _build_compressed_tree(paths):
    """
    Build a compressed directory tree from an already-filtered list of Paths.

    Two compressions applied:
    1. Single-child directory chains are collapsed into one line:
           main/java/com/example/
       instead of four separate indented lines.
    2. Sibling files sharing an extension are grouped with brace notation:
           {OrderService,UserService,PaymentService}.java

    Returns the full "Directory structure:\\n..." string.
    """
    # Build nested dict: name → {} (directory) or None (file)
    root_node = {}
    root = Path(".")
    for path in paths:
        parts = path.relative_to(root).parts
        node  = root_node
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        last = parts[-1]
        if path.is_dir():
            node.setdefault(last, {})
        else:
            node.setdefault(last, None)

    lines = []
    _render_compressed_node(root_node, lines, "")
    return "Directory structure:\n" + "\n".join(lines)


def _render_compressed_node(node, lines, indent):
    """Recursively render a tree node with chain-collapse and brace grouping."""
    dirs  = {k: v for k, v in node.items() if isinstance(v, dict)}
    files = [k for k, v in node.items() if v is None]

    # Render directories — collapse single-child pure-dir chains
    for dirname in sorted(dirs):
        children   = dirs[dirname]
        compressed = dirname
        current    = children
        while len(current) == 1:
            only_key, only_val = next(iter(current.items()))
            if isinstance(only_val, dict):
                compressed += "/" + only_key
                current = only_val
            else:
                break
        lines.append(f"{indent}{compressed}/")
        _render_compressed_node(current, lines, indent + "  ")

    # Group sibling files by extension, brace-expand when > 1 stem
    by_ext = {}
    no_ext = []
    for fname in files:
        ext = Path(fname).suffix
        if ext:
            by_ext.setdefault(ext, []).append(Path(fname).stem)
        else:
            no_ext.append(fname)

    for fname in sorted(no_ext):
        lines.append(f"{indent}{fname}")
    for ext in sorted(by_ext):
        stems = sorted(by_ext[ext])
        if len(stems) == 1:
            lines.append(f"{indent}{stems[0]}{ext}")
        else:
            lines.append(f"{indent}{{{','.join(stems)}}}{ext}")


def _render_md_file(path: Path) -> str:
    """Read and render a single .md file for the skeleton. Returns '' on error/empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > 5000 and path.name.lower() not in ("claude.md", "agents.md"):
            text = text[:5000] + "\n\n... [TRUNCATED due to length]"
        return f"<!-- {path} -->\n{text}" if text else ""
    except OSError:
        return ""


def _assemble_skeleton() -> str:
    """Combine cached tree + md parts into the skeleton string."""
    docs = "\n\n".join(v for v in _skeleton_md_parts.values() if v)
    return _skeleton_tree + ("\n\n" + docs if docs else "")


def _refresh_single_md(path_str: str) -> bool:
    """Re-read one .md file if its hash changed. Returns True if content changed."""
    global _skeleton_md_hashes, _skeleton_md_parts
    p = Path(path_str)
    if not p.exists():
        changed = path_str in _skeleton_md_parts
        _skeleton_md_hashes.pop(path_str, None)
        _skeleton_md_parts.pop(path_str, None)
        return changed
    h = _file_hash(p)
    if _skeleton_md_hashes.get(path_str) == h:
        return False  # content unchanged — nothing to do
    _skeleton_md_hashes[path_str] = h
    _skeleton_md_parts[path_str] = _render_md_file(p)
    return True


def _refresh_tree_only() -> str:
    """Rebuild only the directory tree, reuse cached .md parts."""
    global _skeleton_tree
    all_paths = [p for p in sorted(Path(".").rglob("*")) if not _is_skipped(p)]
    _skeleton_tree = _build_compressed_tree(all_paths)
    return _assemble_skeleton()


def build_skeleton() -> str:
    """Full rglob pass: rebuild tree + hash-check every .md file (reuses cache on match)."""
    global _skeleton_tree, _skeleton_md_hashes, _skeleton_md_parts
    all_paths, md_files = [], []
    for path in sorted(Path(".").rglob("*")):
        if _is_skipped(path):
            continue
        all_paths.append(path)
        if path.suffix == ".md" and path.is_file():
            md_files.append(path)

    _skeleton_tree = _build_compressed_tree(all_paths)

    new_hashes, new_parts = {}, {}
    for path in md_files:
        path_str = str(path)
        h = _file_hash(path)
        new_hashes[path_str] = h
        # Reuse cached rendered string when the file hasn't changed
        if _skeleton_md_hashes.get(path_str) == h and path_str in _skeleton_md_parts:
            new_parts[path_str] = _skeleton_md_parts[path_str]
        else:
            new_parts[path_str] = _render_md_file(path)
    _skeleton_md_hashes = new_hashes
    _skeleton_md_parts  = new_parts

    return _assemble_skeleton()


# ---------------------------------------------------------------------------
# RAG — method chunking, auto-tune, embed, retrieve
# ---------------------------------------------------------------------------

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
    seen: dict = {}
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


def auto_tune(source_files, chunks=None, quiet=False):
    """
    Model selection (file count):
      <  50 files  → all-MiniLM-L6-v2        (22 MB,  fast)
       50-199       → all-mpnet-base-v2       (420 MB, better depth)
      200+          → nomic-ai/nomic-embed-text-v1.5  (best for large repos)

    TOP_K uses chunk sizes when available (accurate), otherwise file sizes.
    """
    global EMBED_MODEL, TOP_K, embedder

    n = len(source_files)

    if n < 50:
        chosen_model = "all-MiniLM-L6-v2"
    elif n < 200:
        chosen_model = "all-mpnet-base-v2"
    else:
        chosen_model = "nomic-ai/nomic-embed-text-v1.5"

    if chosen_model != EMBED_MODEL or embedder is None:
        EMBED_MODEL = chosen_model
        if quiet:
            embedder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        else:
            with _Spinner(f"{_T_RAG} Loading {_ANSI_BOLD}{chosen_model}{_ANSI_RESET}"):
                embedder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
            print(f"{_T_RAG} {_ANSI_GREEN}Loaded{_ANSI_RESET} {chosen_model}")

    if chunks:
        n_units     = len(chunks)
        total_chars = sum(len(c["text"]) for c in chunks)
    else:
        n_units     = n
        total_chars = sum(f.stat().st_size for f in source_files if f.exists())

    avg_tokens = max(1, (total_chars // n_units) // 4) if n_units else 1
    TOP_K      = max(2, min(15, round(TARGET_RETRIEVED_TOKENS / avg_tokens)))

    unit_label = f"{n_units} chunks" if chunks else f"{n} files"
    if not quiet:
        print(
            f"{_T_RAG} Auto-tune → {_ANSI_BOLD}{n} files{_ANSI_RESET} → {unit_label} | "
            f"~{avg_tokens} tok/chunk | TOP_K={_ANSI_BOLD}{TOP_K}{_ANSI_RESET} | model={EMBED_MODEL}"
        )


def _file_hash(path: Path) -> str:
    """MD5 of file bytes — fast change detection, not cryptographic."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_cache(source_files: list, embed_model: str, quiet=False) -> tuple:
    """
    Load cached embeddings from disk.

    Returns (cached_store, stale_files):
      cached_store — chunk_id → {text, emb} for files whose hash matches the manifest
      stale_files  — list of Path objects that need re-chunking + re-embedding
    """
    try:
        manifest      = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("embed_model") != embed_model:
            raise ValueError("embed model changed — full re-index required")
        old_hashes    = manifest["files"]
        cached_index  = pickle.loads(CACHE_INDEX.read_bytes())
    except Exception as exc:
        if (CACHE_MANIFEST.exists() or CACHE_INDEX.exists()) and not quiet:
            print(f"{_T_CACHE} Miss ({exc}); re-indexing everything.")
        return {}, list(source_files)

    cached_store: dict = {}
    stale: list        = []
    for f in source_files:
        key    = str(f)
        f_hash = _file_hash(f)
        if f_hash == old_hashes.get(key):
            # Copy this file's chunks straight from the disk cache
            prefix = key + "::"
            for cid, val in cached_index.items():
                if cid == key or cid.startswith(prefix):
                    cached_store[cid] = val
        else:
            stale.append(f)

    hit  = len(source_files) - len(stale)
    miss = len(stale)
    if not quiet:
        print(f"{_T_CACHE} {_ANSI_GREEN}{hit}{_ANSI_RESET} files hit, "
              f"{_ANSI_YELLOW}{miss}{_ANSI_RESET} stale/new.")
    return cached_store, stale


def _save_cache(embed_model: str) -> None:
    """Persist chunk_store and file-hash manifest to CACHE_DIR."""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        manifest = {
            "embed_model": embed_model,
            "files": dict(_file_hashes),
        }
        with lock:
            store_snapshot = dict(chunk_store)
        CACHE_INDEX.write_bytes(pickle.dumps(store_snapshot, protocol=pickle.HIGHEST_PROTOCOL))
        CACHE_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    except Exception as e:
        print(f"[Cache] Failed to save: {e}")


def index_files(quiet=False):
    """
    Chunk all supported source files, auto-tune, then embed.
    Files whose MD5 matches the on-disk manifest are loaded from the pickle cache;
    only changed or new files are re-chunked and re-embedded.
    """
    global _source_files, _file_hashes

    if not quiet:
        print(f"{_T_RAG} Scanning project files...", end="", flush=True)
    source_files = [
        p for p in Path(".").rglob("*")
        if not _is_skipped(p) and p.is_file() and p.suffix.lower() in INDEXABLE_EXTENSIONS
    ]
    if not source_files:
        if not quiet:
            print(f"\r{_T_RAG} No supported source files found.")
        return
    if not quiet:
        print(f"\r{_T_RAG} Found {_ANSI_BOLD}{len(source_files)}{_ANSI_RESET} source files.")

    _source_files = source_files
    _file_hashes  = {str(f): _file_hash(f) for f in source_files}

    auto_tune(source_files, quiet=quiet)   # selects + loads embed model before cache check

    cached_store, stale_files = _load_cache(source_files, EMBED_MODEL, quiet=quiet)

    # Chunk + embed only the stale/new files
    new_chunks: list = []

    def process_file_chunks(f):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            return chunk_file(str(f), text)
        except OSError:
            return []

    if stale_files:
        if not quiet:
            print(f"{_T_RAG} Chunking {len(stale_files)} file(s)...", end="", flush=True)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunks in executor.map(process_file_chunks, stale_files):
                new_chunks.extend(chunks)
        if not quiet:
            print(f"\r{_T_RAG} Chunked → {len(new_chunks)} chunks.")
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunks in executor.map(process_file_chunks, stale_files):
                new_chunks.extend(chunks)

    if new_chunks:
        doc_prefix = _DOC_PREFIX.get(EMBED_MODEL, "")
        show_bar   = (len(new_chunks) > 100) and not quiet
        if not show_bar and not quiet:
            print(f"{_T_RAG} Embedding {len(new_chunks)} chunk(s)...", end="", flush=True)
        embeddings = embedder.encode(
            [doc_prefix + c["text"] for c in new_chunks],
            normalize_embeddings=True,
            show_progress_bar=show_bar,
        )
        if not show_bar and not quiet:
            print(f"\r{_T_RAG} Embedded  {len(new_chunks)} chunk(s).  ")
        new_store = {
            c["id"]: {"text": c["text"], "emb": emb}
            for c, emb in zip(new_chunks, embeddings)
        }
    else:
        new_store = {}

    merged = {**cached_store, **new_store}

    # Refine TOP_K using all chunks (cached + new)
    all_chunk_texts = [{"text": v["text"]} for v in merged.values()]
    auto_tune(source_files, chunks=all_chunk_texts, quiet=quiet)

    with lock:
        chunk_store.clear()
        chunk_store.update(merged)

    if new_store:
        _save_cache(EMBED_MODEL)

    cached_n = len(source_files) - len(stale_files)
    if not quiet:
        print(
            f"{_T_RAG} {_ANSI_GREEN}Index ready{_ANSI_RESET} — "
            f"{_ANSI_BOLD}{len(source_files)}{_ANSI_RESET} files → "
            f"{_ANSI_BOLD}{len(merged)}{_ANSI_RESET} chunks "
            f"({_ANSI_GREEN}{cached_n} cached{_ANSI_RESET}, {len(stale_files)} re-embedded)"
        )


def reindex_file(path):
    """Re-chunk and re-embed a single changed source file, then update the disk cache."""
    if embedder is None:
        return
    try:
        p      = Path(path)
        text   = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_file(path, text)

        doc_prefix = _DOC_PREFIX.get(EMBED_MODEL, "")
        embeddings = embedder.encode(
            [doc_prefix + c["text"] for c in chunks],
            normalize_embeddings=True,
        )

        with lock:
            for k in _chunks_for_file(path):
                del chunk_store[k]
            for chunk, emb in zip(chunks, embeddings):
                chunk_store[chunk["id"]] = {"text": chunk["text"], "emb": emb}

        _file_hashes[path] = _file_hash(p)
        _save_cache(EMBED_MODEL)
        print(f"{_T_RAG} Re-indexed {_ANSI_CYAN}{path}{_ANSI_RESET} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"{_T_ERR} Failed to re-index {path}: {e}")


def _chunk_label(chunk_id):
    """'src/Foo.java::doThing' → 'Foo.java::doThing'"""
    if "::" in chunk_id:
        path, method = chunk_id.rsplit("::", 1)
        return f"{Path(path).name}::{method}"
    return Path(chunk_id).name


def _chunks_for_file(filepath):
    """Return all chunk IDs belonging to a given file (whole-file + method chunks)."""
    prefix = filepath + "::"
    return [k for k in chunk_store if k == filepath or k.startswith(prefix)]


_COMMENT_EXTS_C = frozenset({".java", ".js", ".ts", ".tsx", ".go", ".rs"})


def _strip_comments(text: str, ext: str) -> str:
    """Remove comments from a code chunk to cut retrieved-context token usage.

    Strips only code comments — not the file-path header line or preamble —
    so the structural context (package, imports, class declaration) is preserved.
    """
    if ext in _COMMENT_EXTS_C:
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)   # /* block */
        text = re.sub(r'//[^\n]*',  '', text)                    # // line
    elif ext == ".py":
        text = re.sub(r'#[^\n]*', '', text)                      # # line
    # Collapse runs of 3+ blank lines left behind by removal
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _dedup_retrieved_context(top_pairs):
    """
    Build retrieved context string, deduplicating the per-file preamble.

    Chunks from the same file share an identical preamble (package + imports +
    class header).  Instead of repeating it for every method, we emit it once
    and list all retrieved methods underneath — saving 5-20 % of retrieved
    context tokens when multiple methods from the same file rank highly.

    Chunk text format (set by _chunk_with_treesitter):
        // {filepath}\\n{preamble}\\n    // ...\\n{method_body}\\n

    Whole-file fallback chunks (no '::' in their ID) are included as-is.
    Comments are stripped from method bodies and whole-file text before assembly.
    """
    _SEP = "\n    // ...\n"

    # Preserve retrieval order: first chunk seen for each file wins for header.
    file_order  = []                           # filepaths in retrieval order
    file_data   = {}                           # filepath → {header, methods[]}
    whole_files = []

    with lock:
        for chunk_id, _score in top_pairs:
            if chunk_id not in chunk_store:
                continue
            text = chunk_store[chunk_id]["text"]
            if "::" not in chunk_id:
                ext = Path(chunk_id).suffix.lower()
                whole_files.append(_strip_comments(text, ext))
                continue
            filepath, method_name = chunk_id.rsplit("::", 1)
            ext = Path(filepath).suffix.lower()
            if _SEP in text:
                header, body = text.split(_SEP, 1)
            else:
                header, body = f"// {filepath}", text
            if filepath not in file_data:
                file_order.append(filepath)
                file_data[filepath] = {"header": header, "methods": []}
            # Strip comments from the method body only — header holds the preamble
            file_data[filepath]["methods"].append(
                (method_name, _strip_comments(body, ext).strip())
            )

    parts = whole_files[:]
    for filepath in file_order:
        d = file_data[filepath]
        method_blocks = [f"    // {name}\n{body}" for name, body in d["methods"]]
        parts.append(d["header"] + "\n" + "\n\n".join(method_blocks))

    return "\n\n".join(parts)


def retrieve(query, token_budget=None):
    """
    Returns (context_str, hits) where hits = [(chunk_id, score), ...].

    token_budget: target retrieved-context size in tokens. Defaults to
    TARGET_RETRIEVED_TOKENS. k is derived proportionally from TOP_K so that
    the retrieved set fits within the budget.

    Two-stage filtering:
      1. Absolute floor: drop chunks below MIN_SCORE.
      2. Relative floor: drop chunks below RELATIVE_SCORE_FLOOR × top_score,
         eliminating low-relevance stragglers that would fill the budget with noise.
    """
    budget = token_budget or TARGET_RETRIEVED_TOKENS
    base_k = TOP_K or 4
    k = max(2, round(base_k * budget / TARGET_RETRIEVED_TOKENS))

    with lock:
        if not chunk_store:
            return "", []
        ids  = list(chunk_store.keys())
        embs = np.stack([chunk_store[cid]["emb"] for cid in ids])

    query_prefix = _QUERY_PREFIX.get(EMBED_MODEL, "")
    q_emb        = embedder.encode(query_prefix + query, normalize_embeddings=True)
    scores       = embs @ q_emb   # (n_chunks,) — one vectorised op

    # Stage 1: absolute floor + top-k
    top_pairs = [(ids[i], float(scores[i]))
                 for i in np.argsort(-scores)
                 if scores[i] >= MIN_SCORE][:k]

    # Stage 2: relative floor — drop stragglers far below the best match
    if top_pairs:
        threshold = top_pairs[0][1] * RELATIVE_SCORE_FLOOR
        top_pairs = [(cid, s) for cid, s in top_pairs if s >= threshold]

    ctx = _dedup_retrieved_context(top_pairs)
    return ctx, top_pairs


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------

def _build_system_blocks(skeleton):
    """Shared system-block builder used by warm_cache, chat, and one_shot."""
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton, "cache_control": {"type": "ephemeral"}},
    ]
    return blocks


def _apply_skeleton(new_skeleton: str):
    """Store assembled skeleton into shared state under lock."""
    global skeleton_context, last_interaction
    with lock:
        skeleton_context = new_skeleton
        last_interaction = time.time()


def _update_skeleton():
    """Full skeleton rebuild (startup / forced). Updates shared state under lock."""
    _apply_skeleton(build_skeleton())


def warm_cache(quiet=False):
    global session_cost
    with lock:
        skeleton = skeleton_context
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1,
            system=_build_system_blocks(skeleton),
            messages=[{"role": "user", "content": "ok"}],
        )
        cost = calculate_cost(response.usage)
        _accumulate_usage(response.usage)
        with lock:
            session_cost += cost

        if not quiet:
            print(f"\n{_ANSI_DIM}{'═'*56}{_ANSI_RESET}")
            print_stats(response.usage, label="Cache")
            print(f"{_ANSI_DIM}{'═'*56}{_ANSI_RESET}\n")
    except Exception as e:
        print(f"{_T_ERR} Cache warm failed: {e}")


def full_refresh():
    """Startup and heartbeat: rebuild skeleton + re-index all source files + warm cache."""
    with _Spinner("Building skeleton") as spinner:
        _update_skeleton()
        spinner.update("Indexing source files")
        index_files(quiet=True)
        spinner.update("Warming cache")
        warm_cache(quiet=True)


def refresh_skeleton_only():
    """Called on structural changes (new/deleted/moved file): rebuild tree, keep .md cache."""
    _apply_skeleton(_refresh_tree_only())
    warm_cache()


def refresh_md_file(path_str: str):
    """Called when a single .md file changes: re-read only that file if hash changed."""
    if _refresh_single_md(path_str):
        _apply_skeleton(_assemble_skeleton())
        warm_cache()


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

_file_timers: dict = {}   # {filepath: Timer} — ensures only one pending reindex per file


def _debounce(src: str, fn, delay: float = 1.5) -> None:
    """Cancel any pending timer for src and schedule fn after delay seconds."""
    if src in _file_timers:
        _file_timers[src].cancel()
    t = threading.Timer(delay, fn)
    _file_timers[src] = t
    t.start()


def _remove_file_from_index(path: str) -> None:
    """Drop all chunks for path from chunk_store + _file_hashes and persist the cache."""
    with lock:
        for k in _chunks_for_file(path):
            del chunk_store[k]
    _file_hashes.pop(path, None)
    if EMBED_MODEL:
        _save_cache(EMBED_MODEL)


class SourceHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if ext == ".md":
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
            _debounce(src, lambda s=src: reindex_file(s))

    def on_created(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if ext == ".md":
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
            _debounce(src, lambda s=src: reindex_file(s))
            refresh_skeleton_only()   # new source file changes the directory tree

    def on_deleted(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if src in _file_timers:
            _file_timers.pop(src).cancel()
        if ext in INDEXABLE_EXTENSIONS:
            _remove_file_from_index(src)
        if ext == ".md":
            refresh_md_file(src)   # cleans up the md cache entry + rebuilds
        else:
            refresh_skeleton_only()

    def on_moved(self, event):
        if event.is_directory:
            return
        src, dest   = event.src_path, event.dest_path
        src_ext     = Path(src).suffix.lower()
        dest_ext    = Path(dest).suffix.lower()
        for p in (src, dest):
            if p in _file_timers:
                _file_timers.pop(p).cancel()
        if src_ext in INDEXABLE_EXTENSIONS:
            _remove_file_from_index(src)
        if dest_ext in INDEXABLE_EXTENSIONS:
            _debounce(dest, lambda d=dest: reindex_file(d))
        # Clean up any stale .md cache entry for the old path
        if src_ext == ".md":
            _skeleton_md_hashes.pop(src, None)
            _skeleton_md_parts.pop(src, None)
        # If either end is .md, update just that file; otherwise just rebuild tree
        if dest_ext == ".md":
            _debounce(dest, lambda d=dest: refresh_md_file(d))
        else:
            refresh_skeleton_only()


# ---------------------------------------------------------------------------
# Heartbeat — keeps skeleton cache alive; index is maintained by the watcher
# ---------------------------------------------------------------------------

def heartbeat():
    while not stop_event.is_set():
        stop_event.wait(HEARTBEAT_SECS)
        with lock:
            idle = time.time() - last_interaction
        if idle > CACHE_TTL_SECS and not stop_event.is_set():
            warm_cache()


# ---------------------------------------------------------------------------
# Edit helpers — parse, diff, confirm, write
# ---------------------------------------------------------------------------

def _colorize_diff(lines):
    result = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            result.append(f"{_ANSI_GREEN}{line}{_ANSI_RESET}")
        elif line.startswith("-") and not line.startswith("---"):
            result.append(f"{_ANSI_RED}{line}{_ANSI_RESET}")
        elif line.startswith("@@"):
            result.append(f"{_ANSI_CYAN}{line}{_ANSI_RESET}")
        else:
            result.append(line)
    return result


_SR_PATTERN = re.compile(
    r"^<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n?$",
    re.DOTALL,
)


def parse_edit_blocks(text):
    """Return list of edit dicts parsed from Claude's fenced code blocks.

    Each dict has:
      {"type": "edit", "path": str, "search": str, "replace": str}  — SEARCH/REPLACE
      {"type": "new",  "path": str, "content": str}                 — new / full-file
    """
    edits = []
    for m in _ANY_BLOCK.finditer(text):
        path = m.group(1).strip()
        body = m.group(2)
        sr = _SR_PATTERN.match(body)
        if sr:
            edits.append({"type": "edit", "path": path,
                          "search": sr.group(1), "replace": sr.group(2)})
        else:
            edits.append({"type": "new", "path": path, "content": body})
    return edits


def _resolve_new_content(edit):
    """Return (old_content, new_content) for an edit dict, or raise on failure."""
    p = Path(edit["path"])
    if edit["type"] == "new":
        old = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        return old, edit["content"]
    # SEARCH/REPLACE
    if not p.exists():
        raise FileNotFoundError(f"File not found: {edit['path']}")
    old = p.read_text(encoding="utf-8", errors="ignore")
    if edit["search"] not in old:
        raise ValueError(f"SEARCH block not found in {edit['path']}")
    return old, old.replace(edit["search"], edit["replace"], 1)


def show_diff(path, new_content, old_content=None):
    p = Path(path)
    if old_content is None:
        old_content = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if old_content:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        ))
        if diff:
            print("".join(_colorize_diff(line + "\n" for line in diff)))
        else:
            print(f"  {_ANSI_DIM}(no changes detected in {path}){_ANSI_RESET}\n")
    else:
        preview = new_content[:600] + ("…" if len(new_content) > 600 else "")
        print(f"{_ANSI_GREEN}  [NEW FILE]{_ANSI_RESET} {path}\n{preview}\n")


def apply_edits(edits):
    """Show diffs, ask confirmation, apply SEARCH/REPLACE and new-file edits."""
    if not edits:
        print(f"{_T_EDIT} No file blocks found in response.")
        return

    # Pre-resolve: compute old+new content, mark failures
    resolved = []
    for e in edits:
        try:
            old, new = _resolve_new_content(e)
            resolved.append({**e, "_old": old, "_new": new})
        except Exception as ex:
            print(f"{_T_ERR} {ex}")
            resolved.append({**e, "_skip": True})

    applicable = [r for r in resolved if not r.get("_skip")]
    if not applicable:
        print(f"{_T_EDIT} No applicable changes.\n")
        return

    print(f"\n{_ANSI_BOLD}{'═'*56}{_ANSI_RESET}")
    for r in applicable:
        is_new = r["type"] == "new" and not Path(r["path"]).exists()
        tag = f"{_ANSI_GREEN}NEW{_ANSI_RESET}" if is_new else f"{_ANSI_CYAN}EDIT{_ANSI_RESET}"
        print(f"  [{tag}] {r['path']}")
    print(f"{_ANSI_BOLD}{'═'*56}{_ANSI_RESET}\n")

    for r in applicable:
        path = r["path"]
        print(f"{_ANSI_CYAN}── {path} {_ANSI_RESET}" + "─" * max(0, 50 - len(path)))
        show_diff(path, r["_new"], old_content=r["_old"])

    # Non-interactive (piped) → auto-apply
    if not sys.stdin.isatty():
        auto = True
    else:
        print(f"Apply {_ANSI_BOLD}{len(applicable)}{_ANSI_RESET} change(s)? "
              f"[{_ANSI_GREEN}y{_ANSI_RESET}/{_ANSI_RED}n{_ANSI_RESET}] ", end="", flush=True)
        try:
            auto = input().strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            auto = False

    if not auto:
        print(f"{_T_EDIT} Cancelled.\n")
        return

    written = 0
    for r in applicable:
        try:
            p = Path(r["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(r["_new"], encoding="utf-8")
            print(f"{_T_EDIT} Wrote {_ANSI_CYAN}{r['path']}{_ANSI_RESET}")
            written += 1
        except Exception as ex:
            print(f"{_T_ERR} Failed to write {r['path']}: {ex}")

    print(f"{_T_EDIT} {_ANSI_GREEN}{written}/{len(applicable)}{_ANSI_RESET} change(s) applied.\n")


# ---------------------------------------------------------------------------
# History compression — summarize old turns via Haiku instead of dropping them
# ---------------------------------------------------------------------------

def _summarize_turns(messages: list) -> tuple:
    """Call SUMMARY_MODEL to compress a list of turn messages into a short summary."""
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        lines.append(f"{role}: {content}")
    dialogue = "\n\n".join(lines)
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                "Summarize the following conversation turns concisely (under 300 words).\n"
                "Capture: what was asked, decisions reached, files edited and what changed, "
                "key facts established about the codebase.\n"
                "Output only the summary — no preamble.\n\n"
                f"<turns>\n{dialogue}\n</turns>"
            ),
        }],
    )
    return response.content[0].text.strip(), response.usage


def _maybe_compress_history():
    """Summarize the oldest SUMMARIZE_BATCH turns when history exceeds MAX_HISTORY_TURNS."""
    global conversation_history, session_cost
    if len(conversation_history) <= MAX_HISTORY_TURNS * 2:
        return
    batch_msgs   = SUMMARIZE_BATCH * 2
    to_summarize = conversation_history[:batch_msgs]
    remaining    = conversation_history[batch_msgs:]
    try:
        print(f"{_T_SYS} Compressing {SUMMARIZE_BATCH} old turns...", end="", flush=True)
        summary_text, usage = _summarize_turns(to_summarize)
        _accumulate_usage(usage)
        with lock:
            session_cost += calculate_cost(usage)
        summary_pair = [
            {"role": "user",      "content": f"[Summary of earlier conversation]\n{summary_text}"},
            {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
        ]
        conversation_history = summary_pair + remaining
        print(f" done ({usage.input_tokens}→{usage.output_tokens} tok)")
    except Exception as e:
        print(f"\n{_T_ERR} History compression failed ({e}) — truncating instead.")
        conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]


# ---------------------------------------------------------------------------
# Chat — multi-turn with per-turn RAG injection
# ---------------------------------------------------------------------------

def _print_reply(text):
    """Render Claude's reply — rich Markdown when available, plain text otherwise."""
    if _RICH_AVAILABLE:
        console.print()
        console.print(_RichMarkdown(text))
        console.print()
    else:
        print(f"\n{text}\n")


def chat(query):
    global last_interaction, session_cost

    # Route first so the retrieval budget matches the model/effort selection.
    routed_model, effort, max_tok = route_query(query)
    token_budget = _RETRIEVAL_BUDGET[effort]

    retrieved_ctx, hits = retrieve(query, token_budget=token_budget)

    if hits:
        names      = ", ".join(_chunk_label(p) for p, _ in hits)
        scores_str = "  ".join(f"{_chunk_label(p)} {s:.2f}" for p, s in hits)
        print(f"{_T_RAG} {_ANSI_CYAN}{names}{_ANSI_RESET}")
        print(f"      {_ANSI_DIM}{scores_str}{_ANSI_RESET}")

    with lock:
        skeleton = skeleton_context

    # Compress old turns into a summary when history grows too long, then apply
    # a hard sliding-window cap as a fallback safety net.
    _maybe_compress_history()
    trimmed = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    # Add an ephemeral cache breakpoint at the end of the history
    if trimmed:
        last_msg = trimmed[-1]
        if isinstance(last_msg["content"], str):
            trimmed[-1] = {
                "role": last_msg["role"],
                "content": [{"type": "text", "text": last_msg["content"], "cache_control": {"type": "ephemeral"}}]
            }

    # Build current user message as a content-block array so the retrieved
    # chunks get their own ephemeral cache breakpoint (breakpoint ③).
    # On follow-up questions about the same code area the chunks are identical
    # and hit the cache; only the new question text is billed at full price.
    current_blocks: list = []
    if retrieved_ctx:
        current_blocks.append({
            "type": "text",
            "text": f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n",
            "cache_control": {"type": "ephemeral"},
        })
    current_blocks.append({"type": "text", "text": f"Question:\n{query}"})
    messages = trimmed + [{"role": "user", "content": current_blocks}]

    system = _build_system_blocks(skeleton)

    create_kwargs: dict = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=system,
        messages=messages,
    )
    if effort == "max":
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}

    try:
        response = client.messages.create(**create_kwargs)

        reply = _extract_text(response.content)
        conversation_history.append({"role": "user", "content": query})

        edits = parse_edit_blocks(reply)
        # Store a compact version in history so future turns stay lean
        if edits:
            labels = ", ".join(e["path"] for e in edits)
            clean_reply = _ANY_BLOCK.sub("", reply).strip()
            clean_reply = (clean_reply + f"\n[Files edited: {labels}]").strip()
        else:
            clean_reply = reply
        conversation_history.append({"role": "assistant", "content": clean_reply})

        cost = calculate_cost(response.usage)
        _accumulate_usage(response.usage)
        with lock:
            last_interaction = time.time()
            session_cost += cost

        turns = len(conversation_history) // 2

        # Always print explanation text; apply any file blocks that were returned
        explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
        if explanation:
            _print_reply(explanation)
        if edits:
            apply_edits(edits)

        print_stats(response.usage, label=f"Turn {turns}")

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[Error] API call failed: {e}")


# ---------------------------------------------------------------------------
# One-shot (CLI arg / pipe)
# ---------------------------------------------------------------------------

def one_shot(prompt):
    global session_cost

    print(f"{_T_SYS} Building context...", file=sys.stderr)
    _update_skeleton()
    index_files()

    routed_model, effort, max_tok = route_query(prompt)
    retrieved_ctx, hits = retrieve(prompt, token_budget=_RETRIEVAL_BUDGET[effort])
    if hits:
        names = ", ".join(_chunk_label(p) for p, _ in hits)
        print(f"{_T_RAG} {_ANSI_CYAN}{names}{_ANSI_RESET}", file=sys.stderr)

    with lock:
        skeleton = skeleton_context

    ctx_prefix = f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n" if retrieved_ctx else ""
    create_kwargs: dict = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=_build_system_blocks(skeleton),
        messages=[{"role": "user", "content": f"{ctx_prefix}Question:\n{prompt}"}],
    )
    if effort == "max":
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}
    try:
        response = client.messages.create(**create_kwargs)
        reply = _extract_text(response.content)
        edits = parse_edit_blocks(reply)
        explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
        if explanation:
            print(explanation)
        if edits:
            apply_edits(edits)

        cost = calculate_cost(response.usage)
        _accumulate_usage(response.usage)
        with lock:
            session_cost += cost
        print_stats(response.usage, label="Stats", file=sys.stderr)

    except Exception as e:
        raise SystemExit(f"{_T_ERR} API call failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def start_chat():
    full_refresh()

    observer = Observer()
    observer.schedule(SourceHandler(), path=".", recursive=True)
    observer.start()
    threading.Thread(target=heartbeat, daemon=True).start()

    print(f"\n╭── {_ANSI_BOLD}❖ Claude Light{_ANSI_RESET} ──╮")
    print(f"│ {_ANSI_CYAN}{MODEL}{_ANSI_RESET}  |  "
          f"RAG top-{_ANSI_BOLD}{TOP_K}{_ANSI_RESET}  |  "
          f"Embed: {EMBED_MODEL}")
    print(f"╰{'─'*20}╯")
    print(f"{_ANSI_DIM}Commands: /compact  /cost  /help  exit{_ANSI_RESET}\n")

    if _PROMPTTK_AVAILABLE:
        from prompt_toolkit import HTML

        def get_status_bar():
            total_in = session_tokens["input"] + session_tokens["cache_write"] + session_tokens["cache_read"]
            saved = session_tokens["cache_read"]
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            cost = session_cost
            import os
            repo = os.path.basename(os.getcwd())
            
            return HTML(
                f' <b>Repo:</b> <ansicyan>{repo}</ansicyan>  |  '
                f'<b>Tokens:</b> {total_in:,} '
                f'(<ansigreen>{saved:,}</ansigreen> saved, <ansigreen>{ratio:.1f}%</ansigreen>)  |  '
                f'<b>Cost:</b> <ansiyellow>${cost:.4f}</ansiyellow>'
            )

        CACHE_DIR.mkdir(exist_ok=True)
        _slash_completer = _WordCompleter(
            ["/compact", "/clear", "/cost", "/help", "exit", "quit"],
            sentence=True,
        )
        _session = _PromptSession(
            history=_FileHistory(str(CACHE_DIR / "history.txt")),
            auto_suggest=_AutoSuggest(),
            completer=_slash_completer,
            complete_while_typing=False,
            bottom_toolbar=get_status_bar
        )
        def _get_input():
            return _session.prompt("> ").strip()
    else:
        def _get_input():
            total_in = session_tokens["input"] + session_tokens["cache_write"] + session_tokens["cache_read"]
            saved = session_tokens["cache_read"]
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            import os
            print(f"\n[{os.path.basename(os.getcwd())}] Tokens: {total_in:,} ({saved:,} saved, {ratio:.1f}%) | Cost: ${session_cost:.4f}")
            return input("> ").strip()

    try:
        while True:
            try:
                query = _get_input()
            except (KeyboardInterrupt, EOFError):
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            if query in ("/clear", "/compact"):
                conversation_history.clear()
                print(f"{_T_SYS} Conversation history compacted.\n")
                continue
            if query == "/cost":
                print_session_summary()
                continue
            if query == "/help":
                print(
                    "  /compact — reset conversation history\n"
                    "  /cost    — show session spend so far\n"
                    "  exit     — quit\n"
                )
                continue

            chat(query)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        observer.stop()
        observer.join()
        print_session_summary()


# ---------------------------------------------------------------------------
# Test Mode
# ---------------------------------------------------------------------------

class MockManager:
    def __init__(self, preset):
        self.preset = preset
        self.files = {}
        self.total_tokens = 0
        self._generate_synthetic_files()
        
    def _generate_synthetic_files(self):
        configs = {
            "small": (5, 10),
            "medium": (50, 15),
            "large": (200, 20),
            "extra-large": (1000, 20)
        }
        num_files, num_methods = configs.get(self.preset, (5, 10))
        for i in range(num_files):
            file_name = f"src/main/java/com/synthetic/Service{i}.java"
            methods = []
            for m in range(num_methods):
                methods.append(f"""
    public void doTask{m}() {{
        System.out.println("Task {m} in Service {i}");
        for(int j=0; j<10; j++) {{
            // realistic logic simulated here
        }}
    }}""")
            content = f"""package com.synthetic;

import java.util.*;

public class Service{i} {{
    private String name = "Service{i}";
    {"".join(methods)}
}}
"""
            self.files[file_name] = content
            self.total_tokens += len(content) // 4
            
    def start(self):
        from unittest.mock import patch
        import __main__
        
        # Patch Path
        self.path_patcher = patch.object(__main__, "Path", new=self._mock_path_class())
        self.path_patcher.start()
        
        # Patch API
        self.api_patcher = patch.object(__main__.client.messages, "create", side_effect=self._mock_create_message)
        self.api_patcher.start()
        
        # Patch print_stats
        self.orig_print_stats = __main__.print_stats
        self.stats_patcher = patch.object(__main__, "print_stats", side_effect=self._mock_print_stats)
        self.stats_patcher.start()
        
        # Patch Embedder
        self.embedder_patcher = patch.object(__main__, "SentenceTransformer", new=self._mock_embedder_class)
        self.embedder_patcher.start()
        
        print(f"{_T_TEST} Initialized '{self.preset}' preset with {len(self.files)} files (~{self.total_tokens:,} tokens).")
        
    def _mock_path_class(self):
        files = self.files
        
        class MockPath:
            def __init__(self, *args):
                self.path = "/".join(str(p) for p in args).replace("\\", "/")
                self.name = self.path.split("/")[-1]
                self.suffix = "." + self.name.split(".")[-1] if "." in self.name else ""
                self.stem = self.name[:-len(self.suffix)] if self.suffix else self.name
                self.parts = tuple(self.path.split("/"))
            
            def __str__(self):
                return self.path
                
            def __lt__(self, other):
                return self.path < getattr(other, "path", str(other))
                
            def __eq__(self, other):
                return self.path == getattr(other, "path", str(other))
                
            def __hash__(self):
                return hash(self.path)
                
            def rglob(self, pattern):
                for f in files:
                    yield MockPath(f)
                    
            def read_text(self, *args, **kwargs):
                if self.path in files:
                    return files[self.path]
                if self.name.endswith(".md"):
                    return ""
                raise OSError(f"File not found: {self.path}")
                
            def read_bytes(self):
                return self.read_text().encode("utf-8")
                
            def is_file(self):
                return self.path in files or self.name.endswith(".md")
                
            def is_dir(self):
                return not self.is_file()
                
            def exists(self):
                return self.path in files or self.path in (".", ".claude_light_cache") or self.path.startswith("src")
                
            def stat(self):
                class Stat:
                    st_size = len(files.get(self.path, ""))
                return Stat()
                
            def relative_to(self, other):
                return MockPath(self.path)
                
            def mkdir(self, *args, **kwargs):
                pass
                
            def write_text(self, *args, **kwargs):
                pass
                
            def write_bytes(self, *args, **kwargs):
                pass
                
        return MockPath

    def _mock_create_message(self, **kwargs):
        system_blocks = kwargs.get("system", [])
        messages = kwargs.get("messages", [])
        retrieved_ctx = ""
        
        for b in system_blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                text = b.get("text", "")
                if "// src/" in text or "package com." in text:
                    retrieved_ctx += text
        
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                if "// src/" in content or "package com." in content:
                    retrieved_ctx += content
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text = b.get("text", "")
                        if "// src/" in text or "package com." in text:
                            retrieved_ctx += text
                    
        injected_tokens = len(retrieved_ctx) // 4
        full_tokens = self.total_tokens
        
        methods_mentioned = []
        if retrieved_ctx:
            import re
            matches = re.findall(r"public void (doTask\d+)", retrieved_ctx)
            if matches:
                methods_mentioned = list(set(matches[:3]))
                
        response_text = f"Simulated test response. Mentioning retrieved methods: {', '.join(methods_mentioned) if methods_mentioned else 'none'}."
        
        class MockUsage:
            def __init__(self, full, injected):
                self.input_tokens = injected
                self.cache_read_input_tokens = injected
                self.cache_creation_input_tokens = 0
                self.output_tokens = 50
                self._full_codebase_tokens = full
                self._injected_tokens = injected
                
        class MockMessage:
            def __init__(self, text, usage):
                class Content:
                    def __init__(self, t):
                        self.text = t
                self.content = [Content(text)]
                self.usage = usage
                
        return MockMessage(response_text, MockUsage(full_tokens, injected_tokens))
        
    def _mock_embedder_class(self, model_name, **kwargs):
        class MockEmbedder:
            def encode(self, sentences, **kwargs):
                import numpy as np
                dim = 768 if "nomic" in model_name else 384
                if isinstance(sentences, str):
                    return np.random.rand(dim).astype(np.float32)
                return np.random.rand(len(sentences), dim).astype(np.float32)
        return MockEmbedder()
        
    def _mock_print_stats(self, usage, label="Stats", file=sys.stdout):
        # Call the original print_stats
        self.orig_print_stats(usage, label, file)
        
        full_tokens = getattr(usage, "_full_codebase_tokens", self.total_tokens)
        injected = getattr(usage, "_injected_tokens", getattr(usage, "input_tokens", 0) + getattr(usage, "cache_read_input_tokens", 0))
        
        PRICE_INPUT  = 3.00
        PRICE_READ   = 0.30
        
        full_cost = (full_tokens / 1_000_000) * PRICE_INPUT
        rag_cost = (injected / 1_000_000) * PRICE_READ
        savings = full_cost - rag_cost
        savings_pct = (savings / full_cost * 100) if full_cost > 0 else 0.0
        
        print(f"\n[{label}] Token Savings Report:")
        print(f"  If full codebase was sent: {full_tokens:,} tokens (${full_cost:.4f})")
        print(f"  With Claude Light RAG + Cache: {injected:,} tokens (${rag_cost:.4f})")
        print(f"  Total Savings: {savings_pct:.1f}%\n", file=file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Claude Light RAG Chat")
    parser.add_argument("--test-mode", choices=["small", "medium", "large", "extra-large"],
                        help="Run in test mode with a synthetic codebase and mocked API.")
    parser.add_argument("query", nargs="*", help="Optional query for one-shot mode.")
    args, unknown = parser.parse_known_args()

    if args.test_mode:
        manager = MockManager(args.test_mode)
        manager.start()

    query_str = " ".join(args.query).strip()
    if query_str:
        one_shot(query_str)
    elif not sys.stdin.isatty():
        one_shot(sys.stdin.read().strip())
    else:
        start_chat()
