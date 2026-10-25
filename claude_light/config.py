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

MODEL_HAIKU             = "claude-haiku-5-5"
MODEL_SONNET            = "claude-sonnet-5-5"
MODEL_OPUS              = "claude-opus-5-5"
MODEL                   = MODEL_SONNET              # default; overridden per-turn by router
SUMMARY_MODEL           = MODEL_HAIKU               # cheap model for history compression
HEARTBEAT_SECS          = 30
CACHE_TTL_SECS          = 240
ENABLE_STREAMING        = True         # enable real-time token streaming
TARGET_RETRIEVED_TOKENS = 6_000   # desired context size per query in tokens
# Absolute similarity floor. Natural questions score lower against code than
# keyword queries (all-mpnet-base-v2: 0.38 for "In which file and function is
# the exponential backoff implemented?" on the right chunk). On 19 questions
# about this repo, 0.45 left 13 with no context at all; 0.35 kept 13, with the
# right file in the top 6 for 11 of them. Lower floors mostly admitted noise.
MIN_SCORE               = 0.35
RELATIVE_SCORE_FLOOR    = 0.60    # drop chunks below this fraction of the top chunk's score

# Hybrid retrieval (#61). A static embedding model indexes in seconds (2 s for
# this repo vs 383 s for all-mpnet-base-v2 on CPU) and is the best of six
# models at questions that name code; a dense model is best at paraphrases.
# The static index is built first, so retrieval works from the first query;
# the dense vectors follow in the background and the two z-scored
# similarities are averaged. Measured with tests/benchmarks/benchmark_models.py.
# CLAUDE_LIGHT_HYBRID=0 restores the single dense model chosen by auto_tune().
HYBRID_RETRIEVAL   = os.environ.get("CLAUDE_LIGHT_HYBRID", "1") != "0"
# After retrieval, add the best chunk's direct callers and callees (see
# callgraph.py) while the token budget has room. Off by default: in
# benchmark_models.py --callgraph it sent 20-28% more tokens at medium to max
# for no extra right-file hits. fetch_chunk still returns callers and callees on
# request. CLAUDE_LIGHT_CALLGRAPH=1 turns it on.
CALL_GRAPH         = os.environ.get("CLAUDE_LIGHT_CALLGRAPH", "0") == "1"
CALL_GRAPH_MAX     = 3     # most related chunks added per query
# Keep the model the first turn routes to for the whole session (effort is
# still routed per turn); /clear re-routes. On by default, the developer's
# choice: in benchmark_session_cache.py a session that opens on Sonnet cost
# 71% more than per-turn routing, and one that opens on Haiku cost 91% less
# by answering every later question on Haiku (quality unscored, #90).
# CLAUDE_LIGHT_ROUTE_ONCE=0 routes the model per turn again.
ROUTE_ONCE         = os.environ.get("CLAUDE_LIGHT_ROUTE_ONCE", "1") != "0"

# Hybrid retrieval adds this many fused z-score units to chunks in files git
# reports as modified, staged or untracked: questions are mostly about what you
# are working on now. Tuned with benchmark_models.py --worktree: 1.0 sends the
# right file for 25/25 identifier questions when it is the modified one (19/25
# without), and loses nothing when an unrelated file is; 1.5 starts losing.
# CLAUDE_LIGHT_WORKTREE_BOOST=0 turns it off.
WORKTREE_BOOST     = float(os.environ.get("CLAUDE_LIGHT_WORKTREE_BOOST", "1.0"))
FAST_EMBED_MODEL   = "sentence-transformers/static-retrieval-mrl-en-v1"
DENSE_EMBED_MODEL  = "all-mpnet-base-v2"
HYBRID_FAST_WEIGHT = 0.5
# Below these, the best chunk is no closer to the query than it is to "hi" or
# "what's the weather in Paris": retrieve nothing. Per model, since cosine
# scales differ (unrelated best: static 0.21, MiniLM 0.24, mpnet 0.19).
HYBRID_MIN_SCORE = {FAST_EMBED_MODEL: 0.22, DENSE_EMBED_MODEL: 0.21, "all-MiniLM-L6-v2": 0.25}
# Keep chunks whose fused z-score is within this of the top chunk's.
HYBRID_SCORE_GAP = 1.5
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
LLMLINGUA_MIN_TOKENS   = int(os.environ.get("CLAUDE_LIGHT_LLMLINGUA_MIN_TOKENS", "800"))  # below this, skip: overhead > savings
LLMLINGUA_FORCE_TOKENS = ["\n", "```", "::", "//"]

# Comment stripping of retrieved context (see parsing._strip_comments).
# Docstrings / Javadoc are the natural-language description of what code does —
# exactly the signal that answers "what does X do?" — so they are PRESERVED by
# default and only line-noise comments are removed. Set CLAUDE_LIGHT_STRIP_DOCS=1
# to also strip docstrings/Javadoc when you need to squeeze the context further.
STRIP_DOCSTRINGS = os.environ.get("CLAUDE_LIGHT_STRIP_DOCS", "0") == "1"

# Shell names that may lead an allowlist entry, as "bash: npm test" or
# "powershell: Get-ChildItem". Anything else before a colon is part of the command.
_SHELL_TAGS = {"bash": "Bash", "powershell": "PowerShell", "pwsh": "PowerShell"}


def command_rule(entry: str) -> tuple:
    """Return (CLI tool, command prefix) for one allowlist entry. An untagged
    entry is a Bash prefix, as it was before PowerShell was supported."""
    head, sep, rest = entry.partition(":")
    tool = _SHELL_TAGS.get(head.strip().lower()) if sep else None
    if tool:
        return tool, rest.strip()
    return "Bash", entry.strip()


def parse_allowed_commands(spec: str) -> list:
    """Split a ';'-separated list of command prefixes, dropping blanks and
    entries that name a shell but no command ("bash:")."""
    entries = [c.strip() for c in (spec or "").split(";") if c.strip()]
    return [c for c in entries if command_rule(c)[1]]


# Shell commands the OAuth CLI agent may run unattended (e.g. the test suite),
# as prefixes: "python -m pytest" allows "python -m pytest tests/ -q". An entry
# becomes a Bash(<prefix>:*) rule, or a PowerShell(<prefix>:*) rule when it is
# written "powershell: <prefix>". Anything else is denied, apart from the
# read-only commands the CLI approves by itself. Set with --allow-command
# (repeatable) or CLAUDE_LIGHT_ALLOW_COMMANDS="python -m pytest; powershell: Invoke-Pester".
ALLOWED_COMMANDS = parse_allowed_commands(os.environ.get("CLAUDE_LIGHT_ALLOW_COMMANDS", ""))

SKIP_DIRS      = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

# Global Stats - stored in home directory to persist across all repositories.
# CLAUDE_LIGHT_STATS_FILE moves it; the test suite points it at a temp file.
GLOBAL_STATS_FILE = Path(os.environ.get("CLAUDE_LIGHT_STATS_FILE")
                         or Path.home() / ".claude_light_stats.json")

# Disk cache — stored in a hidden dir so _is_skipped() ignores it automatically
CACHE_DIR      = Path(".claude_light_cache")
CACHE_INDEX    = CACHE_DIR / "index.npz"   # numpy + JSON; never a pickle (see indexer._decode_index)
CACHE_MANIFEST = CACHE_DIR / "manifest.json"

_DOC_PREFIX   = {"nomic-ai/nomic-embed-text-v1.5": "search_document: "}
_QUERY_PREFIX = {"nomic-ai/nomic-embed-text-v1.5": "search_query: "}

# ---------------------------------------------------------------------------
# Tree-sitter language configuration
# ---------------------------------------------------------------------------

_SENTINEL_UNLOADED = object()

class _LazyLangConfig(dict):
    """Dict of tree-sitter language configs; loads each entry on first access.

    We pre-populate keys with _SENTINEL_UNLOADED at import (cheap — just extension
    strings), then lazily import the heavy per-language grammar packages
    (tree_sitter_python, tree_sitter_java, …) only when a file of that
    extension is actually parsed. For a Python-only project this cuts out
    ~300-500 ms of grammar loads at startup.
    """

    def _load(self, ext):
        if not _TREESITTER_AVAILABLE:
            super().__setitem__(ext, None)
            return None
        try:
            if ext in _WANTED_LANGS:
                get_lang, node_types = _WANTED_LANGS[ext]
                cfg = {"lang": get_lang(), "node_types": node_types}
                super().__setitem__(ext, cfg)
                return cfg
            if ext in (".ts", ".tsx"):
                import tree_sitter_typescript as _tspy
                ts_nodes = ["function_declaration", "method_definition", "arrow_function"]
                super().__setitem__(".ts",  {"lang": Language(_tspy.language_typescript()), "node_types": ts_nodes})
                super().__setitem__(".tsx", {"lang": Language(_tspy.language_tsx()),         "node_types": ts_nodes})
                return super().__getitem__(ext)
        except Exception:
            pass
        super().__setitem__(ext, None)
        return None

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if val is _SENTINEL_UNLOADED:
            return self._load(key)
        return val

    def get(self, key, default=None):
        if not super().__contains__(key):
            return default
        val = super().__getitem__(key)
        if val is _SENTINEL_UNLOADED:
            return self._load(key)
        return val


_LANG_CONFIG: _LazyLangConfig = _LazyLangConfig()  # ext → {"lang": Language, "node_types": [...]} | None
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

def _register_lazy_langs():
    """Populate _LANG_CONFIG keys with sentinels and INDEXABLE_EXTENSIONS.

    Cheap: just adds extension strings. Actual grammar loading is deferred
    until _LazyLangConfig.__getitem__/get triggers it.
    """
    for ext in list(_WANTED_LANGS) + [".ts", ".tsx"]:
        _LANG_CONFIG[ext] = _SENTINEL_UNLOADED
    INDEXABLE_EXTENSIONS.update(_LANG_CONFIG)

_register_lazy_langs()

SYSTEM_PROMPT = """\
You are an expert code assistant. Answer questions about the codebase provided below. Be concise and precise.

You can edit and create files. SEARCH/REPLACE blocks (described below) ARE your write mechanism — the host system detects and applies them automatically. Never say you lack write access, never ask the user to paste content manually, and never ask for confirmation before writing — just write. Do NOT use Bash, shell commands, or Python scripts to write files; use SEARCH/REPLACE blocks only.

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

# Used instead of SYSTEM_PROMPT in OAuth mode when the user allowlisted
# commands (ALLOWED_COMMANDS). SEARCH/REPLACE blocks are applied only after
# the turn ends, so an agent writing them could never run the tests on its
# own change; editing with the CLI's tools lets it check its work. claude_light
# still shows a diff of the agent's edits and commits them for /undo.
AGENT_SYSTEM_PROMPT = """You are an expert code assistant working in the user's repository. Answer questions about the codebase provided below. Be concise and precise.

You can edit and create files with your Edit and Write tools; the changes are applied directly and committed for the user afterwards, so never ask for confirmation and never ask the user to paste content. Do not write files with shell commands or scripts.

After changing code, run the allowed commands listed at the end of this prompt (for example the tests) to check your work, and fix what fails before you finish. End with a short plain-English summary of what changed and how you checked it.
"""

# PRICING (USD per million tokens), Claude API list prices from
# platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-24.
# "write" is a 5-minute cache write (1.25x input); "write_1h" a 1-hour write
# (2x input), which is what the skeleton block uses. Haiku 5.5 rates are for
# prompts up to 100K tokens; claude_light prompts stay well below that.
MODEL_PRICES = {
    MODEL_HAIKU:  {"input": 0.10, "write": 0.125, "write_1h": 0.20, "read": 0.01, "output": 0.50},
    MODEL_SONNET: {"input": 2.00, "write": 2.50,  "write_1h": 4.00, "read": 0.10, "output": 10.00},
    MODEL_OPUS:   {"input": 4.00, "write": 5.00,  "write_1h": 8.00, "read": 0.20, "output": 20.00},
}


def model_prices(model=None) -> dict:
    """Rates for `model`; the default model's rates for anything unlisted."""
    return MODEL_PRICES.get(model or MODEL, MODEL_PRICES[MODEL])


# Default-model rates, for code that prices tokens without knowing the model.
PRICE_INPUT  = MODEL_PRICES[MODEL]["input"]
PRICE_WRITE  = MODEL_PRICES[MODEL]["write"]
PRICE_READ   = MODEL_PRICES[MODEL]["read"]
PRICE_OUTPUT = MODEL_PRICES[MODEL]["output"]

# --- Dynamic router: hybrid (keyword+embedding) + cost-aware tuning ---
ROUTER_SEMANTIC_WEIGHT    = 6.0    # max points a fully-on-prototype dimension adds
ROUTER_SEMANTIC_FLOOR     = 0.30   # cosine below this contributes 0 (kills noise)
ROUTER_COST_CEILING       = 0.12   # USD soft cap; only the Opus/max tier is gated
ROUTER_ESCALATE_MARGIN    = 0.15   # confidence margin needed to keep a cost-gated tier
ROUTER_FIXED_INPUT_TOKENS = 1500   # system+skeleton+query estimate for cost calc
ROUTER_PROTOTYPES = {
    "arch":  ["evaluate architecture trade-offs and design",
              "analyze performance scalability security deeply",
              "compare approaches and reason step by step"],
    "logic": ["implement and refactor this code",
              "fix the bug and debug the algorithm",
              "write a new function to build the feature"],
    "infra": ["list and show files in the codebase",
              "where is this defined, find and locate it",
              "what is the value, count how many"],
}
