import os
import re
import sys
import time
import difflib
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import anthropic

try:
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

# Prerequisites:
#   pip install sentence-transformers numpy
#   pip install tree-sitter tree-sitter-java tree-sitter-python \
#               tree-sitter-go tree-sitter-rust tree-sitter-javascript tree-sitter-typescript
#   apt install python3-watchdog python3-anthropic
#   Note: sentence-transformers pulls in PyTorch (~1.5 GB on first install)

# --- CONFIG ---
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    raise SystemExit("[Error] Set the ANTHROPIC_API_KEY environment variable.")

MODEL                   = "claude-sonnet-4-6"
HEARTBEAT_SECS          = 30
CACHE_TTL_SECS          = 240
TARGET_RETRIEVED_TOKENS = 6_000   # desired context size per query in tokens
MIN_SCORE               = 0.45    # discard retrieved files below this similarity
MAX_HISTORY_TURNS       = 6       # sliding window — older turns are dropped

# Auto-tuned at runtime — do not set manually
EMBED_MODEL = None
TOP_K       = None
SKIP_DIRS      = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

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


SYSTEM_PROMPT = (
    "You are an expert code assistant. "
    "Answer questions about the codebase provided below. "
    "Be concise and precise."
)

EDIT_INSTRUCTION = """\
You are being asked to make code changes.

For every file you create or modify output the COMPLETE file content in a \
fenced code block tagged with the relative file path:

```java:src/main/java/com/example/Foo.java
// full file content here
```

Rules:
- Always output the COMPLETE file — never partial snippets or diffs.
- Use the correct path relative to the project root.
- You may output multiple blocks for multiple files.
- After all code blocks, write a short plain-English summary of what changed.
"""

# Matches  ```[lang]:path/to/File.java\n...content...\n```
_EDIT_BLOCK = re.compile(r"```[a-zA-Z]*:([^\n`]+)\n(.*?)```", re.DOTALL)

client   = anthropic.Anthropic(api_key=API_KEY)
embedder = None   # initialised by auto_tune() before first use

# nomic-embed requires explicit prefixes for docs vs queries; MiniLM does not
_DOC_PREFIX   = {"nomic-ai/nomic-embed-text-v1.5": "search_document: "}
_QUERY_PREFIX = {"nomic-ai/nomic-embed-text-v1.5": "search_query: "}

# 2026 PRICING (USD per 1 Million Tokens)
PRICE_INPUT  = 3.00
PRICE_WRITE  = PRICE_INPUT * 1.25   # $3.75
PRICE_READ   = PRICE_INPUT * 0.10   # $0.30
PRICE_OUTPUT = 15.00

# Shared state
skeleton_context     = ""   # cached: dir tree + .md files
chunk_store          = {}   # {chunk_id → {"text": str, "emb": np.ndarray}}
                             # chunk_id = "filepath" (whole-file) or "filepath::method"
conversation_history = []   # clean turns; retrieved context is NOT stored here
session_cost         = 0.0
last_interaction     = time.time()
lock                 = threading.Lock()
stop_event           = threading.Event()


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
        f"[{label}]  {total_input:,} tokens  |  "
        f"cached {read_tokens:,} ({hit_pct:.1f}%)  |  new {write_tokens:,}",
        file=file,
    )
    print(
        f"[Cost]   ${actual_cost:.4f}  |  "
        f"saved ${savings:.4f} ({savings_pct:.1f}% vs no-cache)  |  "
        f"session ${session_cost:.4f}",
        file=file,
    )


# ---------------------------------------------------------------------------
# Skeleton — cached system prompt (directory tree + markdown docs)
# ---------------------------------------------------------------------------

def _is_skipped(path):
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)


def build_skeleton():
    """Single rglob pass: builds directory tree + collects .md content."""
    tree_lines, md_parts = [], []
    for path in sorted(Path(".").rglob("*")):
        if _is_skipped(path):
            continue
        depth  = len(path.relative_to(".").parts) - 1
        indent = "  " * depth
        suffix = "/" if path.is_dir() else ""
        tree_lines.append(f"{indent}{path.name}{suffix}")
        if path.suffix == ".md" and path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    md_parts.append(f"<!-- {path} -->\n{text}")
            except OSError:
                pass
    tree = "Directory structure:\n" + "\n".join(tree_lines)
    docs = "\n\n".join(md_parts)
    return tree + ("\n\n" + docs if docs else "")


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


def auto_tune(source_files, chunks=None):
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
        print(f"[RAG] Loading {chosen_model}...")
        EMBED_MODEL = chosen_model
        embedder    = SentenceTransformer(EMBED_MODEL)

    if chunks:
        n_units     = len(chunks)
        total_chars = sum(len(c["text"]) for c in chunks)
    else:
        n_units     = n
        total_chars = sum(f.stat().st_size for f in source_files if f.exists())

    avg_tokens = max(1, (total_chars // n_units) // 4) if n_units else 1
    TOP_K      = max(2, min(15, round(TARGET_RETRIEVED_TOKENS / avg_tokens)))

    unit_label = f"{n_units} chunks" if chunks else f"{n} files"
    print(
        f"[RAG] Auto-tune → {n} files → {unit_label} | "
        f"~{avg_tokens} tokens/chunk | TOP_K={TOP_K} | model={EMBED_MODEL}"
    )


def index_files():
    """Chunk all supported source files into symbols, auto-tune, then batch-embed."""
    source_files = [
        p for p in Path(".").rglob("*")
        if not _is_skipped(p) and p.is_file() and p.suffix.lower() in INDEXABLE_EXTENSIONS
    ]
    if not source_files:
        print("[RAG] No supported source files found.")
        return

    auto_tune(source_files)          # load model first

    print(f"[RAG] Chunking {len(source_files)} files...")
    all_chunks = []
    for f in source_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            all_chunks.extend(chunk_file(str(f), text))
        except OSError:
            pass

    auto_tune(source_files, chunks=all_chunks)   # refine TOP_K now we know chunk sizes

    doc_prefix = _DOC_PREFIX.get(EMBED_MODEL, "")
    to_embed   = [doc_prefix + c["text"] for c in all_chunks]
    embeddings = embedder.encode(
        to_embed, normalize_embeddings=True,
        show_progress_bar=len(all_chunks) > 100,
    )

    new_store = {
        c["id"]: {"text": c["text"], "emb": emb}
        for c, emb in zip(all_chunks, embeddings)
    }
    with lock:
        chunk_store.clear()
        chunk_store.update(new_store)
    print(f"[RAG] Index ready — {len(source_files)} files → {len(new_store)} chunks")


def reindex_file(path):
    """Re-chunk and re-embed a single changed source file."""
    if embedder is None:
        return
    try:
        text   = Path(path).read_text(encoding="utf-8", errors="ignore")
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

        print(f"[RAG] Re-indexed {path} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"[RAG] Failed to re-index {path}: {e}")


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


def retrieve(query, k=None):
    """
    Returns (context_str, hits) where hits = [(chunk_id, score), ...].
    k defaults to the auto-tuned TOP_K.
    Uses vectorised matrix multiply instead of per-chunk dot products.
    """
    k = k or TOP_K or 4
    with lock:
        if not chunk_store:
            return "", []
        ids  = list(chunk_store.keys())
        embs = np.stack([chunk_store[cid]["emb"] for cid in ids])

    query_prefix = _QUERY_PREFIX.get(EMBED_MODEL, "")
    q_emb        = embedder.encode(query_prefix + query, normalize_embeddings=True)
    scores       = embs @ q_emb   # (n_chunks,) — one vectorised op

    top_pairs = [(ids[i], float(scores[i]))
                 for i in np.argsort(-scores)
                 if scores[i] >= MIN_SCORE][:k]

    with lock:
        ctx = "\n\n".join(chunk_store[p]["text"] for p, _ in top_pairs if p in chunk_store)
    return ctx, top_pairs


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------

def _build_system_blocks(skeleton, retrieved_ctx=None):
    """Shared system-block builder used by warm_cache, chat, and one_shot."""
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton, "cache_control": {"type": "ephemeral"}},
    ]
    if retrieved_ctx:
        blocks.append({"type": "text", "text": retrieved_ctx,
                        "cache_control": {"type": "ephemeral"}})
    return blocks


def _update_skeleton():
    """Rebuild skeleton and update shared state under lock."""
    global skeleton_context, last_interaction
    new_skeleton = build_skeleton()
    with lock:
        skeleton_context = new_skeleton
        last_interaction = time.time()


def warm_cache():
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
        with lock:
            session_cost += cost

        print(f"\n{'═'*56}")
        print_stats(response.usage, label="Cache")
        print(f"{'═'*56}\n")
    except Exception as e:
        print(f"[System] Cache warm failed: {e}")


def full_refresh():
    """Startup and heartbeat: rebuild skeleton + re-index all source files + warm cache."""
    print("[System] Full refresh...")
    _update_skeleton()
    index_files()
    warm_cache()


def refresh_skeleton_only():
    """Called when a .md file changes: rebuild skeleton + re-warm cache."""
    _update_skeleton()
    warm_cache()


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

_file_timers: dict = {}   # {filepath: Timer} — ensures only one pending reindex per file

class SourceHandler(FileSystemEventHandler):
    def on_modified(self, event):
        src = event.src_path
        ext = Path(src).suffix.lower()
        if src in _file_timers:
            _file_timers[src].cancel()
        if ext == ".md":
            t = threading.Timer(1.5, refresh_skeleton_only)
            _file_timers[src] = t
            t.start()
        elif ext in INDEXABLE_EXTENSIONS:
            t = threading.Timer(1.5, lambda s=src: reindex_file(s))
            _file_timers[src] = t
            t.start()


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

_ANSI_RED   = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_CYAN  = "\033[36m"
_ANSI_RESET = "\033[0m"


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


def parse_edit_blocks(text):
    """Return list of (filepath, content) from Claude's fenced code blocks."""
    return [(m.group(1).strip(), m.group(2)) for m in _EDIT_BLOCK.finditer(text)]


def show_diff(path, new_content):
    p = Path(path)
    if p.exists():
        old_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        ))
        if diff:
            print("".join(_colorize_diff(line + "\n" for line in diff)))
        else:
            print(f"  (no changes detected in {path})\n")
    else:
        preview = new_content[:600] + ("…" if len(new_content) > 600 else "")
        print(f"{_ANSI_GREEN}  [NEW FILE]{_ANSI_RESET} {path}\n{preview}\n")


def apply_edits(edits):
    """Show diffs, ask confirmation, write files."""
    if not edits:
        print("[Edit] No file blocks found in response.")
        return

    print(f"\n{'═'*56}")
    for path, _ in edits:
        tag = f"{_ANSI_GREEN}NEW{_ANSI_RESET}" if not Path(path).exists() else f"{_ANSI_CYAN}MOD{_ANSI_RESET}"
        print(f"  [{tag}] {path}")
    print(f"{'═'*56}\n")

    for path, content in edits:
        print(f"── {path} " + "─" * max(0, 50 - len(path)))
        show_diff(path, content)

    # Non-interactive (piped) → auto-apply
    if not sys.stdin.isatty():
        auto = True
    else:
        print(f"Apply {len(edits)} change(s)? [y/n] ", end="", flush=True)
        try:
            auto = input().strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            auto = False

    if not auto:
        print("[Edit] Cancelled.\n")
        return

    written = 0
    for path, content in edits:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            print(f"[Edit] Wrote {path}")
            written += 1
        except Exception as e:
            print(f"[Edit] Failed to write {path}: {e}")

    print(f"[Edit] {written}/{len(edits)} file(s) written.\n")


# ---------------------------------------------------------------------------
# Chat — multi-turn with per-turn RAG injection
# ---------------------------------------------------------------------------

def chat(query, edit_mode=False):
    global last_interaction, session_cost

    retrieved_ctx, hits = retrieve(query)

    if hits:
        names      = ", ".join(_chunk_label(p) for p, _ in hits)
        scores_str = "  ".join(f"{_chunk_label(p)} {s:.2f}" for p, s in hits)
        print(f"[RAG] {names}")
        print(f"      {scores_str}")

    with lock:
        skeleton = skeleton_context

    # Sliding window: keep only the most recent MAX_HISTORY_TURNS turns so
    # history tokens don't grow unboundedly.
    trimmed    = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    # In edit mode, prepend the formatting instruction to the user content.
    # Store only the clean query in history so future turns stay lean.
    prefix  = f"{EDIT_INSTRUCTION}\n\n" if edit_mode else ""
    content = f"{prefix}{query}"
    messages   = trimmed + [{"role": "user", "content": content}]
    max_tokens = 8192 if edit_mode else 2048   # full-file writes need more headroom

    system = _build_system_blocks(skeleton, retrieved_ctx)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

        reply = response.content[0].text
        conversation_history.append({"role": "user",      "content": query})
        conversation_history.append({"role": "assistant", "content": reply})

        cost = calculate_cost(response.usage)
        with lock:
            last_interaction = time.time()
            session_cost += cost

        turns = len(conversation_history) // 2

        if edit_mode:
            # Print Claude's explanation (text outside the code blocks)
            explanation = _EDIT_BLOCK.sub("", reply).strip()
            if explanation:
                print(f"\nClaude: {explanation}\n")
            apply_edits(parse_edit_blocks(reply))
        else:
            print(f"\nClaude: {reply}\n")

        print_stats(response.usage, label=f"Turn {turns}")

    except Exception as e:
        print(f"\n[Error] API call failed: {e}")


# ---------------------------------------------------------------------------
# One-shot (CLI arg / pipe)
# ---------------------------------------------------------------------------

def one_shot(prompt):
    global session_cost

    print("[System] Building context...", file=sys.stderr)
    _update_skeleton()
    index_files()

    retrieved_ctx, hits = retrieve(prompt)
    if hits:
        names = ", ".join(_chunk_label(p) for p, _ in hits)
        print(f"[RAG] {names}", file=sys.stderr)

    with lock:
        skeleton = skeleton_context

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=_build_system_blocks(skeleton, retrieved_ctx),
            messages=[{"role": "user", "content": prompt}],
        )
        print(response.content[0].text)

        cost = calculate_cost(response.usage)
        with lock:
            session_cost += cost
        print_stats(response.usage, label="Stats", file=sys.stderr)

    except Exception as e:
        raise SystemExit(f"[Error] API call failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def start_chat():
    full_refresh()

    observer = Observer()
    observer.schedule(SourceHandler(), path=".", recursive=True)
    observer.start()
    threading.Thread(target=heartbeat, daemon=True).start()

    print(f"Ready. (Claude: {MODEL}  |  RAG top-{TOP_K}  |  Embed: {EMBED_MODEL})")
    print("Commands: /clear  /cost  /help  exit\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        if query == "/clear":
            conversation_history.clear()
            print("[System] Conversation history cleared.\n")
            continue
        if query == "/cost":
            print(f"[Cost] Session total: ${session_cost:.4f}  |  Turns: {len(conversation_history) // 2}\n")
            continue
        if query == "/help":
            print(
                "  /edit <prompt>  — ask Claude to write changes, review diff, confirm write\n"
                "  /clear          — reset conversation history\n"
                "  /cost           — show session spend so far\n"
                "  exit            — quit\n"
            )
            continue

        if query.startswith("/edit "):
            chat(query[6:].strip(), edit_mode=True)
            continue

        chat(query)

    stop_event.set()
    observer.stop()
    observer.join()
    print(f"\n[Session] Total cost: ${session_cost:.4f}  |  Turns: {len(conversation_history) // 2}")


if __name__ == "__main__":
    # One-shot:    python3 script.py "your question"
    # Pipe:        echo "your question" | python3 script.py
    # Interactive: python3 script.py
    if len(sys.argv) > 1:
        one_shot(" ".join(sys.argv[1:]))
    elif not sys.stdin.isatty():
        one_shot(sys.stdin.read().strip())
    else:
        start_chat()
