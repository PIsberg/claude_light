# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`claude_light.py` is an interactive CLI chat tool for querying and editing a multi-language codebase using Claude. It uses a hybrid RAG + prompt caching strategy: project docs and directory structure are cached as a system prompt, while source files are split into method/function-level chunks via tree-sitter, embedded, and retrieved per-query.

## Running

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 claude_light.py

# One-shot (no interactive loop)
python3 claude_light.py "What does OrderService do?"

# Pipe
echo "List all REST endpoints" | python3 claude_light.py

# Simulation / test mode (no API key or costs required)
python3 claude_light.py --test-mode small    # 5 files, 10 methods each
python3 claude_light.py --test-mode medium   # 50 files, 15 methods each
python3 claude_light.py --test-mode large    # 200 files, 20 methods each
python3 claude_light.py --test-mode extra-large  # 1000 files, 20 methods each
```

Run from the root of a project. Exits with an error if `ANTHROPIC_API_KEY` is not set (unless using `--test-mode`).

**API key resolution order:** environment variable → `~/.anthropic` file → `./.env` file in current directory.

## Dependencies

**Required:**
```bash
pip install sentence-transformers numpy
apt install python3-watchdog python3-anthropic
# Note: sentence-transformers pulls in PyTorch (~1.5 GB on first install)
```

**Optional (strongly recommended):**
```bash
pip install tree-sitter tree-sitter-java tree-sitter-python \
    tree-sitter-go tree-sitter-rust tree-sitter-javascript tree-sitter-typescript
pip install rich          # formatted markdown output for Claude responses
pip install prompt_toolkit  # command history, auto-complete, auto-suggest
```

Without tree-sitter, chunking falls back to whole-file mode. Without rich/prompt_toolkit, the tool degrades gracefully to plain text output and basic `input()`.

## Architecture

**Hybrid RAG + prompt caching.** Two tiers:

| Tier | Content | Strategy |
|---|---|---|
| Skeleton (cached system prompt) | Directory tree + all `.md` files | Ephemeral cache — paid once, re-read cheaply |
| Source files | Method/function-level chunks via tree-sitter | RAG — top-K chunks retrieved per query |

**Three concurrent threads:**

1. **Main thread** — input loop; retrieves relevant chunks per query via `retrieve()`, injects them as a second cached system block, stores only the clean query in `conversation_history`
2. **Watchdog** (`SourceHandler`) — source file change → `reindex_file()` (re-chunks + re-embeds that file, cancels any pending timer first via 1.5 s debounce); `.md` change → `refresh_skeleton_only()`
3. **Heartbeat daemon** — checks every 30 s; if idle > 4 min (`CACHE_TTL_SECS`), calls `warm_cache()` to keep the ephemeral cache alive

**Multi-language support** (`_WANTED_LANGS`): tree-sitter grammars extract method/function nodes from `.java`, `.py`, `.js`, `.go`, `.rs`, `.ts`, `.tsx`. Falls back to whole-file chunking if tree-sitter is unavailable or a grammar is missing. Python decorators are included in their function chunk.

**Method-level chunking** (`chunk_java_file` / tree-sitter path): Each chunk includes a file path comment and the class preamble (package + imports + class header) so it is self-contained. Overloaded method names get numeric suffixes. Chunk IDs use `filepath::methodName` (`::` is not valid in source identifiers). Falls back to whole-file if no methods are found.

**Disk cache** (`.claude_light_cache/`): Embeddings are persisted to `index.pkl` between runs; a `manifest.json` tracks per-file MD5 hashes and the embedding model used. Only changed or new files are re-embedded on restart. Command history (prompt_toolkit) is also stored here.

**Auto-tuning** (`auto_tune`): called twice in `index_files()` — first to select and load the embedding model (based on file count), then again after chunking to set `TOP_K` based on actual chunk sizes, targeting `TARGET_RETRIEVED_TOKENS` of context per query.

| File count | Embedding model |
|---|---|
| < 50 | `all-MiniLM-L6-v2` (22 MB) |
| 50–199 | `all-mpnet-base-v2` (420 MB) |
| 200+ | `nomic-ai/nomic-embed-text-v1.5` |

**Scoring**: `retrieve()` uses a single matrix multiply (`embs @ q_emb`) over all chunk embeddings — no per-chunk loop. Chunks below `MIN_SCORE` (0.45) are dropped before sending.

**Token optimisations:**
- Skeleton cached; retrieved chunks also cached (second `cache_control` block) — repeated queries over the same code pay $0.30/M instead of $3.00/M
- Conversation history capped at `MAX_HISTORY_TURNS` (6) via sliding window
- `build_skeleton()` does a single `rglob("*")` pass to build both the directory tree and collect `.md` content (`.md` files > 5 000 chars are truncated except `CLAUDE.md` / `agents.md`)

**Key helpers:**
- `_build_system_blocks(skeleton, retrieved_ctx=None)` — constructs the system prompt list for all API calls
- `_update_skeleton()` — rebuilds skeleton and updates shared state under lock
- `_chunks_for_file(filepath)` — returns all chunk IDs for a file (used by `reindex_file`)
- `_print_reply(text)` — renders Claude's response as rich markdown when available, plain text otherwise

**Interactive commands**: `/clear`, `/cost`, `/help`, `exit`/`quit`, `Ctrl+C`.

Every prompt is handled the same way — Claude decides whether to answer in prose or return file edits. When the response contains ` ```lang:path ``` ` blocks, the script diffs (with ANSI colour), confirms, and writes them automatically. One-shot and piped modes auto-apply without confirmation.


## Key Config (top of script)

| Variable | Purpose |
|---|---|
| `TARGET_RETRIEVED_TOKENS` | Token budget for retrieved context per query (default 6 000) |
| `MIN_SCORE` | Minimum cosine similarity to include a chunk (default 0.45) |
| `MAX_HISTORY_TURNS` | Sliding-window size for conversation history (default 6) |
| `HEARTBEAT_SECS` | How often the heartbeat thread wakes (default 30) |
| `CACHE_TTL_SECS` | Idle seconds before heartbeat warms the cache (default 240) |
| `MODEL` | Claude model ID (default `claude-sonnet-4-5`) |
| `PRICE_INPUT` / `PRICE_WRITE` / `PRICE_READ` / `PRICE_OUTPUT` | Token pricing constants ($3.00 / $3.75 / $0.30 / $15.00 per M) |
| `SKIP_DIRS` | Directory names excluded from indexing (e.g. `.git`, `node_modules`) |
| `EMBED_MODEL`, `TOP_K` | Set at runtime by `auto_tune()` — do not set manually |
