# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`claude_light.py` is an interactive CLI chat tool for querying and editing a Java codebase using Claude. It uses a hybrid RAG + prompt caching strategy: project docs and directory structure are cached as a system prompt, while Java source files are split into method-level chunks, embedded, and retrieved per-query.

## Running

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 claude_light.py

# One-shot (no interactive loop)
python3 claude_light.py "What does OrderService do?"

# Pipe
echo "List all REST endpoints" | python3 claude_light.py
```

Run from the root of a Java project. Exits with an error if `ANTHROPIC_API_KEY` is not set.

## Dependencies

```bash
pip install sentence-transformers numpy
apt install python3-watchdog python3-anthropic
# Note: sentence-transformers pulls in PyTorch (~1.5 GB on first install)
```

## Architecture

**Hybrid RAG + prompt caching.** Two tiers:

| Tier | Content | Strategy |
|---|---|---|
| Skeleton (cached system prompt) | Directory tree + all `.md` files | Ephemeral cache — paid once, re-read cheaply |
| Java sources | Method-level chunks from all `.java` files | RAG — top-K chunks retrieved per query |

**Three concurrent threads:**

1. **Main thread** — input loop; retrieves relevant chunks per query via `retrieve()`, injects them as a second cached system block, stores only the clean query in `conversation_history`
2. **Watchdog** (`SourceHandler`) — `.java` change → `reindex_file()` (re-chunks + re-embeds that file, cancels any pending timer first); `.md` change → `refresh_skeleton_only()`
3. **Heartbeat daemon** — checks every 30s; if idle >4 min, calls `warm_cache()` to keep the ephemeral cache alive

**Method-level chunking** (`chunk_java_file`): brace-depth scanner splits each `.java` file into one chunk per method/constructor. Each chunk includes the preamble (package + imports + class header) so it is self-contained. Chunk IDs use `filepath::methodName` (`::` is not valid in Java identifiers). Falls back to whole-file if no methods are found.

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
- `build_skeleton()` does a single `rglob("*")` pass to build both the directory tree and collect `.md` content

**Key helpers:**
- `_build_system_blocks(skeleton, retrieved_ctx=None)` — constructs the system prompt list for all API calls
- `_update_skeleton()` — rebuilds skeleton and updates shared state under lock
- `_chunks_for_file(filepath)` — returns all chunk IDs for a file (used by `reindex_file`)

**Interactive commands**: `/edit <prompt>`, `/clear`, `/cost`, `/help`, `exit`/`quit`, `Ctrl+C`.

`/edit` mode instructs Claude to return complete files in ` ```lang:path ``` ` blocks; the script diffs, confirms, and writes them. One-shot and piped modes auto-apply edits without confirmation.

## Key Config (top of script)

| Variable | Purpose |
|---|---|
| `TARGET_RETRIEVED_TOKENS` | Token budget for retrieved context per query (default 6 000) |
| `MIN_SCORE` | Minimum cosine similarity to include a chunk (default 0.45) |
| `MAX_HISTORY_TURNS` | Sliding-window size for conversation history (default 6) |
| `MODEL` | Claude model ID |
| `PRICE_*` | Token pricing constants used for cost/savings display |
| `EMBED_MODEL`, `TOP_K` | Set at runtime by `auto_tune()` — do not set manually |
