# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`claude_light` is an interactive CLI chat tool for querying and editing a multi-language codebase using Claude. It uses a hybrid RAG + prompt caching strategy: project docs and directory structure are cached as a system prompt, while source files are split into method/function-level chunks via tree-sitter, embedded, and retrieved per-query.

## Repository Layout

| Path | Purpose |
|---|---|
| `claude_light/` | Main package — all application code |
| `claude_light.py` | Thin shim that delegates to `claude_light/__main__.py` |
| `installers/` | OS-specific installers (`install.sh`, `install_macos.sh`, `install.ps1`) |
| `tests/unit/` | Unit tests for each module |
| `tests/integration/` | Integration tests (git, benchmarks) |
| `tests/utilities/` | Utility/mock tests |
| `tests/linting/` | Syntax and regression linting checks |
| `tests/benchmarks/` | Benchmark scripts (not run by default pytest) |
| `tests/fixtures/` | Static fixture data (excluded from pytest discovery) |
| `docs/architecture.md` | In-depth implementation walkthrough |
| `docs/BENCHMARKS.md` | Benchmark methodology and interpretation |
| `scripts/` | SBOM generation scripts |

## Running

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 claude_light.py

# One-shot (no interactive loop)
python3 claude_light.py "What does OrderService do?"

# Pipe
echo "List all REST endpoints" | python3 claude_light.py

# Simulation / test mode (no API key or costs required)
python3 claude_light.py --test-mode small        # 5 files, 10 methods each
python3 claude_light.py --test-mode extra-large  # 1000 files, 20 methods each
```

Run from the root of a project. Exits with an error if no authentication is found (unless using `--test-mode`).

**Authentication resolution order:**
1. **Environment Variable**: `ANTHROPIC_API_KEY=sk-ant-...`
2. **Local Dotfiles**: `~/.anthropic` or project-local `./.env`
3. **Automation Token**: `~/.claude_light_automation_token` (recommended for Windows)
4. **Claude CLI OAuth**: `~/.claude/.credentials.json` (detected via standard `claude auth login`)

The tool supports both **API Keys** (usage-based billing) and **Claude Pro Subscriptions** (flat-rate). It determines the mode dynamically based on the key found.

## Installation

```bash
# Linux
bash installers/install.sh

# macOS (creates .venv + run.sh wrapper, handles Homebrew + zsh)
bash installers/install_macos.sh

# Windows PowerShell 5.1+
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\installers\install.ps1
```

## Dependencies

**Required:**
```bash
pip install "sentence-transformers>=5.3.0" "numpy>=2.4.3" "watchdog>=6.0.0" "anthropic>=0.86.0"
# Note: sentence-transformers pulls in PyTorch (~1.5 GB on first install)
```

**Optional (strongly recommended):**
```bash
pip install \
    "tree-sitter>=0.25.2" \
    "tree-sitter-java>=0.23.5" \
    "tree-sitter-python>=0.25.0" \
    "tree-sitter-go>=0.25.0" \
    "tree-sitter-rust>=0.24.1" \
    "tree-sitter-javascript>=0.25.0" \
    "tree-sitter-typescript>=0.23.2" \
    "rich>=14.3.3" \
    "prompt_toolkit>=3.0.52" \
    "einops>=0.8.2"
```

Without tree-sitter, chunking falls back to whole-file mode. Without rich/prompt_toolkit, the tool degrades gracefully to plain text output and basic `input()`.

## Tests

```bash
# Run all tests (unit + integration + utilities + linting)
pytest

# Single test file
pytest tests/unit/test_retrieval.py

# Single test function
pytest tests/unit/test_retrieval.py::test_retrieve_top_k

# Benchmarks (excluded from default run, require sentence-transformers)
python tests/benchmarks/benchmark.py
python tests/benchmarks/benchmark_retrieval.py --split dev
python tests/benchmarks/benchmark_cost.py --dry-run
```

`conftest.py` sets `ANTHROPIC_API_KEY=sk-ant-test-mock-key` so unit tests never make real API calls. `pytest.ini` excludes `tests/fixtures/` and `tests/benchmarks/` from collection.

## Package Architecture

The `claude_light/` package is organized by responsibility:

| Module | Role |
|---|---|
| `config.py` | All constants, API key resolution, tree-sitter language loading, pricing |
| `state.py` | Shared mutable state (`chunk_store`, `conversation_history`, `embedder`, etc.) protected by `threading.Lock` |
| `main.py` | Entry point; chat loop, signal handlers, heartbeat thread, watchdog lifecycle |
| `llm.py` | API calls, response streaming, query routing, skeleton/history management |
| `retrieval.py` | Cosine similarity search (`embs @ q_emb`), dedup of multi-chunk files |
| `indexer.py` | File watcher (`SourceHandler`), incremental re-embed, disk cache (`.claude_light_cache/`) |
| `skeleton.py` | Compressed directory tree + `.md` file assembly for the cached system prompt |
| `parsing.py` | Tree-sitter AST traversal, method-level chunk extraction for all supported languages |
| `editor.py` | Parses `SEARCH/REPLACE` blocks from Claude responses, applies and diffs edits |
| `linter.py` | Pre-apply syntax validation (Python `ast`, tree-sitter for other languages) |
| `git_manager.py` | Auto-commit of AI edits; `/undo` reverts to the previous commit |
| `executor.py` | Loads `SentenceTransformer` model, runs `auto_tune()`, executes `/run <cmd>` |
| `streaming.py` | Real-time token streaming, thinking-block handling, usage accumulation |
| `retry.py` | Exponential backoff (2s→4s→8s, max 3 attempts) for transient API errors |
| `session_manager.py` | Persists conversation history to `.claude_light_cache/session.json` |
| `ui.py` | ANSI colour, rich markdown rendering, diff display, cost formatting |
| `testing.py` | Test-mode helpers (synthetic file/chunk generation, mock API key) |
| `__init__.py` | Re-exports all public symbols; wraps callables to sync shared state across modules |

**`__init__.py` design note:** Because modules import each other's globals at load time, `__init__.py` uses a `_wrap()` decorator that calls `_sync_bindings()` before and `_refresh_exports()` after every public function call. This keeps all modules' references to shared state (e.g. `state.embedder`, `state.chunk_store`) consistent without circular imports.

## Architecture

**Hybrid RAG + prompt caching. Two tiers:**

| Tier | Content | Strategy |
|---|---|---|
| Skeleton (cached system prompt) | Directory tree + all `.md` files | Ephemeral cache — paid once, re-read cheaply |
| Source files | Method/function-level chunks via tree-sitter | RAG — top-K chunks retrieved per query |

**Three concurrent threads:**

1. **Main thread** — input loop; retrieves relevant chunks per query via `retrieve()`, injects them as a second cached system block, stores only the clean query in `conversation_history`
2. **Watchdog** (`SourceHandler`) — source file change → `reindex_file()` (re-chunks + re-embeds that file, cancels any pending timer first via 1.5 s debounce); `.md` change → `refresh_md_file()`
3. **Heartbeat daemon** — checks every 30 s; if idle > 4 min (`CACHE_TTL_SECS`), calls `warm_cache()` to keep the ephemeral cache alive

**Multi-language support** (`_WANTED_LANGS` in `config.py`): tree-sitter grammars extract method/function nodes from `.java`, `.py`, `.js`, `.go`, `.rs`, `.ts`, `.tsx`. Falls back to whole-file chunking if tree-sitter is unavailable or a grammar is missing. Python decorators are included in their function chunk.

**Method-level chunking** (`parsing.py`): Each chunk includes a file path comment and the class preamble (package + imports + class header) so it is self-contained. Overloaded method names get numeric suffixes. Chunk IDs use `filepath::methodName` (`::` is not valid in source identifiers). Falls back to whole-file if no methods are found.

**Disk cache** (`.claude_light_cache/`): Embeddings are persisted to `index.pkl` between runs; a `manifest.json` tracks per-file MD5 hashes and the embedding model used. Only changed or new files are re-embedded on restart. Command history (prompt_toolkit) and session state are also stored here.

**Auto-tuning** (`auto_tune` in `executor.py`): called twice in `index_files()` — first to select and load the embedding model (based on file count), then again after chunking to set `TOP_K` based on actual chunk sizes, targeting `TARGET_RETRIEVED_TOKENS` of context per query.

| File count | Embedding model |
|---|---|
| < 50 | `all-MiniLM-L6-v2` (22 MB) |
| 50–199 | `all-mpnet-base-v2` (420 MB) |
| 200+ | `nomic-ai/nomic-embed-text-v1.5` (requires `einops`) |

**Scoring**: `retrieve()` uses a single matrix multiply (`embs @ q_emb`) over all chunk embeddings — no per-chunk loop. Two filters: `MIN_SCORE` (0.45, absolute floor) and `RELATIVE_SCORE_FLOOR` (0.60, fraction of the top chunk's score). Per-effort token budgets in `_RETRIEVAL_BUDGET` scale the context window from 1 500 (low) to 9 000 (max) tokens.

**Token optimisations:**
- Skeleton cached; retrieved chunks also cached (second `cache_control` block) — repeated queries over the same code pay $0.30/M instead of $3.00/M
- Conversation history capped at `MAX_HISTORY_TURNS` (6) via sliding window; old turns are summarised in batches of `SUMMARIZE_BATCH` (3) using the cheap Haiku model
- `build_skeleton()` renders the directory tree with two compressions: (1) single-child directory chains collapsed (`main/java/com/example/`), (2) sibling files sharing an extension grouped with brace notation (`{OrderService,UserService}.java`). Typical savings: 30–50% of skeleton tokens.

**Key functions by module:**
- `llm.py`: `route_query()` — weighted intent classifier (Arch/Logic/Infra) selects model + effort; `_build_system_blocks()` — constructs the system prompt list; `_update_skeleton()` — rebuilds skeleton under lock; `_extract_text()` — joins text blocks, skips thinking blocks
- `skeleton.py`: `_build_compressed_tree()` / `_render_compressed_node()` — tree with chain collapse + brace grouping
- `retrieval.py`: `_dedup_retrieved_context()` — emits per-file preamble once when multiple chunks from the same file are retrieved
- `indexer.py`: `_chunks_for_file()` — all chunk IDs for a file path

**Interactive commands**: `/clear`, `/compact`, `/cost`, `/help`, `/run <cmd>`, `/undo`, `exit`/`quit`, `Ctrl+C`.

**Auto-commit safety net**: When Claude makes file changes, the tool automatically commits them to git (`Claude: [summary]`). `/undo` reverts the last commit. Gracefully skipped if not in a git repo.

For a deeper dive, see [docs/architecture.md](docs/architecture.md).

## Reliability Features

**Auto-Retry** (`retry.py`): exponential backoff 2s→4s→8s (capped at 60s), max 3 attempts. Non-retriable errors (401/403/404/400) fail immediately.

**Thread-Safe State** (`state.py`): all shared state protected by `threading.Lock`. Global stats (`~/.claude_light_stats.json`) loaded on startup, saved on every usage update.

**Streaming** (`streaming.py`): real-time token streaming; thinking blocks shown as a progress indicator; accurate token tracking from streaming events.

**Watchdog Lifecycle** (`main.py`): observer stopped and joined on exit (5 s timeout); SIGINT/SIGTERM handlers set `stop_event`; startup failures handled gracefully.

## Key Config (`claude_light/config.py`)

| Variable | Purpose |
|---|---|
| `TARGET_RETRIEVED_TOKENS` | Token budget for retrieved context per query (default 6 000) |
| `MIN_SCORE` | Absolute cosine similarity floor to include a chunk (default 0.45) |
| `RELATIVE_SCORE_FLOOR` | Drop chunks below this fraction of the top chunk's score (default 0.60) |
| `_RETRIEVAL_BUDGET` | Per-effort token budgets: low=1500, medium=3000, high=6000, max=9000 |
| `MAX_HISTORY_TURNS` | Compress+cap when stored turns exceed this (default 6) |
| `SUMMARIZE_BATCH` | Old turns collapsed per summary call (default 3) |
| `GLOBAL_STATS_FILE` | Path to global savings persistence (`~/.claude_light_stats.json`) |
| `HEARTBEAT_SECS` | How often the heartbeat thread wakes (default 30) |
| `CACHE_TTL_SECS` | Idle seconds before heartbeat warms the cache (default 240) |
| `MODEL_HAIKU` / `MODEL_SONNET` / `MODEL_OPUS` | Model ID constants used by the router |
| `SUMMARY_MODEL` | Model used for history compression (set to `MODEL_HAIKU`) |
| `PRICE_INPUT` / `PRICE_WRITE` / `PRICE_READ` / `PRICE_OUTPUT` | Token pricing ($3.00 / $3.75 / $0.30 / $15.00 per M) |
| `SKIP_DIRS` | Directory names excluded from indexing |
| `EMBED_MODEL`, `TOP_K` | Set at runtime by `auto_tune()` — do not set manually |

## SBOM

```bash
bash scripts/generate_sbom.sh   # Linux / macOS
.\scripts\generate_sbom.ps1    # Windows
```

Also generated automatically as a GitHub Actions artifact on every push to `main`.

## Benchmarks

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full methodology.

| Script | What it tests | API key needed |
|---|---|---|
| `tests/benchmarks/benchmark.py` | Analytical token savings (synthetic) | No |
| `tests/benchmarks/benchmark_retrieval.py` | RAG quality — Hit@K, MRR on SWE-bench Lite | No |
| `tests/benchmarks/benchmark_cost.py` | Real-world cost vs naive baseline | Yes (~$0.10–0.20) |
