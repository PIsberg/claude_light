# CLAUDE.md

Guidance for Claude Code in this repository. This file is kept short on purpose: it routes to the documents that hold the detail. Read the document for your task before changing code.

## What this project is

`claude_light` is an interactive CLI for querying and editing a multi-language codebase with Claude. It caches a project skeleton in the prompt cache and retrieves only the method-level code chunks each question needs (tree-sitter chunks, embedded, top-K per query).

## Where to read

| Task | Read |
|---|---|
| Repo map, what each file does, agent harness | [HARNESS.md](HARNESS.md) |
| How a part works (routing, caching, retrieval, indexing, compression) | [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) |
| Config constants and env vars | [docs/CONFIG.md](docs/CONFIG.md) |
| Running, adding and fixing tests, test layout | [TESTING.md](TESTING.md) and [docs/test_strategy.md](docs/test_strategy.md) |
| What CI runs on a push or PR | [docs/WORKFLOW.md](docs/WORKFLOW.md) |
| Benchmarks and how to measure a token change | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| Token-savings levers and their measured results | [TOKEN-SAVINGS.md](TOKEN-SAVINGS.md) |
| Compression design rationale | [docs/llmlingua_plan.md](docs/llmlingua_plan.md) |
| Index of all docs | [docs/DOCS.md](docs/DOCS.md) |

## Commands

```bash
# Interactive
ANTHROPIC_API_KEY=sk-ant-... python3 claude_light.py

# One-shot, or piped
python3 claude_light.py "What does OrderService do?"
echo "List all REST endpoints" | python3 claude_light.py

# Resume this project's previous conversation
python3 claude_light.py --continue

# Pick the agent: claude (default), gemini or codex
python3 claude_light.py --provider codex "What does OrderService do?"

# Offline, no API key: the OAuth agent may run these commands
python3 claude_light.py --allow-command "python -m pytest"
python3 claude_light.py --test-mode small        # 5 files, 10 methods each

# Tests (see TESTING.md for the full picture)
pytest
```

### Tools, `/budget` and affected tests

- **`fetch_chunk(id)`**: returns up to 5 method bodies by `path::symbol` or bare name, with callers and callees. It needs the index to be ready.
- **`shell`**: runs one command from the project root, only if it starts with an `--allow-command` prefix. Commands containing `; & | < > `` ` `` $` or a newline are rejected, and PowerShell commands also `( ) { } @` (an argument in those is evaluated). The OAuth CLI agent gets the same rule from the `cli_shell_guard.py` `PreToolUse` hook, because the CLI's own allow rule checks only the first statement. It is always in `TOOLS`, so the cached prompt prefix stays stable.
- **`/budget`**: `/budget 2.50` sets a spend cap, `/budget off` removes it, `/budget` shows cap and spend. At the cap, routing drops to Haiku at low effort. While a budget is set, the route line shows a per-turn estimate.
- **Affected tests**: after an auto-commit, if a pytest command is allowed, the tests that mention the edited module run. On failure it offers to revert the commit and never reverts on its own. It does not run after OAuth-agent edits.

Run from the root of the project you want to query. Exits with an error if no authentication is found, unless `--test-mode` is used; `--help` works without one. The check is `providers.active().require_auth()`, called by `main()` after the arguments are parsed, not at import.

Authentication, in order: `ANTHROPIC_API_KEY`; `~/.anthropic` or `./.env`; `~/.claude_light_automation_token` (recommended on Windows); `~/.claude/.credentials.json` from `claude auth login`. An API key bills per token; a Pro subscription is flat-rate. The mode follows the credential found. That order is the `claude` provider's. `--provider gemini` needs the Antigravity CLI `agy` on PATH (https://antigravity.google; it keeps no credential file, so a missing sign-in shows on the first question, and `search_web` still runs); `--provider codex` needs `codex login`. Both run in an empty temporary directory and have no caching, effort, top tier, `/budget` or shell; each says so when you use one.

Install with `installers/install.sh`, `installers/install_macos.sh` or `installers/install.ps1`. Dependencies come from `requirements.txt`, which the installers and the CI test and fuzz jobs all install. Python 3.12 or newer.

## Invariants: check these before editing

- **Neutral code reaches a provider only through `claude_light/providers/`.** No import of a provider's SDK and no provider binary name outside `providers/<name>.py` (`tests/unit/test_provider_isolation.py` enforces it). Use `providers.active()` and `providers.DEFAULT`.
- **Background threads never print.** The watcher and the heartbeat call `ui.notify()`. A print lands inside a streamed reply or over the prompt being typed.
- **Shared state goes through `state.py` under `state.lock`.** Modules import each other's globals at load time, so `__init__.py` wraps public functions in `_wrap()` to sync bindings. Do not add a new cross-module global without going through it.
- **`EMBED_MODEL` and `TOP_K` are set at runtime by `auto_tune()`.** Do not set them by hand.
- **`state.turn_count` counts turns.** Do not derive it from `conversation_history`, which compression shrinks.
- **Chunk ids are `filepath::methodName`.** Watchdog paths go through `indexer._project_path()`, or a re-index duplicates the file's chunks.
- **Never load `index.pkl`.** The disk cache loads with `allow_pickle=False` because a cloned repo can ship a hostile pickle.
- **Tests never call the real API or write the real stats or session files.** `tests/conftest.py` sets a mock key, `CLAUDE_LIGHT_HYBRID=0`, a temp `CLAUDE_LIGHT_STATS_FILE` and a temp `CLAUDE_LIGHT_SESSION_DIR`. Keep those settings when you add tests.
- **Auto-commit is the undo path.** AI edits commit as `Claude: [summary]`. `/undo` refuses unless the last commit is one of those and no file in it has changed since.
- **The top-level effort never changes per request.** Requests send `BASE_EFFORT`; `_apply_effort()` puts the turn's effort in an effort-only system message before the question. A per-request `output_config.effort` drops the cached history.
- **`route_query()` is a pure router; `_finalize_route()` applies the session.** The budget cap, the top-tier prompt (Opus 5.5, Fable 5.1 or Sonnet 5.5) and `ROUTE_ONCE` live in `_finalize_route()`, so benchmarks can call `route_query()` without pinning or prompting.
- **The first turn's model is kept for the session** (`config.ROUTE_ONCE`, on by default) in `state.session_model`; `/clear` resets it, and only a top-tier route can change it. Tests rely on the autouse fixture in `tests/conftest.py` that clears it, or one test's route pins the next test's model.
- **A token change needs a measurement.** Run the relevant benchmark before and after, and keep only changes that show a gain. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
