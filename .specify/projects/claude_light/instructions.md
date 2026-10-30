# claude_light: project instructions

These rules come from the repo's `CLAUDE.md`, `TESTING.md` and `.github/workflows/test.yml`, and from how `claude_light/llm.py` and `claude_light/config.py` work on 2026-10-30. Read `CLAUDE.md` and `HARNESS.md` before you change code.

## Providers (Claude, Gemini, Codex)

- Each provider lives in its own module, under one package (for example `claude_light/providers/claude.py`, `gemini.py`, `codex.py`). Nothing outside a provider's module may import that provider's SDK or name its CLI binary.
- Provider-neutral code (routing, retrieval, indexing, compression, the tool loop, the auto-commit path) talks to providers only through one shared interface that the plan defines. It must not branch on a provider's name.
- Moving Claude behind the interface changes no Claude behaviour. The existing Claude tests pass unchanged, and so does the token benchmark in CI (`tests/benchmarks/benchmark.py` plus `tests/linting/check_regression.py`).
- Claude has two transports today. API_KEY mode uses the `anthropic` SDK, and OAUTH mode runs the `claude` CLI as a subprocess (`providers/claude.py: _make_cli_subprocess_call`). Gemini and Codex run their CLIs in an empty temporary directory, because the gemini CLI executed a repo's `.gemini/settings.json` hooks when started in it, and agy needs a per-run `--settings` deny on command, mcp and file or it runs the user's globally allowed MCP shell. Gemini and Codex start as CLI subprocess transports: Google's Antigravity CLI `agy` for Gemini (1.4.11 installed here; the `gemini` CLI is refused for individual accounts, decision D10) and the `codex` CLI (codex-cli 0.162.1 installed here). Before the plan relies on a flag, check it with the installed CLI's `--help`, not from memory.
- The user picks the provider explicitly (a CLI flag or an env var, as the plan decides). Without a choice, behaviour stays exactly as it is now: Claude, with the current auth order.
- Model ids, prices and effort levels belong to their provider. `config.py` keeps the Claude constants. Gemini and Codex get their own constants, inside their own modules or a per-provider config section.
- Anthropic-only features (prompt caching, `output_config.effort`, the Opus/Fable top-tier prompt, `/budget` USD pricing) must degrade explicitly for a provider without them. The user sees that a feature is unavailable; it is never silently ignored.
- The CLI shell guard applies to every agent CLI. A provider whose CLI can run shell commands gets the same `--allow-command` prefix rule and metacharacter rejection that `cli_shell_guard.py` enforces for Claude. If a CLI cannot be restricted that way, the plan says so and that provider runs with shell disabled.
- Edits made by any provider's agent go through the same auto-commit path (`Claude: [summary]` or a provider-neutral prefix the plan decides), so `/undo` keeps working.

## Invariants (from CLAUDE.md, apply unchanged)

- Background threads never print; they call `ui.notify()`.
- Shared state goes through `state.py` under `state.lock`; a new cross-module global goes through `__init__._wrap()`.
- `route_query()` stays pure; session effects live in `_finalize_route()`.
- Never load `index.pkl`; the disk cache loads with `allow_pickle=False`.

## Tests

- Tests never call a real API or a real CLI and never write the real stats or session files. Mock the subprocess and the SDK. Keep the `tests/conftest.py` settings.
- New tests go in `tests/unit/` or `tests/integration/` in the existing pytest style, so CI runs them.
- The end-to-end check against the real `agy` (Gemini) and `codex` CLIs is a manual or opt-in step, not a CI test. Its result is reported with the commands run and their output. If a CLI is not logged in, report it as not run, not as passed.

## Docs

- Update `README.md`, `CLAUDE.md` (Commands and Authentication), `docs/CONFIG.md`, `docs/architecture/ARCHITECTURE.md` and `HARNESS.md` in the same change.
