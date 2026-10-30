# Data Model: Gemini and Codex Providers

Only new or changed state. The Provider interface itself is in `contracts/provider.md`.

## `state.provider` (new, `claude_light/state.py`)

- Type: `Provider | None`, default `None`, which `providers.active()` reads as Claude (FR-002).
- Written once, by `providers.select()` from `main.main()`, under `state.lock`. Never changed
  while a session runs (spec Assumptions: one provider per launch).
- Not re-exported by `claude_light/__init__.py`, so `_sync_bindings()`/`_wrap()` need no entry,
  as with `state.session_model`.
- Tests set it with `monkeypatch.setattr(state, "provider", ...)`, which restores it after the test.

## `client` (moved, `claude_light/llm.py` to `claude_light/providers/claude.py`)

- The Anthropic client (`MockClient` under the test key, `_LazyClient` otherwise) moves unchanged
  with Claude's transport (D2).
- `claude_light/__init__.py` imports `claude_light.providers.claude` inside `try/except
  ImportError`, so the package loads without it (US4). When present, `_MODULE_ATTR_MAP["client"]`
  becomes `[(claude, "client")]`, `"print_stats"` gains `(claude, "print_stats")`, and
  `_refresh_exports` takes `client` from it, so `_wrap()` syncs both as before (constitution IV).
  When absent, the package exports no `client`.

## Session file (changed, `claude_light/session.py`)

| Provider | File in `config.SESSION_DIR` | Content |
|---|---|---|
| claude | `<digest>.json` (unchanged) | unchanged |
| gemini, codex | `<digest>.<provider>.json` | same keys as Claude's |

`<digest>` is today's `session_file()` hash of the project path. Rules (FR-016, D4):

- `save()` and `clear()` touch only the active provider's file.
- `restore()` with the active provider's file present: as today.
- `restore()` without it, but with another provider's file for the project: return the mismatch
  message of `contracts/provider.md` (with two such files, the newer by mtime names `<saved>`),
  change no state, and open no file for writing.
- Neither present: today's "No previous conversation" line.
