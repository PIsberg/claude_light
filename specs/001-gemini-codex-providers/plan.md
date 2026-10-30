# Implementation Plan: Gemini and Codex Providers

**Branch**: `001-gemini-codex-providers` | **Date**: 2026-10-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-gemini-codex-providers/spec.md`

## Summary

A `claude_light/providers/` package puts Claude, Gemini and Codex behind one contract
(`contracts/provider.md`). Provider-neutral code (`llm.chat`, `llm._one_shot_turn`,
`llm._summarize_turns`, `llm.warm_cache`, routing, `main`) calls `providers.active()` and checks
capabilities, never a name. Claude's client, its two transports and its cache warm move unchanged
from `llm.py` into `providers/claude.py` (D2); the existing tests change only in their patch
targets. Gemini and Codex run their CLIs headless as answer engines with tools and shell off (D1):
they get the same skeleton and retrieved chunks, and their edits come back as SEARCH/REPLACE
blocks through the existing `editor.apply_edits` and `git_manager.auto_commit`, so `/undo` works
unchanged.

## Technical Context

**Language/Version**: Python 3.12+

**Primary Dependencies**: none new. Gemini and Codex are driven as subprocesses (spec
Assumptions); `anthropic` stays Claude's, imported lazily.

**Storage**: per-project session JSON in `config.SESSION_DIR`, one file per provider (`data-model.md`)

**Testing**: pytest, `tests/unit/`, subprocess and `shutil.which` mocked, under `tests/conftest.py`

**Target Platform**: Windows, macOS, Linux terminals (npm `.cmd` shims on Windows: `research.md` R4)

**Project Type**: CLI

**Performance Goals**: SC-001, an answer within 60 s on a small project

**Constraints**: prompt on stdin, never on the command line (Windows 8,191-character limit);
Claude's requests byte-identical (FR-006, constitution III)

**Scale/Scope**: 3 providers; 6 new files under `claude_light/providers/`; 6 changed modules
(`llm.py`, `main.py`, `ui.py`, `session.py`, `state.py`, `__init__.py`); about 700 of
`llm.py`'s 1,719 lines move to `providers/claude.py`

## Constitution Check

Constitution 1.0.0. Checked before Phase 0 and again after the design: no violation.

| Rule | Tasks | How |
|---|---|---|
| I. Hermetic tests | T002, T005, T007, T009, T012, T015, T017, T019, T021, T023, T027 | `subprocess.Popen` and `shutil.which` patched, credential files under `tmp_path`, conftest settings kept |
| II. Tested behaviour | T003, T006, T008, T010, T013, T016, T018, T020, T022, T024, T028, each after its tests | tests in `tests/unit/`, asserting output, files and commits |
| III. Measured token changes | T003, T018, T026 | Claude's request is unchanged by design (T003 moves `transcript`, T018 the transport, byte for byte); T026 runs both CI gates against the committed baselines |
| IV. Background threads never print | T018, T024 | `llm.warm_cache` calls the provider's `warm`, a no-op except Claude's (T018); `heartbeat` returns before any print without `caching` (T024) |
| IV. Shared state via `state.py` | T001, T003, T018 | `state.provider`, written under `state.lock`, not re-exported; `client` moves and its `_wrap()` binding follows it (`data-model.md`) |
| IV. `route_query()` pure | T022 | provider effects only in `_finalize_route()` |
| IV. Top-level effort never changes | T022 | `BASE_EFFORT` untouched; only the effort-only message is skipped without `effort` |
| V. Contained shell access | T013, T016, T024, T028 | both CLIs run with shell off (`research.md` R1-R3); `--allow-command` prints the notice |
| VI. Docs move with code | T025, T028 | the five docs FR-014 names |
| Not applicable | none | IV `index.pkl`: cache untouched. IV `turn_count`: not touched. Development Workflow: delivery, the main session's |

## Design

Where neutral code changes, each driven by a capability of `contracts/provider.md`:

| Code | Change | Task |
|---|---|---|
| `llm.chat`, `llm._one_shot_turn` | `providers.active().complete(create_kwargs)` replaces `_make_streaming_api_call(**create_kwargs)` | T003 |
| `llm._summarize_turns` | builds `instructions`, returns `providers.active().summarize(instructions)`; today's two Claude branches become `ClaudeProvider.summarize` | T003 |
| `llm._make_streaming_api_call` OAUTH branch | flattening moved to `providers/base.py: transcript`, called from here | T003 |
| `main.main` | `--provider`, `providers.select()`, `provider.require_auth()` replaces `llm.require_auth()` | T003 |
| `llm.client`, `_make_cli_subprocess_call`, `_make_streaming_api_call`, `_api_tool_loop`, their helpers, `ClaudeNotLoggedIn` | moved unchanged to `providers/claude.py` (D2) | T018 |
| `llm.warm_cache` | becomes `providers.active().warm(quiet)`; its body becomes `ClaudeProvider.warm` | T018 |
| `except ClaudeNotLoggedIn` in `llm.chat`, `main.start_chat` | catch `providers/base.py: NotLoggedIn` | T018 |
| `claude_light/__init__.py` binding map | `client` and `print_stats` entries follow the move (`data-model.md`) | T018 |
| `llm._finalize_route` | without `top_tier`: no prompt, `state.session_model` untouched, model = `model_label()`, contract route line | T022 |
| `llm._apply_effort` | effort-only message only with `effort` | T022 |
| `main._budget_command` | without `usd_budget`: message, `state.session_budget` unchanged | T022 |
| `llm.chat`, `_one_shot_turn`, `compact_history`, `_maybe_compress_history` | `usage is None`: no cost, no accumulation, stats line replaced by the message | T010 |
| `main._status_text`, `ui.print_session_summary`, `ui.print_session_footer` | without `usage`: the message instead of figures | T010 |
| `main.heartbeat`, `main.start_chat` | without `caching`: no heartbeat thread | T024 |
| `AUTH_MODE == "OAUTH"` in `chat`, `_one_shot_turn`, `one_shot`, `main._index_ready_for_query` | becomes `AUTH_MODE == "OAUTH" and provider.has(AGENT)`, so the index is awaited and no agent-edit snapshot is taken for Gemini or Codex | T024 |
| `main.start_chat` header | `provider.header() or (MODEL, access_label(API_KEY_SOURCE))` | T024 |
| `main.main` | without `shell` and with `config.ALLOWED_COMMANDS`: the shell notice | T024 |

The `AUTH_MODE` checks stay, ANDed with a capability, because existing tests patch
`llm.AUTH_MODE` and `main.AUTH_MODE` (`research.md` R5). For Claude every capability is true, so
each expression reduces to today's.

Edge cases the spec lists and no task adds code for: an unknown name is argparse's `choices`
(T003); a failed or unparseable turn already leaves history, `turn_count` and git untouched,
since `llm.chat` appends and commits only after a reply (T002 pins it); "context larger than a
provider's limit" cannot arise, because the largest retrieval budget is 9,000 tokens
(`config._RETRIEVAL_BUDGET`), applied before any provider sees the request (FR-003).

## Project Structure

### Documentation (this feature)

```text
specs/001-gemini-codex-providers/
├── plan.md, research.md, data-model.md, quickstart.md, tasks.md
├── contracts/provider.md
└── verification.md      # written by T026
```

### Source Code (repository root)

```text
claude_light/
├── providers/
│   ├── __init__.py      # NAMES, get, select, active
│   ├── base.py          # Provider, capabilities, ProviderError, NotLoggedIn, transcript
│   ├── runner.py        # run_cli, kill_process_tree (moved from llm)
│   ├── claude.py        # ClaudeProvider, client, the API and CLI transports (moved from llm), offer_login
│   ├── gemini.py        # GeminiProvider
│   └── codex.py         # CodexProvider
├── llm.py  main.py  ui.py  session.py  state.py  __init__.py   # changed as in Design
tests/unit/
├── test_providers.py  test_provider_runner.py  test_session_provider.py
├── test_provider_gemini.py  test_provider_codex.py  test_provider_isolation.py
└── test_provider_route.py  test_provider_usage.py  test_provider_startup.py
```

**Structure Decision**: one package, one module per provider (FR-004); the subprocess runner
gets its own module because both CLI providers share it and Claude's CLI transport reuses its
process-tree kill.

## Complexity Tracking

None.

## Decisions

### D1. How do Gemini and Codex run?

**Answer (2026-10-30)**: A.

- **A (recommended): as answer engines, tools and shell off.** Argv in `contracts/provider.md`.
  They answer from the skeleton and retrieved chunks only, so a question needing a file outside
  retrieval gets a weaker answer than Claude's OAuth agent, which can open files. Edits arrive as
  SEARCH/REPLACE blocks (`config.SYSTEM_PROMPT` already asks for them), so FR-009 needs no new
  code. FR-010 is met by its second branch: shell disabled, user told (T024); US2 scenario 3 is
  moot and scenario 4 applies. Codex's restriction was probed (`research.md` R2), Gemini's agy
  settings override too, except `search_web` (R3, D10). Tasks as written.
- **B: as agents that edit files.** Gemini through agy (D10) with a `--settings` override that
  allows edits (`contracts/provider.md`),
  Codex `--sandbox workspace-write` plus execpolicy rules. Each needs its own shell guard (Codex
  hooks need `--dangerously-bypass-hook-trust`), neither mechanism verified here; both get the
  `agent` capability and go through `llm._commit_agent_edits`. Adds 2 slices (one per CLI guard)
  and changes T013, T016, T024; if a guard proves unenforceable, FR-010 disables shell anyway.

### D2. Where does Claude's transport code live?

**Answer (2026-10-30)**: B. Claude's transport moves fully to `providers/claude.py`, existing
tests changed only in their patch targets. Added as slice S12 (T017, T018) in Round 2, before S6,
whose isolation test passes only once the transport has left `llm.py`; S6 (T019, T020) keeps only
the login flow. The `Spec:` lines below go to spec.md through the product-owner.

- **A (recommended): it stays in `llm.py`.** `client`, `_make_cli_subprocess_call`,
  `_make_streaming_api_call`, `_api_tool_loop` and `warm_cache` stay; `providers/claude.py` holds
  the SDK import, the binary name, the login flow, the messages and the adapter. Existing tests
  pass unchanged (FR-006; counts in `research.md` R5). Consequence for US4: changing Claude's
  transport still edits `llm.py`; deleting `providers/claude.py` disables Claude only (T012 and T015 pin
  that `llm` still imports). SC-003 holds by the isolation rule.
- **B: move it all to `providers/claude.py`.** About 1,000 lines move, and every patch target
  above changes in `tests/unit/test_llm.py`, `test_effort_cache.py`, `test_claude_light.py`,
  `tests/integration/test_api_tool_loop.py` and others. Adds one slice after T018.
  Spec: FR-006: "Moving Claude behind the shared contract MUST NOT change Claude behaviour: the
  existing Claude tests pass with only their patch targets moved from `claude_light.llm` to
  `claude_light.providers.claude`, and the CI token and retrieval regression gates pass."
  Spec: SC-002: "100% of the existing tests pass, changed only in their patch targets, and the CI
  token and retrieval regression gates pass with no change to Claude's measured token counts."

### D3. What do the route line and header show?

**Answer (2026-10-30)**: A.

- **A (recommended): Claude's unchanged; Gemini and Codex name the provider and "CLI default
  model".** No `-m` is passed (spec Assumptions: default model), and `--ignore-user-config` means
  Codex ignores the user's configured model too (`research.md` R2). Tasks as written.
  Spec: FR-012: "With Gemini or Codex chosen, the route line MUST show the provider and its model,
  or "CLI default model" when the provider's tool picks it. With Claude, chosen or by default, the
  route line is unchanged."
- **B: prefix every route line with the provider, Claude's too.** Changes Claude's output, so
  tests that assert it (`tests/unit/test_top_tier.py`, `test_budget.py`) change, against FR-002
  and FR-006. Changes T022. Spec: FR-002 gains "except that the route line names the provider".
- **C: pin a model id per provider module, passed with `-m`.** The line shows a real id; you name
  the ids; an id the account cannot use fails every turn, and ids go stale. Changes T013, T016,
  T022. Spec: Assumptions: "each provider's default model" becomes "the model id its module pins".

### D4. How are sessions of different providers stored? (schema change)

**Answer (2026-10-30)**: A.

- **A (recommended): one file per provider** (`data-model.md`). Claude's file name and content
  are unchanged (FR-002); a Gemini session never overwrites Claude's, so the edge case "resuming
  it with its own provider still works" holds after later saves. With both files present,
  `--continue` resumes the chosen provider's without a warning. Tasks as written.
- **B: one file with a `provider` field.** The first save after the warning overwrites the other
  provider's conversation. Changes T007, T008.
  Spec: Edge case `--continue`: "... The saved session file is not modified or deleted by the
  warning; the first conversation saved with the chosen provider replaces it."

### D5. Do Gemini and Codex show token counts?

**Answer (2026-10-30)**: A.

- **A (recommended): no.** They report no usage; every figure shows the "unavailable" message
  (FR-008, edge case "shown as unavailable, not as zero"). Tasks as written.
- **B: yes, tokens without cost.** Codex's `turn.completed.usage` is verified (R1); Gemini's agy
  `result.usage` carries input, output, thinking, cache-read and total tokens (R3). Both gain the `usage` capability
  minus pricing; adds one slice and changes T013, T016, T010.

### D6. What prefix do Gemini and Codex commits get?

**Answer (2026-10-30)**: A.

- **A (recommended): `Claude: `, as today** (`git_manager.COMMIT_PREFIX`). `/undo` unchanged; the
  prefix names the tool, claude_light, and identifies an AI edit (FR-009). Tasks as written.
- **B: per provider (`Gemini: `, `Codex: `).** `git_manager._commit_subject` takes the provider,
  `undo_last_commit` accepts every registered prefix; adds a slice with tests in
  `tests/unit/test_git_manager.py`, and `CLAUDE.md`'s auto-commit invariant is reworded (T025).

### D7. Shield Gemini and Codex from the user's own instruction files?

**Answer (2026-10-30)**: A.

- **A (recommended): no; no FR asks.** `~/.gemini/GEMINI.md` and `~/.codex/AGENTS.md` may reach
  the model, as they would in the tools themselves. The Claude path's HOME isolation
  (`providers/claude.py: _make_cli_subprocess_call`) is not copied.
- **B: yes.** A temporary HOME with the credentials copied in, per CLI; Gemini's credential copy
  is unverified (R3). Adds one slice per CLI.

### D8. Also select the provider by environment variable?

**Answer (2026-10-30)**: A.

- **A (recommended): no; `--provider` only** (FR-001 asks for an explicit choice at start).
- **B: also `CLAUDE_LIGHT_PROVIDER`.** Needs its own unknown-name check listing `NAMES`, a row in
  `docs/CONFIG.md`, and tests; changes T002, T003, T025.

### D9. What verdict does Gemini's manual run get when Google rejects the account?

**Answer (2026-10-30)**: A. The user has no other Gemini credential, so T026 runs Gemini without
`GEMINI_API_KEY` and records `fail` with its output.

Gemini here is logged in but refused with `IneligibleTierError` (`research.md` R3).

- **A (recommended): run it and record `fail` with the output**, or the result of a run with
  `GEMINI_API_KEY` set if you provide one. The run still shows FR-011's error path. T026 as
  written.
- **B: record `not run`, treating a rejected account like a missing login.** Changes T026.
  Spec: FR-015: "...; a tool that is not logged in, or whose account the tool rejects, is reported
  as "not run"."

### D10. Which CLI drives the Gemini provider?

**Answer (2026-10-30)**: agy (Antigravity CLI) replaces the gemini CLI in providers/gemini.py, one
code path; agy's default model (no --model, as D3); built as a fix round on PR #117.

Google refuses the gemini CLI for individual accounts (`research.md` R3). Contract:
`contracts/provider.md` Gemini adapter; tasks T027-T029.
