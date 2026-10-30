# claude_light Constitution

## Core Principles

### I. Hermetic Tests

Tests MUST NOT call a real API or a real agent CLI (`claude`, `gemini`, `codex`), and MUST NOT
write the real stats or session files. Every test MUST run under the `tests/conftest.py`
settings: a mock API key, `CLAUDE_LIGHT_HYBRID=0`, a temporary `CLAUDE_LIGHT_STATS_FILE` and a
temporary `CLAUDE_LIGHT_SESSION_DIR`. SDKs and subprocesses are mocked at the boundary.

Rationale: a test that reaches the network bills, flakes, and passes or fails on credentials
rather than on code.

### II. Tested Behaviour

Every behaviour change MUST ship with a pytest test under `tests/unit/` or `tests/integration/`,
in the existing style, so `.github/workflows/test.yml` runs it. The test MUST assert what a
caller observes, not internals.

Rationale: a check that only ran on one machine is not coverage.

### III. Measured Token Changes

A change that can affect tokens sent or received MUST be measured before and after with the
benchmarks in `docs/BENCHMARKS.md`, and kept only if it shows a gain or is neutral by design.
The CI token and retrieval regression gates (`tests/linting/check_regression.py`) MUST pass.

Rationale: token cost is the product; an unmeasured change to it is a guess.

### IV. Runtime Invariants

These MUST hold in every change:
- Background threads never print; they call `ui.notify()`.
- Shared state goes through `state.py` under `state.lock`; a new cross-module global goes
  through `__init__._wrap()`.
- `route_query()` stays pure; session effects (budget cap, top-tier prompt, `ROUTE_ONCE`) live
  in `_finalize_route()`.
- `index.pkl` is never loaded; the disk cache loads with `allow_pickle=False`.
- `state.turn_count` counts turns and is never derived from `conversation_history`.
- The top-level effort never changes per request.

Rationale: each of these guards a defect that has already occurred (garbled output, stale
bindings, pinned benchmark routes, hostile pickles, lost cache).

### V. Contained Shell Access

Any agent CLI that can run shell commands MUST be limited to the `--allow-command` prefixes,
with commands containing `; & | < > `` ` `` $` or a newline rejected, and for PowerShell also
`( ) { } @`. A CLI that cannot be restricted this way MUST run with shell access disabled.

Rationale: claude_light runs agents inside cloned repositories it does not control.

### VI. Docs Move With Code

Docs MUST change in the same change as the behaviour they describe: `README.md`, `CLAUDE.md`,
`HARNESS.md`, `docs/CONFIG.md` and `docs/architecture/ARCHITECTURE.md` as the change touches
them. A section describing removed behaviour MUST be removed.

Rationale: an agent acts on the docs; stale docs produce wrong edits.

## Sources

Principles I, III, IV and V restate rules in `CLAUDE.md`. Principle II comes from `TESTING.md`
and `.github/workflows/test.yml`. Principle VI is inferred from the docs routing table in
`CLAUDE.md`.

## Development Workflow

Work goes on a feature branch off `main` and lands by pull request. CI (`test.yml`) runs the
unit, integration and utilities suites and the token and retrieval regression gates; a PR is
ready only when every check is green.

## Governance

This constitution takes precedence over other practice in this repository. Amendments are made
with `speckit-constitution` and approved by the maintainer. Versioning follows semantic
versioning: MAJOR for removing or redefining a principle, MINOR for adding a principle or
materially expanding one, PATCH for wording. Every spec audit and final verification checks
compliance with these principles.

**Version**: 1.0.0 | **Ratified**: 2026-10-30 | **Last Amended**: 2026-10-30
