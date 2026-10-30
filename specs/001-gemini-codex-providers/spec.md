# Feature Specification: Gemini and Codex Providers

**Feature Branch**: `001-gemini-codex-providers` (stacked on `ci/107-diagram-check`)

**Created**: 2026-10-30

**Status**: Draft

**Input**: User description: "implement support for gemini and codex too. make it modular and well written code so each claude, gemini and codex is seperated in code. at the end test each codex and gemini client with claude_light to see that it works too"

## Clarifications

### Session 2026-10-30

- Q: What happens when `--continue` resumes a session saved with a different provider? → A: claude_light warns, naming both providers, and starts a fresh conversation; the old session file is left untouched.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask a question through Gemini or Codex (Priority: P1)

A user who has the Gemini agent tool or the Codex agent tool installed and logged in on their machine picks that provider when starting claude_light. They ask a question about their codebase and get an answer built from the same retrieved code chunks and project skeleton that a Claude session would use.

**Why this priority**: Without a working question-and-answer path no other part of the feature has value.

**Independent Test**: Start claude_light once with each of the two new providers against a small test project, ask "What does OrderService do?", and check that a non-empty answer that refers to the project's code is printed.

**Acceptance Scenarios**:

1. **Given** the Gemini tool is installed and logged in, **When** the user starts claude_light choosing Gemini and asks a question, **Then** the answer is printed and the route/status line names Gemini as the provider.
2. **Given** the Codex tool is installed and logged in, **When** the user starts claude_light choosing Codex and asks a question, **Then** the answer is printed and the route/status line names Codex as the provider.
3. **Given** the user makes no provider choice, **When** claude_light starts, **Then** it behaves exactly as before this feature: Claude, with the existing credential order.
4. **Given** the user chose Gemini or Codex, **When** a question is asked, **Then** no Claude credential is required and no Claude request is made.

---

### User Story 2 - Edit code through Gemini or Codex with undo (Priority: P2)

A user on Gemini or Codex asks for a code change. The change is applied, committed through the same auto-commit path as Claude edits, and `/undo` reverts it.

**Why this priority**: Editing is a core claude_light capability, but a read-only provider is still useful, so it follows question answering.

**Independent Test**: With each new provider, ask for a one-line change in a scratch git repository, confirm a commit with the auto-commit message exists, run `/undo`, and confirm the file is restored.

**Acceptance Scenarios**:

1. **Given** a clean scratch repository and a chosen new provider, **When** the user asks for a small edit, **Then** the file changes and one auto-commit is created.
2. **Given** that auto-commit is the last commit, **When** the user runs `/undo`, **Then** the edit is reverted.
3. **Given** the provider's tool could run shell commands, **When** a command outside the `--allow-command` prefixes, or one containing a rejected metacharacter, is attempted, **Then** it is refused.
4. **Given** a provider whose tool cannot be restricted to the `--allow-command` prefixes, **When** a session starts, **Then** shell access is disabled for that provider and the user is told so.

---

### User Story 3 - Unsupported features say so (Priority: P2)

A user on Gemini or Codex tries something that exists only for Claude (prompt caching figures, per-turn effort, the top-tier model prompt, `/budget` in dollars). They are told plainly that it is unavailable for that provider; nothing is silently ignored or faked.

**Why this priority**: Silent no-ops would mislead users about cost and quality.

**Independent Test**: With each new provider run `/budget 2.50` and read the output.

**Acceptance Scenarios**:

1. **Given** a chosen provider without dollar pricing, **When** the user runs `/budget 2.50`, **Then** a message states the budget is unavailable for that provider and no cap is applied.
2. **Given** a chosen provider without the top-tier model prompt, **When** a question is routed, **Then** no Opus/Fable/Sonnet prompt appears.

---

### User Story 4 - Maintainer adds or changes one provider in isolation (Priority: P3)

A maintainer edits or removes one provider's code without touching the other two or the provider-neutral parts (routing, retrieval, indexing, compression, tool loop, auto-commit).

**Why this priority**: It is the stated goal behind the modular request, but it only pays off once the providers work.

**Independent Test**: Search the repository outside a provider's own module for that provider's SDK import or CLI binary name; the search finds nothing. Delete one provider's module in a scratch copy: the other two still run.

**Acceptance Scenarios**:

1. **Given** the finished code, **When** provider-neutral modules are searched for the names of the Anthropic SDK, the `claude`, `gemini` and `codex` binaries, **Then** no match is found.
2. **Given** the Claude provider moved behind the shared provider contract, **When** the existing test suite and the CI token benchmark run, **Then** all pass unchanged.

---

### User Story 5 - Verified against the real tools (Priority: P3)

Before the work is called done, the maintainer runs claude_light against the real Gemini tool and the real Codex tool and records the commands and their output.

**Why this priority**: Mocked tests prove the wiring, not that the real tools accept what claude_light sends.

**Independent Test**: Follow the recorded commands for each tool on a test project and compare the output.

**Acceptance Scenarios**:

1. **Given** a tool is installed and logged in, **When** the manual run is done, **Then** the report lists the exact commands and output and states pass or fail.
2. **Given** a tool is not logged in, **When** the manual run is attempted, **Then** the report states "not run", never "passed".

---

### Edge Cases

- The chosen provider's tool is not installed: claude_light exits before the first question with a message naming the missing tool and how to get it.
- The tool is installed but not logged in: the same early exit, naming the login step; no partial answer is printed.
- The tool fails or times out mid-answer: the user sees an error naming the provider, the session stays usable, and no auto-commit is made for a failed turn.
- The tool's output cannot be parsed: the user sees an error naming the provider and the turn is not recorded as answered.
- Two providers' credentials are both present: only the chosen provider's is used.
- An unknown provider name is given: claude_light exits listing the valid names.
- A provider has no cost or token figures: the status line shows them as unavailable, not as zero.
- Retrieved context larger than a provider's limit: it is trimmed the same way for every provider, and the user is told when it was.
- The user resumes a saved session (`--continue`) made with a different provider: claude_light warns, naming the saved and the chosen provider, and starts a fresh conversation. The saved session file is not modified or deleted, and resuming it with its own provider still works.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to choose Claude, Gemini or Codex explicitly when starting claude_light.
- **FR-002**: With no provider chosen, claude_light MUST behave exactly as before, including the existing Claude credential order and both Claude transports.
- **FR-003**: With Gemini or Codex chosen, a question MUST be answered from the same project skeleton and top-K retrieved chunks as for Claude.
- **FR-004**: Each provider's code MUST live in its own module; no code outside a provider's module may import that provider's SDK or name its tool binary.
- **FR-005**: Provider-neutral code MUST NOT branch on a provider's name; it MUST reach providers only through one shared contract.
- **FR-006**: Moving Claude behind the shared contract MUST NOT change Claude behaviour: the existing Claude tests pass unchanged and the CI token and retrieval regression gates pass.
- **FR-007**: Model ids, prices and effort levels MUST belong to their provider; Gemini and Codex MUST NOT use Claude constants.
- **FR-008**: A feature a provider lacks (prompt caching, per-turn effort, top-tier model prompt, dollar budget) MUST produce a visible "unavailable for <provider>" message when the user uses it, and MUST NOT be silently ignored.
- **FR-009**: Edits by any provider MUST go through the same auto-commit path so `/undo` works; the commit message prefix MUST identify the change as an AI edit.
- **FR-010**: Shell access by Gemini or Codex MUST obey the `--allow-command` prefix rule and the metacharacter rejection; if that cannot be enforced for a tool, shell access MUST be disabled for it and the user told.
- **FR-011**: A missing tool, a missing login, a tool failure and unparseable output MUST each produce an error naming the provider and the remedy, without crashing the session.
- **FR-012**: The status/route line MUST show the active provider and its model.
- **FR-013**: Tests for all three providers MUST run without a real API, a real tool or real stats/session files, under the existing test settings.
- **FR-014**: Documentation (README, CLAUDE.md commands and authentication, configuration reference, architecture doc, harness map) MUST describe provider selection, per-provider authentication and per-provider unavailable features in the same change.
- **FR-015**: Before completion, claude_light MUST be run once against the real Gemini tool and once against the real Codex tool, and the commands and output recorded; a tool that is not logged in is reported as "not run".
- **FR-016**: When `--continue` finds a saved session made with a different provider than the one chosen, claude_light MUST print a warning naming both providers, MUST start a fresh conversation, and MUST leave the saved session file byte-identical.

### Key Entities

- **Provider**: One of Claude, Gemini, Codex. Has a name, an authentication requirement, its models and prices (if any), and a list of features it supports.
- **Provider selection**: The user's explicit choice for a session; absent means Claude.
- **Capability**: A named feature a provider may lack (caching, effort, top-tier prompt, dollar budget, shell restriction); drives the "unavailable" messages.
- **Manual verification record**: Commands run against each real tool, their output, and a pass, fail or not-run verdict.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user with a logged-in Gemini or Codex tool gets a printed answer to a codebase question within 60 seconds on a small test project, for each of the two providers.
- **SC-002**: 100% of the existing tests pass unchanged, and the CI token and retrieval regression gates pass with no change to Claude's measured token counts.
- **SC-003**: 0 matches for provider SDK imports or tool binary names outside the owning provider's module.
- **SC-004**: For each new provider, an edit followed by `/undo` leaves the working tree identical to its starting state in 100% of 3 trial runs.
- **SC-005**: 100% of the listed Claude-only features (caching figures, effort, top-tier prompt, dollar budget) show an explicit "unavailable" message under each new provider.
- **SC-006**: 0 commands outside the allowed prefixes, and 0 commands with a rejected metacharacter, execute under any provider across the guard's test cases.
- **SC-007**: The manual verification record exists for both new tools, each with a verdict of pass, fail or not run.

## Assumptions

- The Gemini and Codex tools are already installed (Gemini 0.44.1, Codex 0.162.1 on the maintainer's machine); claude_light does not install them or log users in.
- Providers are chosen per launch; switching provider inside a running session is out of scope.
- Gemini and Codex are driven through their installed command-line tools, not through vendor SDKs, in this version.
- Prompt caching, per-turn effort, and dollar budgets stay Claude-only; equivalents for other providers are out of scope.
- Both new providers support question answering and editing; if a tool cannot support editing safely, that provider is read-only and the user is told.
- The end-to-end run against real tools is manual and opt-in; it is not part of CI.
- Provider-specific model routing between several models of one provider is out of scope beyond using each provider's default model.
