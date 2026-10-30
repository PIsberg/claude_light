# Research: Gemini and Codex Providers

Checked 2026-10-30 on the maintainer's machine. "Help" means the installed CLI's `--help`;
"probe" means a real run in the session scratch directory, outside the repository.

## R1. Flags each argv relies on

| Flag | Source | Use |
|---|---|---|
| agy (Gemini, D10) | probe, R3 | every agy flag and why |
| `codex exec ... -` | help (0.162.1): "if `-` is used, instructions are read from stdin" | prompt on stdin; avoids the Windows 8,191-character command line |
| `codex exec --json` | help + probe: `thread.started`, `turn.started`, `item.completed` (`item.type == "agent_message"`, `text`), `turn.completed` (`usage`) | reply = last agent message |
| `--ephemeral`, `--skip-git-repo-check` | help | no session files; claude_light also runs outside git |
| `--ignore-user-config` | help + probe | drops `~/.codex/config.toml`, and with it user MCP servers; auth still works (probe answered) |
| `--sandbox read-only`, `--disable shell_tool`, `--disable unified_exec`, `--disable apps` | help; `codex features list` shows `shell_tool`, `unified_exec`, `apps` | no shell, no writes, no connectors |

Decision: the Codex argv in `contracts/provider.md`; Gemini's in R3.

## R2. Codex shell is disabled only with user config ignored

Probe 1, `--disable shell_tool --disable unified_exec` with user config: the model listed
`mcp__lean_ctx__ctx_shell` and called it. User MCP servers keep shell access.
Probe 2, adding `--ignore-user-config --disable apps`, asked to write `marker2.txt` and run
`echo SHELLRAN`: it answered "no shell command tool is available", emitted no command item, and no
file appeared. Decision: all four restrictions together (FR-010, constitution V). Consequence:
the user's configured Codex model is ignored too, so Codex runs its built-in default (D3).

## R3. Gemini runs through agy, the Antigravity CLI (D10)

Why: the `gemini` CLI exited 1 for this account with `IneligibleTierError: This client is no
longer supported for Gemini Code Assist for individuals ... migrate to Antigravity` (verdict rule
D9).

Probed with agy 1.4.11 (native `agy.exe`). Its default model here is Gemini 3.8 Flash (High), from
the user's agy settings; the argv passes no `--model` (D3).

- Input: `--input-format stream-json` reads NDJSON from stdin; the user line is
  `{"event":"user","message":{"role":"user","content":...}}`, found by probing. `"message"` must
  be an object: a string fails with `cannot unmarshal string into ... StreamInputUserMessage`. A
  plain-text prompt on stdin is not read: the model tries a shell command to read stdin, headless
  mode denies it, and agy prints nothing.
- `-p=`: the empty attached form is required; a bare `-p` swallows the next flag as its prompt and
  agy exits 2.
- Output: `--output-format stream-json` emits `init`, `step_update` and `result` events; the reply
  is `result.response` when `result.status == "SUCCESS"`, and `result.usage` holds input, output,
  thinking, cache-read and total tokens (unused, D5). Exit 3 on a model or agent error, 2 on a
  usage error.
- Permissions: run with the user's own settings, agy executed the globally allowed
  `mcp(lean-ctx/ctx_shell)`. With `--settings` set to
  `{"permissions":{"allow":[],"deny":["command","mcp","file"]},"allowNonWorkspaceAccess":false}`
  the same call failed with `permission check failed for mcp "lean-ctx/ctx_shell": user denied
  permission`. Decision: that override plus `--mode plan` (FR-010, constitution V).
- Limit: `search_web` still ran (three searches) under the override, and no permission kind was
  found that stops it. With the empty cwd and `allowNonWorkspaceAccess` false, what it can send is
  the prompt claude_light gives it.
- Auth: agy keeps no credential file under `~/.gemini/antigravity-cli`, so `require_auth` checks
  only that `agy` is on PATH.

## R4. Starting a CLI on Windows

`gemini` and `codex` are npm `.cmd` shims (`C:\Users\...\npm\codex`). `shutil.which()` resolves
them through PATHEXT, and `subprocess.Popen` starts a `.cmd` path from an argv list. The argv
holds only fixed flags, never user text (that goes on stdin), so cmd.exe re-parsing cannot inject
anything. The Claude path's `shell=True` (`providers/claude.py: _make_cli_subprocess_call`, moved
from `llm`) is not copied.

## R5. Which existing tests change when Claude's transport moves (D2)

Existing tests reach Claude's transport through `claude_light.llm` (grep, 2026-10-30):
`patch("claude_light.llm.client.messages.create")` 36 times, `...llm.stream_chat_response` 21,
`...llm.print_stats` 9, `...llm._make_streaming_api_call` 6, and `patch.object(llm, ...)` on
`client` 3, `_make_cli_subprocess_call` 3, `AUTH_MODE` 3, `_make_streaming_api_call` 2,
`_CLI_IDLE_TIMEOUT_SECS` 2, `_CLI_MAX_SECS` 1, `warm_cache` 1. The eight files:
`tests/unit/test_claude_light.py`, `test_llm.py`, `test_effort_cache.py`, `test_cli_shell_guard.py`,
`test_retry.py`, `test_top_tier.py`, `tests/integration/test_api_tool_loop.py` and
`tests/utilities/test_mocks.py` (which patches `claude_light.llm.client.messages`).

A patch on `llm` does not reach code moved to another module. So each patch whose name the moved
code now looks up in `providers/claude.py` is retargeted to `claude_light.providers.claude`
(FR-006). Patches that still reach their code stay as they are: names neutral `llm` code keeps
using (`print_stats` in `chat`, `AUTH_MODE` in the OAuth checks), and attributes set through a
module object `llm` still imports (`claude_light.llm.subprocess.Popen` sets `Popen` on `subprocess`
itself, 12 patches). A test that steers both `llm`'s and the transport's `AUTH_MODE` patches both.
