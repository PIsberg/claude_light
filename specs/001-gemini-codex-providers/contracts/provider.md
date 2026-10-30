# Contract: providers

The one shared contract of FR-005, the `--provider` flag of FR-001, and the user-visible text of
FR-008, FR-010, FR-011, FR-012 and FR-016. Written as the answers recorded in `plan.md` D1-D10
make it.

## Package `claude_light/providers/`

```python
# claude_light/providers/__init__.py
NAMES: tuple[str, ...] = ("claude", "gemini", "codex")
DEFAULT: str = "claude"             # the provider when none is chosen; neutral code names it only through this
def get(name: str) -> Provider      # imports claude_light.providers.<name> on first use, returns its PROVIDER;
                                    # ValueError naming NAMES for an unknown name
def select(name: str) -> Provider   # get(name), then state.provider = it under state.lock
def active() -> Provider            # state.provider, or get(DEFAULT) when it is None

# claude_light/providers/base.py
CACHING, EFFORT, TOP_TIER, USD_BUDGET, USAGE, AGENT, SHELL = (
    "caching", "effort", "top_tier", "usd_budget", "usage", "agent", "shell")

class ProviderError(Exception):     # str(e) names the provider and the remedy (FR-011)
class NotLoggedIn(ProviderError):   # the tool needs a login: llm.chat re-raises it, main.start_chat calls offer_login

class Provider:
    name: str = ""
    capabilities: frozenset[str] = frozenset()
    def has(self, capability: str) -> bool                       # capability in self.capabilities
    def model_label(self, routed_model: str) -> str              # model the route line shows and the request carries
    def header(self) -> tuple[str, str] | None                   # (model, access) for the startup header; None = existing header
    def require_auth(self) -> None                               # SystemExit(1) with the message below when unusable
    def complete(self, request: dict) -> tuple[str, object | None, bool]   # (reply, usage or None, already printed)
    def summarize(self, instructions: str) -> tuple[str, object | None]    # (summary, usage or None)
    def offer_login(self, retry) -> None                         # after NotLoggedIn; base: no-op
    def warm(self, quiet: bool = False) -> None                  # prefill the prompt cache (llm.warm_cache); base: no-op

def transcript(request: dict) -> tuple[str, str]
    # (system_text, prompt): the flattening llm._make_streaming_api_call's OAUTH branch does today
    # (system block texts joined by "\n"; "CONVERSATION HISTORY:" + "--- Role ---" parts; "USER: <last>")

# claude_light/providers/runner.py
TIMEOUT_SECS: int = 600
def run_cli(provider: str, argv: list[str], stdin_text: str, timeout: float = TIMEOUT_SECS) -> str
    # stdout of a finished run with exit 0; ProviderError otherwise; kills the process tree on timeout
    # or on any exception (KeyboardInterrupt included) before re-raising
def kill_process_tree(proc) -> None   # moved unchanged from llm._kill_process_tree
```

`request` is the dict `llm.chat()` and `llm._one_shot_turn()` already build (`model`, `max_tokens`,
`system`, `messages`, `extra_headers`, `output_config`). Every module defines `PROVIDER`, its one
instance.

## Providers

| | `claude.py: ClaudeProvider` | `gemini.py: GeminiProvider` | `codex.py: CodexProvider` |
|---|---|---|---|
| capabilities | all seven | none | none |
| `model_label(m)` | `m` | `"CLI default model"` | `"CLI default model"` |
| `header()` | `None` | `("CLI default model", "agy CLI")` | `("CLI default model", "codex CLI")` |
| `complete` | `_make_streaming_api_call(**request)`, moved from `llm` | `run_cli` of the argv below | `run_cli` of the argv below |
| `summarize` | the two Claude branches of today's `llm._summarize_turns` | same argv and stdin line, its `content` `"You summarize conversations.\n\n" + instructions` | same argv, stdin `"You summarize conversations.\n\n" + instructions` |
| `warm` | the body of today's `llm.warm_cache` (returns under OAUTH) | base no-op | base no-op |
| `require_auth` | body of today's `llm.require_auth` | `shutil.which("agy")` only: agy keeps no credential file to check (none under `~/.gemini/antigravity-cli`), so a missing login surfaces from the run | `shutil.which("codex")`; `$CODEX_HOME/auth.json` (default `~/.codex/auth.json`) exists |
| also owns | moved from `llm` (D2): `client`, `ClaudeNotLoggedIn(NotLoggedIn)`, `_make_cli_subprocess_call`, `_api_tool_loop` and their helpers; `offer_login` (from `main.start_chat`) | `BINARY = "agy"` (native `agy.exe`, no npm shim), `SETTINGS` | `BINARY = "codex"` |

### Gemini adapter (agy, D10)

argv: `[which("agy"), "--input-format", "stream-json", "--output-format", "stream-json", "--mode", "plan", "--settings", SETTINGS, "-p="]`.
The empty attached `-p=` is required: a bare `-p` swallows the next flag as its prompt and agy
exits 2.

`SETTINGS` is one JSON string:
`{"permissions":{"allow":[],"deny":["command","mcp","file"]},"allowNonWorkspaceAccess":false}`.
Run with the user's own settings, agy executed the user's globally allowed
`mcp(lean-ctx/ctx_shell)`; with this override the same call failed with `permission check failed
for mcp "lean-ctx/ctx_shell": user denied permission`.

cwd: an empty temporary directory, as for every CLI provider.

stdin: exactly one NDJSON line, then EOF:
`{"event":"user","message":{"role":"user","content":<prompt text>}}`, the prompt text being
`system_text + "\n\n" + prompt` from `transcript()`. A plain-text prompt on stdin is not read: the
model tries a shell command to read stdin, headless mode denies it, and agy prints nothing.

stdout: NDJSON events (`init`, `step_update`, `result`). Non-JSON lines are skipped.
- reply: the `result` event's `result.response` when `result.status == "SUCCESS"`;
- a `result` with another status: `ProviderError` naming `result.error`;
- no `result` event: unparseable output, `ProviderError`;
- non-zero exit: the runner's existing error (agy exits 3 on a model or agent error, 2 on a
  usage error).

usage: `result.usage` has input, output, thinking, cache-read and total tokens, but D5 stands: the
provider reports no figures.

capabilities: none; shell disabled with the existing notice.

Known limit: `search_web` still runs under these settings (three searches in the probe), and no
permission kind was found that stops it. With the empty cwd and `allowNonWorkspaceAccess` false,
what it can send is the prompt claude_light gives it.

### Codex adapter

Codex argv, stdin = `system_text + "\n\n" + prompt` from `transcript()`, reply = `text` of the last JSONL line with `type == "item.completed"` and
`item.type == "agent_message"`:
`[which("codex"), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--sandbox", "read-only", "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "apps", "-"]`

Neither argv carries `-m` or any Claude model id (FR-007, D3). Why each flag: `research.md` R1 (Codex), R3 (Gemini).

## Flag

`--provider {claude,gemini,codex}`, default `claude`. An unknown name exits 2 through argparse:
`argument --provider: invalid choice: 'x' (choose from claude, gemini, codex)`.

## Isolation rule (FR-004, SC-003)

For every `claude_light/**/*.py` except `providers/<name>.py` and the registry
`providers/__init__.py` (whose `NAMES` and `DEFAULT` are provider names, not binaries; it imports
no SDK and runs nothing): no `import`/`from` of that provider's SDK (`anthropic` for claude;
`google.genai`, `google.generativeai` for gemini; `openai` for codex), and no string constant,
docstrings excluded, matching
`(?<![\w./-])(claude|gemini|codex)(\.exe|\.cmd)?(?![\w.-]| light)` for that name. f-string parts
count as constants. The rule matches the binary as a bare name (`"claude"`, `which("codex")`) or
as the first word of a command line (`"claude setup-token"`); it does not match a file name with
a suffix (`"claude.md"` in `claude_light/skeleton.py: _FULL_MD_NAMES`, the project doc every
provider's skeleton keeps under FR-003), the product name `claude light` (`claude_light/ui.py`
banner) or `claude_light` (an identifier character follows). Neutral code that needs the
default provider's name uses `providers.DEFAULT`.

## Messages

`<p>` is the provider name. Tests assert the quoted fragments.

| When | Text |
|---|---|
| tool missing (exit 1) | `agy not found. Install the Antigravity CLI.` / `codex CLI not found. Install it: npm install -g @openai/codex` |
| not logged in | Gemini: no startup check; the run's `ProviderError` (a non-`SUCCESS` result) names `result.error` and the remedy `run agy once and sign in` / Codex (exit 1): `codex is not logged in. Run: codex login` |
| binary gone mid-session (`FileNotFoundError` in `run_cli`) | `<p> CLI not found` |
| non-zero exit | `<p> failed (exit <n>): <last non-empty stderr line>. Run <bin> on its own to see the full error.`, where `<bin>` is the binary that ran, `Path(argv[0]).stem` (Gemini: `Run agy on its own ...`) |
| timeout | `<p> timed out after <s> s` |
| unparseable or empty output | `<p> output could not be parsed: <reason>` |
| route line | `◆  <p>  ·  CLI default model  ·  effort unavailable for <p>`, plus `  ·  top tier unavailable for <p>` when routed to the top tier |
| `/budget <x>` | `Budget unavailable for <p>: it has no dollar pricing. No cap set.` |
| turn stats, `/cost`, exit footer, status bar | `tokens, cost and caching unavailable for <p>` |
| `--allow-command` given | `shell access is disabled for <p>: its agent cannot run commands. --allow-command only lets claude_light run the affected tests.` |
| `--continue`, other provider's file | `The saved conversation was made with <saved>; starting a fresh <p> conversation. Resume it with --provider <saved> --continue.` |

A `ProviderError` inside a turn reaches the existing handler in `llm.chat()` (`[Error] API call
failed: <message>`) or `llm._one_shot_turn()` (`SystemExit`).
