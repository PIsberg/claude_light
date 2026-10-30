# Contract: providers

The one shared contract of FR-005, the `--provider` flag of FR-001, and the user-visible text of
FR-008, FR-010, FR-011, FR-012 and FR-016. Written as the answers recorded in `plan.md` D1-D9
make it.

## Package `claude_light/providers/`

```python
# claude_light/providers/__init__.py
NAMES: tuple[str, ...] = ("claude", "gemini", "codex")
def get(name: str) -> Provider      # imports claude_light.providers.<name> on first use, returns its PROVIDER;
                                    # ValueError naming NAMES for an unknown name
def select(name: str) -> Provider   # get(name), then state.provider = it under state.lock
def active() -> Provider            # state.provider, or get("claude") when it is None

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
| `header()` | `None` | `("CLI default model", "gemini CLI")` | `("CLI default model", "codex CLI")` |
| `complete` | `_make_streaming_api_call(**request)`, moved from `llm` | `run_cli` of the argv below | `run_cli` of the argv below |
| `summarize` | the two Claude branches of today's `llm._summarize_turns` | same argv, stdin `"You summarize conversations.\n\n" + instructions` | as Gemini |
| `warm` | the body of today's `llm.warm_cache` (returns under OAUTH) | base no-op | base no-op |
| `require_auth` | body of today's `llm.require_auth` | `shutil.which("gemini")`; `GEMINI_API_KEY`, `GOOGLE_API_KEY` or `GOOGLE_GENAI_USE_VERTEXAI` set, or `~/.gemini/oauth_creds.json` exists | `shutil.which("codex")`; `$CODEX_HOME/auth.json` (default `~/.codex/auth.json`) exists |
| also owns | moved from `llm` (D2): `client`, `ClaudeNotLoggedIn(NotLoggedIn)`, `_make_cli_subprocess_call`, `_api_tool_loop` and their helpers; `offer_login` (from `main.start_chat`) | `BINARY = "gemini"` | `BINARY = "codex"` |

Gemini argv, stdin = `system_text + "\n\n" + prompt` from `transcript()`, reply = stripped stdout:
`[which("gemini"), "-p", "Answer the request given on standard input.", "-o", "text", "--approval-mode", "plan", "--allowed-mcp-server-names", "__claude_light_none__"]`

Codex argv, same stdin, reply = `text` of the last JSONL line with `type == "item.completed"` and
`item.type == "agent_message"`:
`[which("codex"), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--sandbox", "read-only", "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "apps", "-"]`

Neither argv carries `-m` or any Claude model id (FR-007, D3). Why each flag: `research.md`.

## Flag

`--provider {claude,gemini,codex}`, default `claude`. An unknown name exits 2 through argparse:
`argument --provider: invalid choice: 'x' (choose from claude, gemini, codex)`.

## Isolation rule (FR-004, SC-003)

For every `claude_light/**/*.py` except `providers/<name>.py`: no `import`/`from` of that
provider's SDK (`anthropic` for claude; `google.genai`, `google.generativeai` for gemini; `openai`
for codex), and no string constant, docstrings excluded, matching
`(?<![\w./-])(claude|gemini|codex)(\.exe|\.cmd)?(?![\w-])` for that name. f-string parts count as
constants.

## Messages

`<p>` is the provider name. Tests assert the quoted fragments.

| When | Text |
|---|---|
| tool missing (exit 1) | `gemini CLI not found. Install it: npm install -g @google/gemini-cli` / `codex CLI not found. Install it: npm install -g @openai/codex` |
| not logged in (exit 1) | `gemini is not logged in. Run gemini once and sign in, or set GEMINI_API_KEY.` / `codex is not logged in. Run: codex login` |
| binary gone mid-session (`FileNotFoundError` in `run_cli`) | `<p> CLI not found` |
| non-zero exit | `<p> failed (exit <n>): <last non-empty stderr line>. Run <p> on its own to see the full error.` |
| timeout | `<p> timed out after <s> s` |
| unparseable or empty output | `<p> output could not be parsed: <reason>` |
| route line | `◆  <p>  ·  CLI default model  ·  effort unavailable for <p>`, plus `  ·  top tier unavailable for <p>` when routed to the top tier |
| `/budget <x>` | `Budget unavailable for <p>: it has no dollar pricing. No cap set.` |
| turn stats, `/cost`, exit footer, status bar | `tokens, cost and caching unavailable for <p>` |
| `--allow-command` given | `shell access is disabled for <p>: its agent cannot run commands. --allow-command only lets claude_light run the affected tests.` |
| `--continue`, other provider's file | `The saved conversation was made with <saved>; starting a fresh <p> conversation. Resume it with --provider <saved> --continue.` |

A `ProviderError` inside a turn reaches the existing handler in `llm.chat()` (`[Error] API call
failed: <message>`) or `llm._one_shot_turn()` (`SystemExit`).
