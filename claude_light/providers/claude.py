"""Claude provider: the anthropic SDK (API_KEY) or the claude CLI (OAUTH)."""
import sys
import os
import time
import json
import subprocess
import re
import shutil
import threading
from pathlib import Path

import claude_light.config as config
from claude_light.config import API_KEY, AUTH_MODE, MODEL, SUMMARY_MODEL, SYSTEM_PROMPT, ENABLE_STREAMING
from claude_light.ui import (
    _T_SYS, _T_ERR, _T_RAG, _T_CACHE, _T_EDIT,
    _ANSI_GREEN, _ANSI_CYAN, _ANSI_YELLOW, _ANSI_MAGENTA, _ANSI_BOLD, _ANSI_DIM, _ANSI_RESET, _ANSI_RED,
    _SYM_RESP, _SYM_TOOL, print_stats, _Spinner, calculate_cost, _print_reply, usage_unavailable, ResponseView, NullView, notify, choose,
)
import claude_light.state as state
from claude_light import providers
from claude_light.providers import base
from claude_light.providers.base import NotLoggedIn, Provider, transcript
from claude_light.providers.runner import kill_process_tree as _kill_process_tree
from claude_light.retry import retry_with_backoff
from claude_light.streaming import stream_chat_response, accumulate_usage_from_dict, calculate_usage_cost
from claude_light.llm import (
    _extract_text, _build_system_blocks, _accumulate_usage,
    _EXTENDED_CACHE_HEADERS, BASE_EFFORT, _API_TOOL_STEPS,
)


class ClaudeNotLoggedIn(NotLoggedIn):
    """Raised when the official Claude CLI is not logged in."""


if API_KEY == "sk-ant-test-mock-key":
    class MockClient:
        pass
    client = MockClient()
    client.messages = MockClient()
    client.messages.create = None
else:
    class _LazyClient:
        """The Anthropic client, built on first use. Importing the SDK took
        1.3 s warm and 4.9 s cold before the prompt appeared, and OAuth turns
        go through the Claude CLI and never need it."""

        def __init__(self):
            self._client = None
            self._lock = threading.Lock()

        def __getattr__(self, name):
            with self._lock:
                if self._client is None:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=API_KEY)
            return getattr(self._client, name)

    client = _LazyClient()


def _display_path(path: str) -> str:
    """Project-relative, forward-slash path; absolute only if outside cwd."""
    try:
        rel = os.path.relpath(path)
    except ValueError:  # different drive on Windows
        return path.replace("\\", "/")
    return (path if rel.startswith("..") else rel).replace("\\", "/")


def _describe_tool_call(name: str, tool_input) -> str:
    """One-line label for a CLI agent tool call, e.g. 'Read claude_light/llm.py'."""
    inp = tool_input if isinstance(tool_input, dict) else {}
    for key in ("file_path", "notebook_path", "pattern", "command", "url", "query", "path"):
        val = inp.get(key)
        if isinstance(val, str) and val.strip():
            if key in ("file_path", "notebook_path", "path"):
                val = _display_path(val)
            val = " ".join(val.split())
            if len(val) > 80:
                val = val[:77] + "..."
            return f"{name} {val}"
    return name


def _report_denied_commands(denials) -> None:
    """Tell the user which commands the agent was not allowed to run, and how
    to allow them (the agent itself only sees an error)."""
    commands = []
    for d in denials or []:
        cmd = ((d or {}).get("tool_input") or {}).get("command")
        if isinstance(cmd, str) and cmd.strip() and cmd not in commands:
            commands.append(" ".join(cmd.split()))
    if not commands:
        return
    print(f"\n  {_ANSI_DIM}{_SYM_TOOL} not allowed to run:{_ANSI_RESET}")
    for cmd in commands[:3]:
        print(f"    {_ANSI_DIM}{cmd[:100]}{_ANSI_RESET}")
    print(f"  {_ANSI_DIM}  allow a command prefix with --allow-command \"<prefix>\""
          f" or CLAUDE_LIGHT_ALLOW_COMMANDS{_ANSI_RESET}", flush=True)


# The CLI runs a full agent loop (read files, edit, re-read), so a multi-file
# change routinely takes several minutes while streaming events the whole
# time. Kill it only when it goes silent; the overall cap is a backstop.
_CLI_IDLE_TIMEOUT_SECS = 300
# Every built-in CLI tool, denied by name for tool-free calls.
_ALL_CLI_TOOLS = ",".join([
    "Bash", "PowerShell", "Edit", "MultiEdit", "Write", "NotebookEdit", "Read", "Glob",
    "Grep", "LS", "WebFetch", "WebSearch", "Task", "Agent", "TodoWrite",
])
_CLI_MAX_SECS = 3600


def _make_cli_subprocess_call(
    prompt: str,
    model: str = "",
    extra_flags: list | None = None,
    render: bool = True,
    tools: bool = True,
) -> tuple[str, any, str | None]:
    """
    Execute the official 'claude' CLI binary as a subprocess for OAUTH mode.
    Returns (reply_text, usage, session_id).
    Handles the 8,191 character limit on Windows by using temporary context files.
    """
    import shutil
    import tempfile
    import subprocess
    import json

    # 1. Determine the 'claude' CLI binary path
    # On Windows, we use the bare command string 'claude' and shell=True.
    # This allows the shell to find the official '.cmd' or '.ps1' shims which
    # set up the Bun/NPM environment correctly. Resolving full path can bypass these shims.
    if os.name == 'nt':
        claude_bin = "claude"
    else:
        claude_bin = shutil.which("claude")
        if not claude_bin:
            raise RuntimeError("Claude CLI binary not found. Please install with 'npm install -g @anthropic-ai/claude-code'")

    # 2. Build the command
    current_os = os.name
    temp_context_file = None
    temp_system_prompt_file = None
    isolated_home = None  # temp dir used to neutralize CLAUDE.md auto-discovery

    try:
        # Prepare the environment
        env = os.environ.copy()

        # Inject the automation token on all platforms.
        # On Linux/macOS: primary auth mechanism (no Credential Manager).
        # On Windows: fallback — CLI prefers the Credential Manager, but will use
        # this env var if no Credential Manager entry exists.
        if config.AUTH_TOKEN and config.AUTH_TOKEN.startswith("sk-ant-oat01"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = config.AUTH_TOKEN

        # Isolate HOME/USERPROFILE so the Claude CLI's CLAUDE.md auto-discovery
        # cannot pull in the user's personal ~/.claude/CLAUDE.md. Without this,
        # instructions like "always prefer ctx_read/ctx_shell MCP tools" leak
        # into the model's context and it tries tools it does not have here.
        # The `--bare`
        # CLI flag would also disable this, but `--bare` forbids OAuth reads
        # even when CLAUDE_CODE_OAUTH_TOKEN is set, so it breaks subscription
        # users. This HOME-override sidesteps that: we keep OAuth auth via the
        # env var (or by copying the real .credentials.json into the fake HOME)
        # while neutralizing user-memory leakage.
        import tempfile as _tempfile
        import shutil as _shutil
        isolated_home = _tempfile.mkdtemp(prefix="claude_light_home_")
        fake_claude_dir = Path(isolated_home) / ".claude"
        fake_claude_dir.mkdir(parents=True, exist_ok=True)
        (fake_claude_dir / "CLAUDE.md").write_text("", encoding="utf-8")
        # Copy real credentials.json so OAuth still resolves when the env-var
        # token path isn't available (e.g., user only has .credentials.json,
        # not the sk-ant-oat01 automation token).
        real_creds = Path(os.path.expanduser("~")) / ".claude" / ".credentials.json"
        if real_creds.is_file():
            try:
                _shutil.copy2(str(real_creds), str(fake_claude_dir / ".credentials.json"))
            except OSError:
                pass
        env["HOME"] = isolated_home
        env["USERPROFILE"] = isolated_home

        # The HOME override does not cover a repo that lives under the
        # user's home (~/code/x): the CLI walks up parent directories for
        # project memory and finds ~/.claude/CLAUDE.md as an ancestor's
        # .claude/CLAUDE.md. Exclude it by its real path as well.
        real_user_md = (
            os.path.expanduser("~").replace("\\", "/") + "/.claude/CLAUDE.md"
        )
        settings = {"claudeMdExcludes": [real_user_md]}
        if tools and config.ALLOWED_COMMANDS:
            # An allow rule covers only a command's first statement, so
            # `echo a && touch b` ran under Bash(echo:*) (issue #70). The
            # guard blocks chaining for both shells before the rule is read.
            from claude_light import cli_shell_guard
            guard = (f'"{Path(sys.executable).as_posix()}" -I '
                     f'"{Path(cli_shell_guard.__file__).as_posix()}"')
            settings["hooks"] = {"PreToolUse": [{
                "matcher": "Bash|PowerShell",
                "hooks": [{"type": "command", "command": guard}],
            }]}
        settings_path = Path(isolated_home) / "claude_light_settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        # On Windows, ensure critical path variables are present for the Bun-based CLI to find its profile
        if current_os == 'nt':
            # Note: USERPROFILE was just overridden above to isolated_home.
            # APPDATA/LOCALAPPDATA/HOMEDRIVE/HOMEPATH intentionally remain the
            # real user's values so things like Node caches keep working.
            real_home = os.path.expanduser('~')  # real user HOME for app caches
            env.setdefault('APPDATA', os.environ.get('APPDATA', str(Path(real_home) / "AppData" / "Roaming")))
            env.setdefault('LOCALAPPDATA', os.environ.get('LOCALAPPDATA', str(Path(real_home) / "AppData" / "Local")))
            env.setdefault('HOMEDRIVE', os.environ.get('HOMEDRIVE', 'C:'))
            env.setdefault('HOMEPATH', os.environ.get('HOMEPATH', real_home.split(':', 1)[1] if ':' in real_home else real_home))

        # Windows cmd.exe command line limit is 8,191 characters total (all args combined).
        # On Windows, ALWAYS use temp files for both prompt and system prompt.
        # The prompt is an @-mention of its file (expanded by the CLI); the
        # system prompt goes through --system-prompt-file.
        cmd_prompt = prompt
        processed_extra_flags = list(extra_flags) if extra_flags else []

        if current_os == 'nt':
            # Write main prompt to temp file
            fd, path = tempfile.mkstemp(suffix=".md", prefix="claude_light_ctx_")
            temp_context_file = path
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(prompt)
            safe_path = Path(temp_context_file).as_posix()
            cmd_prompt = f"Please follow the instructions and use the codebase context provided in @{safe_path}"

            # Check for --system-prompt in extra_flags and move it to a temp file too
            new_flags = []
            i = 0
            while i < len(processed_extra_flags):
                flag = processed_extra_flags[i]
                if flag == "--system-prompt" and i + 1 < len(processed_extra_flags):
                    # Next item is the system prompt text - move it to a temp file
                    system_prompt_text = processed_extra_flags[i + 1]
                    fd, path = tempfile.mkstemp(suffix=".md", prefix="claude_light_sys_")
                    temp_system_prompt_file = path
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(system_prompt_text)
                    safe_sys_path = Path(temp_system_prompt_file).as_posix()
                    # --system-prompt-file, not `--system-prompt @path`: the
                    # CLI takes the latter literally, so the skeleton never
                    # reached the model (verified with claude 2.1.310).
                    new_flags.append("--system-prompt-file")
                    new_flags.append(safe_sys_path)
                    i += 2
                else:
                    new_flags.append(flag)
                    i += 1
            processed_extra_flags = new_flags

        # No --bare on any OS: it never reads OAuth credentials, so subscription
        # users get "Not logged in" (verified with claude 2.1.310 even with
        # CLAUDE_CODE_OAUTH_TOKEN set and .credentials.json present).
        # stream-json + --verbose + --include-partial-messages gives us per-
        # token content_block_delta events so we can echo text as the model
        # generates it, instead of the old blocking subprocess.run() that
        # showed nothing but "Processing…" for up to 3 minutes.
        #
        # Permissions: acceptEdits lets the agent edit files without asking.
        # With Bash enabled the CLI also runs commands it classifies as
        # read-only; any other command needs an allow rule. With -p and stdin=DEVNULL an
        # unapproved one is denied (the model gets an error), never left
        # waiting on a prompt. config.ALLOWED_COMMANDS is the user's opt-in
        # list, e.g. the test command, so the agent can check its own edits.
        if tools:
            permission_mode = "acceptEdits"
            # A shell stays off unless the user allowlisted a command for it.
            shells = {config.command_rule(c)[0] for c in config.ALLOWED_COMMANDS}
            disallowed = ",".join(t for t in ("PowerShell", "Bash") if t not in shells)
        else:
            # A hidden call over text that may quote repo content (the
            # history summary): no tools at all. --tools "" alone is not
            # relied on, since CLI 2.1.118 ignored it; with the default
            # mode and -p, any edit is denied rather than auto-accepted.
            permission_mode = "default"
            disallowed = _ALL_CLI_TOOLS
        command = [claude_bin]
        command += [
            "-p", cmd_prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--permission-mode", permission_mode,
            # Only claude_light's own settings apply. A repo's
            # .claude/settings*.json could add allow rules or hooks (which run
            # commands) and its .mcp.json could add servers. This also stops
            # the CLI loading the repo's CLAUDE.md, which the skeleton prompt
            # already carries along with every other .md file.
            "--setting-sources", "user",
            "--strict-mcp-config",
            "--settings", settings_path.as_posix(),
            *(["--disallowed-tools", disallowed] if disallowed else []),
        ]
        if not tools:
            command += ["--tools", ""]
        elif config.ALLOWED_COMMANDS:
            rules = [config.command_rule(c) for c in config.ALLOWED_COMMANDS]
            command += ["--allowedTools", ",".join(f"{tool}({prefix}:*)" for tool, prefix in rules)]
        if model:
            command += ["--model", model]
        if processed_extra_flags:
            command += processed_extra_flags
        
        # 3. Execute (streamed)
        # - stdin=DEVNULL so the child can never block waiting on an
        #   inherited stdin (observed as silent "Processing..." hangs).
        # - Popen + line-by-line stdout so stream-json events surface as
        #   they arrive. A watchdog thread enforces the overall timeout
        #   (Popen has no timeout on iteration).
        # - shell=True on Windows ensures the Bun-based CLI finds its shim.
        import threading as _threading

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            encoding='utf-8',
            errors='replace',
            shell=(current_os == 'nt'),
            stdin=subprocess.DEVNULL,
            bufsize=1,
        )

        timed_out = _threading.Event()
        idle_out = _threading.Event()
        start_time = time.monotonic()
        last_activity = [start_time]  # list so the stdout loop can update it

        def _watchdog():
            while proc.poll() is None:
                now = time.monotonic()
                if now - last_activity[0] > _CLI_IDLE_TIMEOUT_SECS:
                    idle_out.set()
                elif now - start_time > _CLI_MAX_SECS:
                    timed_out.set()
                if idle_out.is_set() or timed_out.is_set():
                    try:
                        _kill_process_tree(proc)
                    except Exception:
                        pass
                    return
                time.sleep(min(0.5, _CLI_IDLE_TIMEOUT_SECS / 4))

        _wd = _threading.Thread(target=_watchdog, daemon=True)
        _wd.start()

        # One view for the whole turn: spinner until the first token, the
        # reply as rendered markdown while it streams, a line per tool call.
        view = ResponseView("Thinking") if render else NullView()
        view.__enter__()

        reply_parts: list[str] = []
        streamed_via_delta = False
        usage_data: dict = {}
        session_id = None
        final_result_text: str | None = None
        cli_cost_usd = None
        denials = None
        # The agent narrates in separate text blocks between tool calls.
        # Without a break they print as one run-on paragraph.
        new_block = False
        after_tool = False  # the view already separates text from a tool line
        shown_tool_ids: set = set()

        def _emit_text(text):
            nonlocal new_block, after_tool
            if new_block and reply_parts:
                reply_parts.append("\n\n")
                if not after_tool:
                    view.text("\n\n")
            new_block = after_tool = False
            view.text(text)
            reply_parts.append(text)

        def _show_tool_call(block):
            nonlocal new_block, after_tool
            tool_id = block.get("id")
            if tool_id in shown_tool_ids:
                return
            shown_tool_ids.add(tool_id)
            view.tool(_describe_tool_call(block.get("name", "tool"), block.get("input")))
            view.status("Working")
            new_block = after_tool = True

        for raw_line in proc.stdout:
            last_activity[0] = time.monotonic()
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            ev_type = event.get("type")

            if ev_type == "stream_event":
                # Per-token deltas when --include-partial-messages is active.
                inner = event.get("event", {})
                if inner.get("type") == "content_block_start":
                    block_type = (inner.get("content_block") or {}).get("type")
                    if block_type == "text":
                        new_block = True
                    elif block_type == "thinking":
                        view.status("Thinking")
                elif inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            _emit_text(text)
                            streamed_via_delta = True

            elif ev_type == "assistant":
                # The CLI emits one complete block per event. Text is only
                # printed here if deltas didn't already cover it.
                msg = event.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "tool_use":
                        _show_tool_call(block)
                    elif block.get("type") == "text" and not streamed_via_delta:
                        text = block.get("text", "")
                        if text:
                            new_block = True
                            _emit_text(text)

            elif ev_type == "result":
                final_result_text = event.get("result")
                usage_data = event.get("usage", {}) or {}
                cli_cost_usd = event.get("total_cost_usd")
                denials = event.get("permission_denials")
                session_id = event.get("session_id")

        returncode = proc.wait()

        if not reply_parts and final_result_text and returncode == 0:
            # Fallback: deltas and assistant events missing (older CLI or
            # unexpected format), so show the terminal result.
            reply_parts.append(final_result_text)
            view.text(final_result_text)
        view.close()
        if render:
            _report_denied_commands(denials)

        if timed_out.is_set() or idle_out.is_set():
            try:
                stderr_tail = (proc.stderr.read() or "")[-2000:]
            except Exception:
                stderr_tail = ""
            reason = (
                f"produced no output for {_CLI_IDLE_TIMEOUT_SECS:g}s"
                if idle_out.is_set()
                else f"timed out after {_CLI_MAX_SECS:g}s"
            )
            raise RuntimeError(
                f"Claude CLI {reason}. "
                f"Last stderr: {stderr_tail.strip() or '(empty)'}"
            )

        if returncode != 0:
            try:
                stderr_output = proc.stderr.read() or ""
            except Exception:
                stderr_output = ""
            error_output = stderr_output or (final_result_text or "")
            if "Not logged in" in error_output or "Invalid API key" in error_output:
                raise ClaudeNotLoggedIn("Claude CLI: Not logged in. Please run 'claude auth login' in your terminal.")
            raise RuntimeError(f"Claude CLI error (code {returncode}): {error_output}")

        response_text = "".join(reply_parts)

        class UsageObj:
            pass
        usage = UsageObj()
        usage.input_tokens = usage_data.get("input_tokens", 0)
        usage.output_tokens = usage_data.get("output_tokens", 0)
        usage.cache_creation_input_tokens = usage_data.get("cache_creation_input_tokens", 0)
        usage.cache_read_input_tokens = usage_data.get("cache_read_input_tokens", 0)
        # The CLI splits writes by TTL and reports the exact cost of the turn
        # (every model it ran); calculate_cost() prefers that figure.
        usage.cache_creation = usage_data.get("cache_creation")
        usage.cost_usd = cli_cost_usd

        return response_text.strip(), usage, session_id

    finally:
        # A turn that ends early (Ctrl+C, an error mid-stream) must not leave
        # the agent running: it edits files with permission prompts off.
        try:
            _kill_process_tree(proc)
        except NameError:
            pass  # Popen never ran

        # Always close the view (idempotent) so a raised exception does not
        # leave the spinner running on screen.
        try:
            view.close()
        except NameError:
            pass  # raised before the view existed

        # Cleanup temporary files if created
        for temp_file in (temp_context_file, temp_system_prompt_file):
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

        # Cleanup the isolated HOME directory (contains only the empty
        # CLAUDE.md and a copy of .credentials.json — safe to remove).
        if isolated_home and os.path.isdir(isolated_home):
            try:
                import shutil as _shutil
                _shutil.rmtree(isolated_home, ignore_errors=True)
            except Exception:
                pass

def _make_streaming_api_call(**create_kwargs):
    """
    Make an API call using streaming if enabled, otherwise use non-streaming.
    In OAUTH mode, routes to the official Claude CLI backend via subprocess.
    """
    from claude_light.config import AUTH_MODE

    if AUTH_MODE == "OAUTH":
        model = create_kwargs.get("model", MODEL)

        system_text, prompt = transcript(create_kwargs)
        if config.ALLOWED_COMMANDS:
            # The agent may run the tests, so it edits with its own tools and
            # can check the result (SEARCH/REPLACE lands only after the turn).
            system_text = system_text.replace(SYSTEM_PROMPT, config.AGENT_SYSTEM_PROMPT, 1)
            # Name the tool for each command, so the agent calls Bash or
            # PowerShell to match the shell the command is written for.
            by_tool = {}
            for c in config.ALLOWED_COMMANDS:
                tool, prefix = config.command_rule(c)
                by_tool.setdefault(tool, []).append(prefix)
            system_text += (
                "\n\nYou may run these shell commands to check your work, for example"
                " the tests after an edit: "
                + "; ".join(f"With the {tool} tool: {'; '.join(prefixes)}"
                            for tool, prefixes in by_tool.items())
                + ". Other commands, apart from read-only ones, will be refused."
            )

        extra_flags = ["--system-prompt", system_text]

        try:
            reply_text, usage, _session_id = _make_cli_subprocess_call(
                prompt, model=model, extra_flags=extra_flags
            )
            # _make_cli_subprocess_call renders the reply as the CLI emits
            # stream-json events (ResponseView), so the caller must not
            # print it again.
            return reply_text, usage, True
        except ClaudeNotLoggedIn:
            raise
        except Exception as e:
            print(f"  {_T_ERR}  failed ({e})", flush=True)
            raise

    # API Key mode — use the anthropic client directly
    if ENABLE_STREAMING and hasattr(client.messages, 'stream'):
        try:
            return _api_tool_loop(create_kwargs)  # already streamed to stdout
        except (AttributeError, NotImplementedError):
            @retry_with_backoff
            def _call_api():
                return client.messages.create(**create_kwargs)
            response = _call_api()
            return _extract_text(response.content), response.usage, False
    else:
        @retry_with_backoff
        def _call_api():
            return client.messages.create(**create_kwargs)
        response = _call_api()
        return _extract_text(response.content), response.usage, False

# Most rounds of tool calls in one API-key turn before the model must answer.


class _TurnUsage:
    """Token usage summed over the API calls of one tool-loop turn."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation = {"ephemeral_1h_input_tokens": 0}
        self.cost_usd = None

    def add(self, usage):
        from claude_light.ui import _count, _one_hour_writes
        if isinstance(usage, dict):  # a stream that gave no final message
            self.input_tokens += _count(usage.get("input_tokens", 0))
            self.output_tokens += _count(usage.get("output_tokens", 0))
            self.cache_creation_input_tokens += _count(usage.get("cache_creation_tokens", 0))
            self.cache_read_input_tokens += _count(usage.get("cache_read_tokens", 0))
            return
        self.input_tokens += _count(getattr(usage, "input_tokens", 0))
        self.output_tokens += _count(getattr(usage, "output_tokens", 0))
        self.cache_creation_input_tokens += _count(getattr(usage, "cache_creation_input_tokens", 0))
        self.cache_read_input_tokens += _count(getattr(usage, "cache_read_input_tokens", 0))
        self.cache_creation["ephemeral_1h_input_tokens"] += _one_hour_writes(usage)


def _api_tool_loop(create_kwargs):
    """One API-key turn with read-only tools (read_file, grep, list_files).

    Without tools the model only saw the retrieved chunks and could not open
    a file retrieval missed. Edits still come back as SEARCH/REPLACE blocks in
    the reply text. Returns (reply_text, usage, streamed=True).
    """
    from claude_light import tools
    kwargs = dict(create_kwargs, tools=tools.TOOLS)
    messages = list(create_kwargs["messages"])
    usage = _TurnUsage()
    texts = []
    with ResponseView("Thinking") as view:  # one view, so rounds read as one reply
        _run_tool_rounds(kwargs, messages, usage, texts, view)
    return "\n\n".join(texts), usage, True


def _run_tool_rounds(kwargs, messages, usage, texts, view):
    from claude_light import tools
    for step in range(_API_TOOL_STEPS + 1):
        kwargs["messages"] = messages
        try:
            text, usage_dict = stream_chat_response(client, view=view, **kwargs)
        except ValueError:
            # Tool input the SDK could not parse (eager input streaming).
            # Retry the request once; a second failure is a real error.
            text, usage_dict = stream_chat_response(client, view=view, **kwargs)
        message = usage_dict.get("message")
        usage.add(message.usage if message is not None else usage_dict)
        if text and text.strip():
            texts.append(text.strip())
        # Only a completed tool_use turn runs tools: never on max_tokens (the
        # input may be cut off) or refusal.
        if message is None or getattr(message, "stop_reason", None) != "tool_use":
            break
        if step == _API_TOOL_STEPS:
            print(f"  {_T_ERR}  stopped after {_API_TOOL_STEPS} rounds of tool calls", flush=True)
            break
        results = []
        for block in message.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            view.tool(_describe_tool_call(block.name, block.input))
            view.status("Working")
            output, is_error = tools.run_tool(block.name, block.input)
            result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
            if is_error:
                result["is_error"] = True
            results.append(result)
        # One moving cache breakpoint on the newest results keeps the request
        # within the 4-breakpoint limit while each round reads the last one.
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        b.pop("cache_control", None)
        if results:
            results[-1]["cache_control"] = {"type": "ephemeral"}
        messages = messages + [
            {"role": "assistant", "content": message.content},  # thinking blocks included, unchanged
            {"role": "user", "content": results},
        ]


class ClaudeProvider(Provider):
    name = "claude"
    capabilities = frozenset({base.CACHING, base.EFFORT, base.TOP_TIER,
                              base.USD_BUDGET, base.USAGE, base.AGENT, base.SHELL})

    def model_label(self, routed_model: str) -> str:
        return routed_model

    def require_auth(self) -> None:
        """Exit if no credential was found. main() calls it once the arguments
        are parsed, not at import, so --help works on a fresh install (#101)."""
        from claude_light.ui import _T_ERR, _T_SYS, _ANSI_CYAN, _ANSI_RESET, _ANSI_BOLD
        if not config.AUTH_MODE or (config.AUTH_MODE == "OAUTH"
                                    and not config.AUTH_TOKEN and not config.API_KEY):
            print(f"\n  {_T_ERR}  No authentication found.", file=sys.stderr)
            print(f"  {_T_SYS}  To use your Claude Pro subscription:", file=sys.stderr)
            print(f"          {_ANSI_CYAN}claude auth login{_ANSI_RESET}\n", file=sys.stderr)
            print(f"  {_T_SYS}  Or set {_ANSI_BOLD}ANTHROPIC_API_KEY{_ANSI_RESET} in your environment.", file=sys.stderr)
            sys.exit(1)

    def offer_login(self, retry) -> None:
        """After NotLoggedIn: offer `claude setup-token`, save the token, then retry()."""
        print(f"\n  {_T_ERR}  Claude CLI is not logged in.")
        choice = choose("Set up a long-lived automation token?",
                        [("yes", "Yes, run claude setup-token"), ("no", "No")])
        if choice == "yes":
            claude_bin = shutil.which("claude")
            if claude_bin:
                print(f"  {_T_SYS}  Launching 'claude setup-token'…")
                print(f"  {_ANSI_DIM}Follow the browser prompt, then paste the token below.{_ANSI_RESET}")
                subprocess.run([claude_bin, "setup-token"])
                print(f"\n  Automation Token: ", end="", flush=True)
                try:
                    new_token = input().strip()
                    if new_token.startswith("sk-ant-oat01"):
                        token_path = Path.home() / ".claude_light_automation_token"
                        token_path.write_text(new_token, encoding="utf-8")
                        print(f"  {_T_SYS}  Token saved to {token_path}")
                        config.API_KEY, config.AUTH_MODE, config.API_KEY_SOURCE, config.AUTH_TOKEN = config._resolve_api_key()
                        retry()
                    else:
                        print(f"  {_T_ERR}  Token must start with 'sk-ant-oat01'.")
                except (KeyboardInterrupt, EOFError):
                    print(f"\n  {_T_SYS}  Cancelled.")
            else:
                print(f"  {_T_ERR}  Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        else:
            print(f"  {_T_SYS}  Proceeding without auth. Requests will likely fail.")

    def complete(self, request: dict):
        return _make_streaming_api_call(**request)

    def warm(self, quiet: bool = False) -> None:
        if AUTH_MODE == "OAUTH":
            return
        from claude_light import tools as _tools
        with state.lock:
            skeleton = state.skeleton_context
            # Set after the first turn when ROUTE_ONCE is on: warm what turns read.
            model = state.session_model or MODEL
        try:
            response = client.messages.create(
                model=model,
                # 0 = prefill only: writes the cache, bills no output tokens.
                max_tokens=0,
                # Tools render before the system prompt in the cache prefix; warm
                # with the same list the turns send, or this warms nothing they use.
                tools=_tools.TOOLS,
                system=_build_system_blocks(skeleton),
                messages=[{"role": "user", "content": "ok"}],
                # Effort is rendered into the prompt; warm at the level turns send.
                output_config={"effort": BASE_EFFORT},
                extra_headers=_EXTENDED_CACHE_HEADERS,
            )
            cost = calculate_cost(response.usage, model)
            _accumulate_usage(response.usage, model)
            with state.lock:
                state.session_cost += cost

            if not quiet:
                print_stats(response.usage, label="Cache warm", model=model)
        except Exception as e:
            if quiet:
                notify(f"cache warm failed: {e}", error=True)
            else:
                print(f"{_T_ERR} Cache warm failed: {e}")

    def summarize(self, instructions: str):
        if AUTH_MODE == "OAUTH":
            # No API key in this mode: the summary goes through the CLI too.
            text, usage, _ = _make_cli_subprocess_call(
                instructions, model=SUMMARY_MODEL,
                extra_flags=["--system-prompt", "You summarize conversations."],
                render=False,
                tools=False,  # the turns may quote repo text; nothing may act on it
            )
            return text.strip(), usage

        @retry_with_backoff
        def _call_api():
            return client.messages.create(
                model=SUMMARY_MODEL,
                # Thinking is always on and shares this cap with the summary.
                max_tokens=4_000,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": instructions}],
            )

        response = _call_api()
        return response.content[0].text.strip(), response.usage


PROVIDER = ClaudeProvider()
