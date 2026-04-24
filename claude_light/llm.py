import sys
import os
import time
import json
import subprocess
import shutil
import anthropic
from pathlib import Path

import claude_light.config as config
from claude_light.config import (
    API_KEY, AUTH_MODE, API_KEY_SOURCE, ECONOMY_MODE, MODEL, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS, SUMMARY_MODEL,
    MAX_HISTORY_TURNS, SUMMARIZE_BATCH, _RETRIEVAL_BUDGET, SYSTEM_PROMPT,
    ENABLE_STREAMING
)
from claude_light.ui import (
    _T_ROUTE, _T_SYS, _T_ERR, _T_RAG, _T_CACHE, _T_EDIT,
    _ANSI_GREEN, _ANSI_CYAN, _ANSI_YELLOW, _ANSI_MAGENTA, _ANSI_BOLD, _ANSI_DIM, _ANSI_RESET, _ANSI_RED,
    _SYM_RESP, print_stats, _Spinner, calculate_cost, _print_reply
)
import claude_light.state as state
from claude_light.skeleton import build_skeleton, _refresh_tree_only, _assemble_skeleton, _refresh_single_md
from claude_light.indexer import index_files
from claude_light.retrieval import retrieve
from claude_light.editor import parse_edit_blocks, apply_edits
from claude_light.retry import retry_with_backoff
from claude_light.compressor import compress_context

class ClaudeNotLoggedIn(Exception):
    """Raised when the official Claude CLI is not logged in."""
    pass
from claude_light.streaming import stream_chat_response, accumulate_usage_from_dict, calculate_usage_cost

# Auth check — only abort if nothing was found
if not AUTH_MODE or (AUTH_MODE == "OAUTH" and not config.AUTH_TOKEN and not API_KEY):
    print(f"\n  {_T_ERR}  No authentication found.", file=sys.stderr)
    print(f"  {_T_SYS}  To use your Claude Pro subscription:", file=sys.stderr)
    print(f"          {_ANSI_CYAN}claude auth login{_ANSI_RESET}\n", file=sys.stderr)
    print(f"  {_T_SYS}  Or set {_ANSI_BOLD}ANTHROPIC_API_KEY{_ANSI_RESET} in your environment.", file=sys.stderr)
    sys.exit(1)

if API_KEY == "sk-ant-test-mock-key":
    class MockClient:
        pass
    client = MockClient()
    client.messages = MockClient()
    client.messages.create = None
else:
    # OAuth mode still needs a client object for non-chat calls (e.g. summarisation).
    # It is only used in API_KEY mode for actual chat — OAuth routes via CLI subprocess.
    client = anthropic.Anthropic(api_key=API_KEY if API_KEY else "sk-ant-placeholder")

def route_query(query: str) -> tuple[str, str, int]:
    import re
    q = query.lower()
    words = q.split()
    word_count = len(words)
    
    # 1. Hard Bounds & Quick Routes
    _META_SIGNALS = {"hi", "hello", "help", "who are you", "what are you", "clear history", "compact"}
    if word_count < 5 and any(s in q for s in _META_SIGNALS):
        # Extremely simple/meta queries → Haiku
        return _route_result("low", MODEL_HAIKU, 2_048)
    
    # 2. Define Weighted Signals
    _ARCH_SIGNALS = {
        "architect", "architecture", "design system", "deeply", "deep analysis",
        "performance", "optimize", "security", "scalability", "trade-off",
        "trade off", "tradeoff", "cross-cutting", "evaluate", "compare",
        "strategy", "reasoning", "step by step", "comprehensive", "deadlock", "race condition"
    }
    _LOGIC_SIGNALS = {
        "implement", "write", "create", "add", "refactor", "fix", "debug",
        "build", "develop", "generate", "update", "change", "modify",
        "migrate", "convert", "extend", "integrate", "logic", "algorithm"
    }
    _INFRA_SIGNALS = {
        "list", "show", "where", "what is", "what are",
        "how many", "print", "display", "tell me", "which file",
        "which files", "find", "locate", "count", "search", "grep"
    }

    # 3. Calculate Base Score
    score = 0
    score += sum(4.0 for s in _ARCH_SIGNALS  if s in q)
    score += sum(5.0 for s in _LOGIC_SIGNALS if s in q)
    score += sum(0.5 for s in _INFRA_SIGNALS if s in q)

    # 4. Contextual Multipliers
    # Mentioning a file path (e.g. main.py, lib/utils.py) suggests technical intent
    if re.search(r'[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{2,4}', q):
        score += 3.0

    # Word count implies complexity
    if word_count > 30:
        score += 5.0
    elif word_count > 10:
        score += 2.0
    elif word_count > 4:
        score += 1.5
    
    # Conversation depth - deeper history requires better instruction following
    history_depth = len(state.conversation_history) // 2
    score += (history_depth * 0.5)

    # 5. Determine Effort Level
    if score >= 12.0:
        effort, model, max_tokens = "max",    MODEL_OPUS,   16_000
    elif score >= 5.0:
        effort, model, max_tokens = "high",   MODEL_SONNET,  8_192
    elif score >= 1.5 or (word_count > 15):
        effort, model, max_tokens = "medium", MODEL_SONNET,  4_096
    else:
        effort, model, max_tokens = "low",    MODEL_HAIKU,   2_048

    return _route_result(effort, model, max_tokens)

def _route_result(effort, model, max_tokens) -> tuple[str, str, int]:
    _effort_color = {
        "low":    _ANSI_GREEN,
        "medium": _ANSI_CYAN,
        "high":   _ANSI_YELLOW,
        "max":    _ANSI_MAGENTA,
    }
    color = _effort_color[effort]
    model_short = model.replace("claude-", "").replace("-20251001", "")
    print(
        f"  {_ANSI_DIM}◆  {model_short}  ·  {color}{effort}{_ANSI_RESET}",
        file=sys.stderr,
    )
    return model, effort, max_tokens

def _extract_text(content_blocks) -> str:
    return "".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )

def _accumulate_usage(usage):
    from claude_light.config import PRICE_INPUT, PRICE_READ
    input_tokens = usage.input_tokens
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    output_tokens = usage.output_tokens

    with state.lock:
        state.session_tokens["input"]       += input_tokens
        state.session_tokens["cache_write"] += cache_write
        state.session_tokens["cache_read"]  += cache_read
        state.session_tokens["output"]      += output_tokens
        
        # Update Global Stats
        # Saved tokens = everything that hit the cache (read)
        # Full tokens = everything that didn't hit the cache (input + write)
        state.global_stats["total_tokens_full"]  += (input_tokens + cache_write)
        state.global_stats["total_tokens_saved"] += cache_read
        
        # Dollars saved = tokens * (Full Price - Cache Read Price)
        dollars_saved = (cache_read / 1_000_000.0) * (PRICE_INPUT - PRICE_READ)
        state.global_stats["total_dollars_saved"] += dollars_saved
    
    # Persist after every interaction
    state.save_global_stats()

def _accumulate_compression_stats(info: dict) -> None:
    """Record tokens saved by LLMLingua-2 in global_stats.

    Priced at PRICE_WRITE: the retrieved-context block is new on nearly every
    query (unique top-K chunks), so its first trip through the API is a cache
    write. Pricing at PRICE_INPUT would overstate savings; PRICE_READ would
    understate them. PRICE_WRITE is the honest middle path.
    """
    if not info or info.get("skipped"):
        return
    from claude_light.config import PRICE_WRITE
    delta = max(0, info["tokens_before"] - info["tokens_after"])
    if delta == 0:
        return
    dollars = (delta / 1_000_000.0) * PRICE_WRITE
    with state.lock:
        state.global_stats["total_tokens_pre_compress"]    += info["tokens_before"]
        state.global_stats["total_tokens_post_compress"]   += info["tokens_after"]
        state.global_stats["total_dollars_saved_llmlingua"] += dollars
    state.save_global_stats()

def _summarize_turns(messages: list) -> tuple:
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        lines.append(f"{role}: {content}")
    dialogue = "\n\n".join(lines)
    
    @retry_with_backoff
    def _call_api():
        return client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize the following conversation turns concisely (under 300 words).\n"
                    "Capture: what was asked, decisions reached, files edited and what changed, "
                    "key facts established about the codebase.\n"
                    "Output only the summary — no preamble.\n\n"
                    f"<turns>\n{dialogue}\n</turns>"
                ),
            }],
        )
    
    response = _call_api()
    return response.content[0].text.strip(), response.usage

def _maybe_compress_history():
    if len(state.conversation_history) <= MAX_HISTORY_TURNS * 2:
        return
    batch_msgs   = SUMMARIZE_BATCH * 2
    to_summarize = state.conversation_history[:batch_msgs]
    remaining    = state.conversation_history[batch_msgs:]
    try:
        print(f"  {_T_SYS}  Compressing history…", end="", flush=True)
        summary_text, usage = _summarize_turns(to_summarize)
        _accumulate_usage(usage)
        with state.lock:
            state.session_cost += calculate_cost(usage)
        summary_pair = [
            {"role": "user",      "content": f"[Summary of earlier conversation]\n{summary_text}"},
            {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
        ]
        state.conversation_history = summary_pair + remaining
        print(f" done")
    except Exception as e:
        print(f"\n  {_T_ERR}  History compression failed ({e}) — truncating.")
        state.conversation_history = state.conversation_history[-(MAX_HISTORY_TURNS * 2):]

def _build_system_blocks(skeleton):
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton, "cache_control": {"type": "ephemeral"}},
    ]
    return blocks


def _git_modified_snapshot():
    """Return the set of files currently dirty in the working tree.

    Used in OAUTH mode to detect edits the Claude CLI agent made directly
    via its own Read/Edit/Bash tools (`--tools ""` doesn't actually disable
    those in CLI 2.1.118). We snapshot before the API call and diff after
    so we only commit files the agent touched during this turn, not
    whatever the user had dirty beforehand.
    """
    from claude_light import git_manager
    if not git_manager.is_git_repo():
        return set()
    return set(git_manager.get_modified_files())


def _commit_agent_edits(new_files, explanation, auto_apply=False):
    """Commit edits the OAUTH CLI agent made directly (outside our
    SEARCH/REPLACE pipeline). Shows a git diff per file first; if stdin is
    interactive and auto_apply is False, asks before committing.
    """
    from claude_light import git_manager
    files = sorted(new_files)
    if not files:
        return

    print()
    for f in files:
        print(f"  {_T_EDIT}  {f}  {_ANSI_DIM}(agent edit){_ANSI_RESET}")

    # Render the diff straight from git so we don't have to track old content
    import subprocess as _sub
    for f in files:
        print(f"  {_ANSI_DIM}── {f}{_ANSI_RESET}")
        try:
            diff_out = _sub.run(
                ["git", "diff", "--", f],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                stdin=_sub.DEVNULL,
            ).stdout
        except Exception:
            diff_out = ""
        # Cap the diff so a huge refactor doesn't scroll the terminal.
        lines = diff_out.splitlines()
        max_lines = 80
        for line in lines[:max_lines]:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"{_ANSI_GREEN}{line}{_ANSI_RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"{_ANSI_RED}{line}{_ANSI_RESET}")
            else:
                print(f"{_ANSI_DIM}{line}{_ANSI_RESET}")
        if len(lines) > max_lines:
            print(f"  {_ANSI_DIM}… {len(lines) - max_lines} more line(s) truncated{_ANSI_RESET}")

    if auto_apply:
        commit = True
    elif not sys.stdin.isatty():
        commit = True
    else:
        print(
            f"  Commit {_ANSI_BOLD}{len(files)}{_ANSI_RESET} agent-made change(s)? "
            f"[{_ANSI_GREEN}y{_ANSI_RESET}/{_ANSI_RED}n{_ANSI_RESET}] ",
            end="", flush=True,
        )
        try:
            commit = input().strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            commit = False

    if not commit:
        print(f"  {_T_EDIT}  Kept in working tree; not committed.\n")
        return

    git_manager.auto_commit(files, explanation)


class _Heartbeat:
    """Animated 'Processing… (Ns)' indicator with an elapsed-seconds counter.

    Used as a context manager around a potentially long blocking call so the
    user can tell the process is alive and how long it's been going. Call
    .stop() as soon as real output starts so the counter doesn't interleave
    with the response stream.
    """

    def __init__(self, message: str = "Processing", interval: float = 1.0):
        import threading as _threading
        self.message = message
        self.interval = interval
        self._stop_event = _threading.Event()
        self._thread: _threading.Thread | None = None
        self._start_time: float = 0.0
        self._stopped = False

    def __enter__(self):
        import threading as _threading
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._stopped = False
        # Initial frame so the user sees *something* immediately.
        print(
            f"\r\033[K  {_ANSI_DIM}⏺  {self.message}…{_ANSI_RESET}",
            end="", flush=True,
        )
        self._thread = _threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def stop(self):
        """Stop the heartbeat and clear the line. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        print("\r\033[K", end="", flush=True)

    def _run(self):
        # Tick on self.interval without burning a CPU; stop() flips the event.
        while not self._stop_event.wait(self.interval):
            elapsed = int(time.monotonic() - self._start_time)
            print(
                f"\r\033[K  {_ANSI_DIM}⏺  {self.message}… ({elapsed}s){_ANSI_RESET}",
                end="", flush=True,
            )


def _make_cli_subprocess_call(
    prompt: str,
    model: str = "",
    extra_flags: list | None = None,
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
        # into the model's context and it hallucinates <tool_call> blocks we
        # cannot honor (we pass --tools "" so no real tool exists). The `--bare`
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
        # The @file syntax is supported by the Claude CLI and bypasses cmd.exe limits.
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
                    # Use @file syntax for system prompt
                    new_flags.append("--system-prompt")
                    new_flags.append(f"@\\{safe_sys_path}")
                    i += 2
                else:
                    new_flags.append(flag)
                    i += 1
            processed_extra_flags = new_flags

        # --bare disables the CLI system prompt for a cleaner response.
        # On Windows it also breaks Credential Manager access, so we omit it there.
        # stream-json + --verbose + --include-partial-messages gives us per-
        # token content_block_delta events so we can echo text as the model
        # generates it, instead of the old blocking subprocess.run() that
        # showed nothing but "Processing…" for up to 3 minutes.
        #
        # --dangerously-skip-permissions: the CLI runs with stdin=DEVNULL so
        # any interactive permission prompt would block forever. This flag is
        # the standard way to run the CLI non-interactively; the user has
        # already opted in by running claude_light in this directory.
        # Bash and PowerShell are still blocked separately because they can
        # execute arbitrary shell commands beyond just file edits.
        command = [claude_bin]
        if current_os != 'nt':
            command.append("--bare")
        command += [
            "-p", cmd_prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--dangerously-skip-permissions",
            "--disallowed-tools", "Bash,PowerShell",
        ]
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
        _CLI_TIMEOUT_SECS = 180

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
        start_time = time.monotonic()

        def _watchdog():
            while proc.poll() is None:
                if time.monotonic() - start_time > _CLI_TIMEOUT_SECS:
                    timed_out.set()
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return
                time.sleep(0.5)

        _wd = _threading.Thread(target=_watchdog, daemon=True)
        _wd.start()

        # Heartbeat: animated "Processing… (Ns)" so the user can tell the
        # process is alive during warmup (before the first token arrives).
        # Stops the moment the first text event lands so it doesn't
        # interleave with streamed output.
        heartbeat = _Heartbeat("Processing")
        heartbeat.__enter__()

        header_printed = False
        reply_parts: list[str] = []
        streamed_via_delta = False
        usage_data: dict = {}
        session_id = None
        final_result_text: str | None = None

        def _ensure_header():
            nonlocal header_printed
            if not header_printed:
                heartbeat.stop()  # clears the "Processing…" line
                print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)
                header_printed = True

        for raw_line in proc.stdout:
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
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            _ensure_header()
                            print(text, end="", flush=True)
                            reply_parts.append(text)
                            streamed_via_delta = True

            elif ev_type == "assistant":
                # Complete message; only print if deltas didn't already cover it.
                if not streamed_via_delta:
                    msg = event.get("message", {})
                    for block in msg.get("content", []):
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                _ensure_header()
                                print(text, end="", flush=True)
                                reply_parts.append(text)

            elif ev_type == "result":
                final_result_text = event.get("result")
                usage_data = event.get("usage", {}) or {}
                session_id = event.get("session_id")

        returncode = proc.wait()

        if header_printed:
            # flush=True because stdout is block-buffered when piped, so the
            # trailing \n would otherwise stay in-buffer and print_stats()
            # would land on the same line as the final response text.
            print(flush=True)

        if timed_out.is_set():
            try:
                stderr_tail = (proc.stderr.read() or "")[-2000:]
            except Exception:
                stderr_tail = ""
            raise RuntimeError(
                f"Claude CLI timed out after {_CLI_TIMEOUT_SECS}s. "
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
        if not response_text and final_result_text:
            # Fallback: deltas and assistant events missing (older CLI or
            # unexpected format) — use the terminal result. Print it now
            # so the user still sees something.
            response_text = final_result_text
            _ensure_header()
            print(response_text, flush=True)

        # Heartbeat.stop() in the finally block clears any lingering
        # "Processing…" line if we never printed real output.

        class UsageObj:
            pass
        usage = UsageObj()
        usage.input_tokens = usage_data.get("input_tokens", 0)
        usage.output_tokens = usage_data.get("output_tokens", 0)
        usage.cache_creation_input_tokens = usage_data.get("cache_creation_input_tokens", 0)
        usage.cache_read_input_tokens = usage_data.get("cache_read_input_tokens", 0)

        return response_text.strip(), usage, session_id

    finally:
        # Always stop the heartbeat (idempotent) so a raised exception
        # doesn't leave the animated line ticking on screen forever.
        try:
            heartbeat.stop()
        except NameError:
            # Raised before Popen succeeded — heartbeat never constructed.
            pass

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
        system_blocks = create_kwargs.get("system", [])
        messages      = create_kwargs.get("messages", [])
        model         = create_kwargs.get("model", MODEL)

        # Every turn: pass the full system prompt via --system-prompt and
        # serialize our own conversation history into the prompt body. We
        # used to use --resume <session_id> for Turn 2+ to reuse CLI-side
        # session state, but that path hangs on large sessions and
        # sometimes resumes an unrelated session from another project.
        # The Anthropic cache still hits on the re-sent skeleton
        # (5-minute TTL, content-hashed), so cost is unchanged in practice.
        def _text_of(content):
            if isinstance(content, list):
                return "\n".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            return content or ""

        system_text = "\n".join(
            b.get("text", "") for b in system_blocks if b.get("text")
        )

        history_parts = []
        for msg in messages[:-1]:
            role = msg["role"].capitalize()
            history_parts.append(f"--- {role} ---\n{_text_of(msg.get('content'))}")
        history_text = "\n".join(history_parts)

        last_user_text = _text_of(messages[-1]["content"]) if messages else ""

        prompt = (
            (f"CONVERSATION HISTORY:\n{history_text}\n\n" if history_text else "")
            + f"USER: {last_user_text}"
        )
        extra_flags = ["--system-prompt", system_text]

        try:
            reply_text, usage, _session_id = _make_cli_subprocess_call(
                prompt, model=model, extra_flags=extra_flags
            )
            # _make_cli_subprocess_call now streams text inline as the CLI
            # emits stream-json events and owns its own "Processing…"
            # heartbeat, so the caller must not print anything here.
            return reply_text, usage, True
        except ClaudeNotLoggedIn:
            raise
        except Exception as e:
            print(f"  {_T_ERR}  failed ({e})", flush=True)
            raise

    # API Key mode — use the anthropic client directly
    if ENABLE_STREAMING and hasattr(client.messages, 'stream'):
        try:
            reply_text, usage_dict = stream_chat_response(client, **create_kwargs)
            class UsageObj:
                pass
            usage = UsageObj()
            usage.input_tokens = usage_dict.get('input_tokens', 0)
            usage.output_tokens = usage_dict.get('output_tokens', 0)
            usage.cache_creation_input_tokens = usage_dict.get('cache_creation_tokens', 0)
            usage.cache_read_input_tokens = usage_dict.get('cache_read_tokens', 0)
            return reply_text, usage, True  # already streamed to stdout
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

def _apply_skeleton(new_skeleton):
    with state.lock:
        state.skeleton_context = new_skeleton
        state.last_interaction = time.time()

def _update_skeleton():
    _apply_skeleton(build_skeleton())

def warm_cache(quiet=False):
    if AUTH_MODE == "OAUTH":
        return
    with state.lock:
        skeleton = state.skeleton_context
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1,
            system=_build_system_blocks(skeleton),
            messages=[{"role": "user", "content": "ok"}],
        )
        cost = calculate_cost(response.usage)
        _accumulate_usage(response.usage)
        with state.lock:
            state.session_cost += cost

        if not quiet:
            print_stats(response.usage, label="Cache warm")
    except Exception as e:
        print(f"{_T_ERR} Cache warm failed: {e}")

def full_refresh():
    with _Spinner("Building skeleton") as spinner:
        _update_skeleton()
        spinner.update("Indexing source files")
        index_files(quiet=True)
        if AUTH_MODE == "API_KEY":
            spinner.update("Warming cache")
            warm_cache(quiet=True)

def refresh_skeleton_only():
    _apply_skeleton(_refresh_tree_only())
    warm_cache()

def refresh_md_file(path_str: str):
    if _refresh_single_md(path_str):
        _apply_skeleton(_assemble_skeleton())
        warm_cache()


def chat(query, auto_apply=False):
    routed_model, effort, max_tok = route_query(query)
    token_budget = _RETRIEVAL_BUDGET[effort]

    from claude_light.indexer import _chunk_label
    retrieved_ctx, hits = retrieve(query, token_budget=token_budget, effort=effort)

    compression_info = None
    if retrieved_ctx and config.LLMLINGUA_ENABLED:
        retrieved_ctx, compression_info = compress_context(retrieved_ctx)
        _accumulate_compression_stats(compression_info)
        if compression_info and not compression_info.get("skipped"):
            pct = compression_info["ratio"] * 100
            print(
                f"  {_T_RAG}  {_ANSI_DIM}LLMLingua "
                f"{compression_info['tokens_before']:,}→{compression_info['tokens_after']:,} "
                f"({pct:.0f}%, {compression_info['elapsed_ms']:.0f} ms){_ANSI_RESET}"
            )

    if hits:
        names = "  ".join(_chunk_label(p) for p, _ in hits)
        print(f"  {_T_RAG}  {_ANSI_DIM}{names}{_ANSI_RESET}")

    with state.lock:
        skeleton = state.skeleton_context

    _maybe_compress_history()
    trimmed = state.conversation_history[-(MAX_HISTORY_TURNS * 2):]

    if trimmed:
        last_msg = trimmed[-1]
        if isinstance(last_msg["content"], str):
            trimmed[-1] = {
                "role": last_msg["role"],
                "content": [{"type": "text", "text": last_msg["content"], "cache_control": {"type": "ephemeral"}}]
            }

    current_blocks = []
    if retrieved_ctx:
        current_blocks.append({
            "type": "text",
            "text": f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n",
            "cache_control": {"type": "ephemeral"},
        })
    import re
    from claude_light.editor import _ANY_BLOCK
    current_blocks.append({"type": "text", "text": f"Question:\n{query}"})
    messages = trimmed + [{"role": "user", "content": current_blocks}]

    system = _build_system_blocks(skeleton)

    create_kwargs = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=system,
        messages=messages,
    )
    if effort == "max":
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}

    # Snapshot dirty files BEFORE the API call so we can distinguish agent
    # edits (OAUTH mode: the CLI's own Read/Edit/Bash tools aren't actually
    # blocked by --tools "") from changes the user had pending.
    pre_modified = _git_modified_snapshot() if AUTH_MODE == "OAUTH" else set()

    try:
        for attempt in range(3):
            reply, usage, was_streamed = _make_streaming_api_call(**create_kwargs)

            edits = parse_edit_blocks(reply)

            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    cost = calculate_cost(usage)
                    _accumulate_usage(usage)
                    with state.lock:
                        state.session_cost += cost

                    print(f"\n  {_T_ERR}  {_ANSI_RED}Syntax errors detected — requesting correction (attempt {attempt+1}/3)…{_ANSI_RESET}")
                    for err in lint_errs:
                        print(f"    {_ANSI_DIM}{err.strip().replace(chr(10), ' ')}{_ANSI_RESET}")

                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."

                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})

                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]:
                                b.pop("cache_control", None)
                    continue

            state.conversation_history.append({"role": "user", "content": query})

            if edits:
                labels = ", ".join(e["path"] for e in edits)
                clean_reply = _ANY_BLOCK.sub("", reply).strip()
                clean_reply = (clean_reply + f"\n[Files edited: {labels}]").strip()
            else:
                clean_reply = reply
            state.conversation_history.append({"role": "assistant", "content": clean_reply})

            cost = calculate_cost(usage)
            _accumulate_usage(usage)
            with state.lock:
                state.last_interaction = time.time()
                state.session_cost += cost

            turns = len(state.conversation_history) // 2

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation and not was_streamed:
                print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)
                _print_reply(explanation)
                sys.stdout.flush()
            if edits:
                apply_edits(edits, explanation=explanation, auto_apply=auto_apply)
            elif AUTH_MODE == "OAUTH":
                # No SEARCH/REPLACE block but the CLI agent may have edited
                # files itself. Diff against the pre-call snapshot; commit
                # anything new so /undo can revert it. Use the user's
                # original query as the commit message — the agent's reply
                # is narration ("Let me first check…") and makes for a
                # noisy commit log.
                post_modified = _git_modified_snapshot()
                agent_edits = post_modified - pre_modified
                if agent_edits:
                    _commit_agent_edits(agent_edits, query, auto_apply=auto_apply)

            print_stats(usage, label=f"Turn {turns}")
            break

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except ClaudeNotLoggedIn:
        # Re-raise so main.py can handle the interactive login trigger
        raise
    except Exception as e:
        print(f"\n[Error] API call failed: {e}")

def one_shot(prompt, auto_apply=False):
    print(f"{_T_SYS} Building context...", file=sys.stderr)
    _update_skeleton()
    index_files()

    routed_model, effort, max_tok = route_query(prompt)
    retrieved_ctx, hits = retrieve(prompt, token_budget=_RETRIEVAL_BUDGET[effort], effort=effort)

    if retrieved_ctx and config.LLMLINGUA_ENABLED:
        retrieved_ctx, info = compress_context(retrieved_ctx)
        _accumulate_compression_stats(info)

    from claude_light.indexer import _chunk_label
    import re
    from claude_light.editor import _ANY_BLOCK

    if hits:
        names = "  ".join(_chunk_label(p) for p, _ in hits)
        print(f"  {_T_RAG}  {_ANSI_DIM}{names}{_ANSI_RESET}", file=sys.stderr)

    with state.lock:
        skeleton = state.skeleton_context

    ctx_prefix = f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n" if retrieved_ctx else ""
    create_kwargs = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=_build_system_blocks(skeleton),
        messages=[{"role": "user", "content": f"{ctx_prefix}Question:\n{prompt}"}],
    )
    if effort == "max":
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}

    # See chat() for rationale — snapshot to detect agent edits in OAUTH mode.
    pre_modified = _git_modified_snapshot() if AUTH_MODE == "OAUTH" else set()

    try:
        for attempt in range(3):
            reply, usage, was_streamed = _make_streaming_api_call(**create_kwargs)

            edits = parse_edit_blocks(reply)

            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    cost = calculate_cost(usage)
                    _accumulate_usage(usage)
                    with state.lock:
                        state.session_cost += cost

                    print(f"  {_T_ERR}  Syntax errors — correcting (attempt {attempt+1}/3)…", file=sys.stderr)
                    for err in lint_errs:
                        print(f"    {err.strip().replace(chr(10), ' ')}", file=sys.stderr)

                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."
                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})

                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]:
                                b.pop("cache_control", None)
                    continue

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation and not was_streamed:
                print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)
                print(explanation)
                sys.stdout.flush()
            if edits:
                apply_edits(edits, explanation=explanation, auto_apply=auto_apply)
            elif AUTH_MODE == "OAUTH":
                # See chat() — use the prompt, not the agent's narration.
                post_modified = _git_modified_snapshot()
                agent_edits = post_modified - pre_modified
                if agent_edits:
                    _commit_agent_edits(agent_edits, prompt, auto_apply=auto_apply)

            cost = calculate_cost(usage)
            _accumulate_usage(usage)
            with state.lock:
                state.session_cost += cost
            print_stats(usage, label="Stats", file=sys.stderr)
            break

    except Exception as e:
        raise SystemExit(f"{_T_ERR} API call failed: {e}")
