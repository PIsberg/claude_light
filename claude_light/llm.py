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
    _T_ROUTE, _T_SYS, _T_ERR, _T_RAG, _T_CACHE,
    _ANSI_GREEN, _ANSI_CYAN, _ANSI_YELLOW, _ANSI_MAGENTA, _ANSI_BOLD, _ANSI_DIM, _ANSI_RESET, _ANSI_RED,
    print_stats, _Spinner, calculate_cost, _print_reply
)
import claude_light.state as state
from claude_light.skeleton import build_skeleton, _refresh_tree_only, _assemble_skeleton, _refresh_single_md
from claude_light.indexer import index_files
from claude_light.retrieval import retrieve
from claude_light.editor import parse_edit_blocks, apply_edits
from claude_light.retry import retry_with_backoff

class ClaudeNotLoggedIn(Exception):
    """Raised when the official Claude CLI is not logged in."""
    pass
from claude_light.streaming import stream_chat_response, accumulate_usage_from_dict, calculate_usage_cost

# Active Mode Announcement
if AUTH_MODE == "API_KEY":
    print(f"{_T_SYS} Auth: {_ANSI_BOLD}API Key{_ANSI_RESET} (Source: {API_KEY_SOURCE})", file=sys.stderr)
elif AUTH_MODE == "OAUTH" and config.AUTH_TOKEN:
    print(f"{_T_SYS} Auth: {_ANSI_BOLD}OAuth/Pro{_ANSI_RESET} (Source: {API_KEY_SOURCE})", file=sys.stderr)
else:
    print(f"\n{_T_ERR} No authentication found.", file=sys.stderr)
    print(f"{_T_SYS} To use your Claude Pro subscription, please authorize with the official CLI:", file=sys.stderr)
    print(f"      {_ANSI_CYAN}claude auth login{_ANSI_RESET}\n", file=sys.stderr)
    print(f"{_T_SYS} Alternatively, set {_ANSI_BOLD}ANTHROPIC_API_KEY{_ANSI_RESET} in your environment.", file=sys.stderr)
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
        "list", "show", "where", "what is", "what are", "how many",
        "print", "display", "tell me", "which file", "which files",
        "find", "locate", "count", "search", "grep"
    }

    # 3. Calculate Base Score
    score = 0
    score += sum(4.0 for s in _ARCH_SIGNALS  if s in q)
    score += sum(2.0 for s in _LOGIC_SIGNALS if s in q)
    score += sum(0.5 for s in _INFRA_SIGNALS if s in q)
    
    # 4. Contextual Multipliers
    # Mentioning a file path (e.g. main.py, lib/utils.py) suggests technical intent
    if re.search(r'[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{2,4}', q):
        score += 3.0
    
    # Large word count implies complexity
    if word_count > 30:
        score += 3.0
    
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
    model_short = model.split("-")[1]
    print(
        f"{_T_ROUTE} effort={color}{effort}{_ANSI_RESET}  "
        f"model={_ANSI_BOLD}{model_short}{_ANSI_RESET}",
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
        print(f"{_T_SYS} Compressing {SUMMARIZE_BATCH} old turns...", end="", flush=True)
        summary_text, usage = _summarize_turns(to_summarize)
        _accumulate_usage(usage)
        with state.lock:
            state.session_cost += calculate_cost(usage)
        summary_pair = [
            {"role": "user",      "content": f"[Summary of earlier conversation]\n{summary_text}"},
            {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
        ]
        state.conversation_history = summary_pair + remaining
        print(f" done ({usage.input_tokens}→{usage.output_tokens} tok)")
    except Exception as e:
        print(f"\n{_T_ERR} History compression failed ({e}) — truncating instead.")
        state.conversation_history = state.conversation_history[-(MAX_HISTORY_TURNS * 2):]

def _build_system_blocks(skeleton):
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton, "cache_control": {"type": "ephemeral"}},
    ]
    return blocks


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
    
    try:
        # Prepare the environment
        env = os.environ.copy()

        # Inject the automation token on all platforms.
        # On Linux/macOS: primary auth mechanism (no Credential Manager).
        # On Windows: fallback — CLI prefers the Credential Manager, but will use
        # this env var if no Credential Manager entry exists.
        if config.AUTH_TOKEN and config.AUTH_TOKEN.startswith("sk-ant-oat01"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = config.AUTH_TOKEN

        # On Windows, ensure critical path variables are present for the Bun-based CLI to find its profile
        if current_os == 'nt':
            # Ensure all common home/profile variables are present
            home_dir = os.path.expanduser('~')
            env.setdefault('USERPROFILE', home_dir)
            env.setdefault('APPDATA', os.environ.get('APPDATA', str(Path(home_dir) / "AppData" / "Roaming")))
            env.setdefault('LOCALAPPDATA', os.environ.get('LOCALAPPDATA', str(Path(home_dir) / "AppData" / "Local")))
            env.setdefault('HOMEDRIVE', os.environ.get('HOMEDRIVE', 'C:'))
            env.setdefault('HOMEPATH', os.environ.get('HOMEPATH', home_dir.split(':', 1)[1] if ':' in home_dir else home_dir))
            
            # Note: We NO LONGER set CLAUDE_CONFIG_DIR explicitly, as it can confuse the Bun binary
            # into looking in a non-standard location for its session keyring on some Windows setups.
            pass
        
        # Windows command line length limit is 8,191 characters. 
        # If the prompt is too long, we use the CLI's '@' feature to load context from a file.
        if current_os == 'nt' and len(prompt) > 8000:
            fd, path = tempfile.mkstemp(suffix=".md", prefix="claude_light_ctx_")
            temp_context_file = path
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # Use @file syntax to bypass cmd.exe limits
            # Ensure path uses forward slashes to avoid CLI escaping issues with Windows backslashes
            safe_path = Path(temp_context_file).as_posix()
            cmd_prompt = f"Please follow the instructions and use the codebase context provided in @{safe_path}"
        else:
            cmd_prompt = prompt

        # --bare disables the CLI system prompt for a cleaner response.
        # On Windows it also breaks Credential Manager access, so we omit it there.
        # --tools "" strips built-in tool definitions from the context (saves ~5-10K tokens).
        command = [claude_bin]
        if current_os != 'nt':
            command.append("--bare")
        command += ["-p", cmd_prompt, "--output-format", "json", "--tools", ""]
        if model:
            command += ["--model", model]
        if extra_flags:
            command += extra_flags
        
        # 3. Execute
        # Use shell=True on Windows to ensure we're in a proper terminal environment that the Bun binary expects
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace',
            shell=(current_os == 'nt')
        )
        
        if result.returncode != 0:
            error_output = result.stderr or result.stdout
            if "Not logged in" in error_output or "Invalid API key" in error_output:
                raise ClaudeNotLoggedIn("Claude CLI: Not logged in. Please run 'claude auth login' in your terminal.")
            raise RuntimeError(f"Claude CLI error (code {result.returncode}): {error_output}")
            
        # 4. Parse Output
        try:
            data = json.loads(result.stdout)
            # In 'json' mode, the CLI returns an object with results
            response_text = data.get("result", "")
            
            # Extract usage from the CLI's "usage" block (not "stats").
            usage_data = data.get("usage", {})
            class UsageObj:
                pass
            usage = UsageObj()
            usage.input_tokens = usage_data.get("input_tokens", 0)
            usage.output_tokens = usage_data.get("output_tokens", 0)
            usage.cache_creation_input_tokens = usage_data.get("cache_creation_input_tokens", 0)
            usage.cache_read_input_tokens = usage_data.get("cache_read_input_tokens", 0)

            session_id = data.get("session_id")
            return response_text, usage, session_id

        except json.JSONDecodeError:
            return result.stdout.strip(), None, None

    finally:
        # Cleanup temporary file if created
        if temp_context_file and os.path.exists(temp_context_file):
            try:
                os.remove(temp_context_file)
            except OSError:
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

        if state.cli_session_id:
            # Turn 2+: CLI already has the skeleton and history — only send the
            # latest user message plus any freshly retrieved context that was
            # injected as a second system block.
            last_user = next(
                (m for m in reversed(messages) if m["role"] == "user"), None
            )
            user_content = last_user["content"] if last_user else ""
            if isinstance(user_content, list):
                user_content = " ".join(
                    b.get("text", "") for b in user_content if b.get("type") == "text"
                )
            # If there is a retrieved-context block (block index 1+), prepend it
            # so the CLI sees fresh RAG results even without re-sending the full skeleton.
            rag_blocks = [b for b in system_blocks[1:] if b.get("text")]
            if rag_blocks:
                rag_text = "\n".join(b["text"] for b in rag_blocks)
                prompt = f"[Retrieved context for this query]\n{rag_text}\n\n{user_content}"
            else:
                prompt = user_content
            extra_flags = ["--resume", state.cli_session_id]
        else:
            # Turn 1: send the full system prompt + history as a flat string.
            system_text = "\n".join(b.get("text", "") for b in system_blocks)
            history_text = ""
            for msg in messages[:-1]:          # everything except the last user msg
                role = msg["role"].capitalize()
                content = msg["content"]
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                history_text += f"--- {role} ---\n{content}\n"
            last_user = messages[-1]["content"] if messages else ""
            if isinstance(last_user, list):
                last_user = " ".join(
                    b.get("text", "") for b in last_user if b.get("type") == "text"
                )
            prompt = (
                f"INSTRUCTIONS:\n{system_text}\n\n"
                + (f"CONVERSATION HISTORY:\n{history_text}\n" if history_text else "")
                + f"USER: {last_user}"
            )
            extra_flags = ["--system-prompt", system_text]

        print(f"{_T_SYS} Processing via Claude CLI...", end="", flush=True)
        try:
            reply_text, usage, session_id = _make_cli_subprocess_call(
                prompt, model=model, extra_flags=extra_flags
            )
            if session_id:
                state.cli_session_id = session_id
            print(" done.")
            return reply_text, usage
        except ClaudeNotLoggedIn:
            print(" failed (auth).")
            raise
        except Exception as e:
            print(f" failed ({e}).")
            raise

    # API Key mode — use the anthropic client directly
    if ENABLE_STREAMING and hasattr(client.messages, 'stream'):
        try:
            # Use streaming
            reply_text, usage_dict = stream_chat_response(client, **create_kwargs)
            # Convert usage dict to response.usage-like object for compatibility
            class UsageObj:
                pass
            usage = UsageObj()
            usage.input_tokens = usage_dict.get('input_tokens', 0)
            usage.output_tokens = usage_dict.get('output_tokens', 0)
            usage.cache_creation_input_tokens = usage_dict.get('cache_creation_tokens', 0)
            usage.cache_read_input_tokens = usage_dict.get('cache_read_tokens', 0)
            return reply_text, usage
        except (AttributeError, NotImplementedError):
            # Fallback to non-streaming if stream not available
            @retry_with_backoff
            def _call_api():
                return client.messages.create(**create_kwargs)
            response = _call_api()
            return _extract_text(response.content), response.usage
    else:
        # Non-streaming path
        @retry_with_backoff
        def _call_api():
            return client.messages.create(**create_kwargs)
        response = _call_api()
        return _extract_text(response.content), response.usage

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
            print(f"\n{_ANSI_DIM}{'═'*56}{_ANSI_RESET}")
            print_stats(response.usage, label="Cache")
            print(f"{_ANSI_DIM}{'═'*56}{_ANSI_RESET}\n")
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

    if hits:
        names      = ", ".join(_chunk_label(p) for p, _ in hits)
        scores_str = "  ".join(f"{_chunk_label(p)} {s:.2f}" for p, s in hits)
        print(f"{_T_RAG} {_ANSI_CYAN}{names}{_ANSI_RESET}")
        print(f"      {_ANSI_DIM}{scores_str}{_ANSI_RESET}")

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

    try:
        for attempt in range(3):
            # Use streaming or non-streaming API call
            reply, usage = _make_streaming_api_call(**create_kwargs)
            
            edits = parse_edit_blocks(reply)
            
            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    cost = calculate_cost(usage)
                    _accumulate_usage(usage)
                    with state.lock:
                        state.session_cost += cost

                    print(f"\n{_T_ERR} {_ANSI_RED}Auto-detected errors in AI's code:{_ANSI_RESET}")
                    for err in lint_errs:
                        err_clean = err.strip().replace('\n', ' ')
                        print(f"  {err_clean}")
                    print(f"[{_ANSI_CYAN}Auto-correction{_ANSI_RESET}] Requesting fix from Claude (attempt {attempt+1}/3)...")
                    
                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."
                    
                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})
                    
                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]: b.pop("cache_control", None)
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
            if explanation:
                _print_reply(explanation)
            if edits:
                apply_edits(edits, explanation=explanation, auto_apply=auto_apply)

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
    
    from claude_light.indexer import _chunk_label
    import re
    from claude_light.editor import _ANY_BLOCK
    
    if hits:
        names = ", ".join(_chunk_label(p) for p, _ in hits)
        print(f"{_T_RAG} {_ANSI_CYAN}{names}{_ANSI_RESET}", file=sys.stderr)

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
    try:
        for attempt in range(3):
            # Use streaming or non-streaming API call
            reply, usage = _make_streaming_api_call(**create_kwargs)
            
            edits = parse_edit_blocks(reply)

            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    cost = calculate_cost(usage)
                    _accumulate_usage(usage)
                    with state.lock:
                        state.session_cost += cost

                    print(f"\n{_T_ERR} Auto-detected errors:", file=sys.stderr)
                    for err in lint_errs:
                        err_clean = err.strip().replace('\n', ' ')
                        print(f"  {err_clean}", file=sys.stderr)
                    print(f"[Auto-correction] Requesting fix from Claude (attempt {attempt+1}/3)...", file=sys.stderr)
                    
                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."
                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})
                    
                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]: b.pop("cache_control", None)
                    continue

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation:
                print(explanation)
            if edits:
                apply_edits(edits, explanation=explanation, auto_apply=auto_apply)

            cost = calculate_cost(usage)
            _accumulate_usage(usage)
            with state.lock:
                state.session_cost += cost
            print_stats(usage, label="Stats", file=sys.stderr)
            break

    except Exception as e:
        raise SystemExit(f"{_T_ERR} API call failed: {e}")
