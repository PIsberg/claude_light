import sys
import os

# Fix for Windows: prevent Intel Fortran runtime from hijacking Ctrl+C
if os.name == 'nt':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
    # Ensure stdout/stderr use UTF-8 so Unicode symbols render correctly in
    # Windows Terminal and modern consoles (graceful no-op on older hosts).
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import threading
import signal
from pathlib import Path
from watchdog.observers import Observer

import claude_light.state as state
from claude_light.config import MODEL, CACHE_DIR, HEARTBEAT_SECS, CACHE_TTL_SECS, AUTH_MODE, ECONOMY_MODE, API_KEY_SOURCE
from claude_light.ui import (
    _ANSI_BOLD, _ANSI_RESET, _ANSI_CYAN, _ANSI_DIM, _ANSI_GREEN, _ANSI_RED, _T_SYS, _T_ERR,
    _UNICODE, print_session_summary, print_session_footer, print_startup_header, ToolExecutionManager,
)
from claude_light.llm import full_refresh, chat, one_shot, warm_cache, compact_history, ClaudeNotLoggedIn
from claude_light.indexer import SourceHandler
from claude_light.compressor import start_background_load as _start_llmlingua_load

# The interactive commands, in the order /help and the "/" menu show them.
_COMMANDS = [
    ("/help",       "show commands and keys"),
    ("/clear",      "start a new conversation"),
    ("/compact",    "summarise the conversation so far, to send fewer tokens"),
    ("/cost",       "token usage and cost for this session"),
    ("/run <cmd>",  "run a shell command and ask Claude about its output"),
    ("/undo",       "revert the last commit Claude made"),
    ("/why",        "explain the last query's routing and retrieval"),
    ("exit",        "quit"),
]
_KEYS = [
    ("enter",       "send"),
    ("alt+enter",   "new line"),
    ("ctrl+c",      "cancel the running turn; at the prompt, quit"),
    ("↑ / ↓" if _UNICODE else "up / down", "previous prompts"),
]

try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.history import FileHistory as _FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _AutoSuggest
    from prompt_toolkit.completion import Completer as _Completer, Completion as _Completion
    from prompt_toolkit.key_binding import KeyBindings as _KeyBindings
    from prompt_toolkit.styles import Style as _PTStyle
    _PROMPTTK_AVAILABLE = True

    class _SlashCompleter(_Completer):
        """The "/" menu: command names with what they do, while typing one."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for cmd, desc in _COMMANDS:
                name = cmd.split()[0]
                if name.startswith(text):
                    yield _Completion(name, start_position=-len(text), display_meta=desc)
except ImportError:
    _PROMPTTK_AVAILABLE = False


def _print_help():
    try:
        from rich.table import Table
        from rich.console import Console
    except ImportError:
        Table = None
    if Table is not None:
        tbl = Table(box=None, padding=(0, 2), show_header=False)
        tbl.add_column(style="cyan", no_wrap=True)
        tbl.add_column(style="dim")
        for cmd, desc in _COMMANDS:
            tbl.add_row(cmd, desc)
        tbl.add_row("", "")
        for key, desc in _KEYS:
            tbl.add_row(key, desc)
        con = Console(file=sys.stdout)
        con.print()
        con.print(tbl)
        return
    print()
    for cmd, desc in _COMMANDS + [("", "")] + _KEYS:
        print(f"  {cmd:<14}{_ANSI_DIM}{desc}{_ANSI_RESET}")


def heartbeat():
    if AUTH_MODE == "OAUTH":
        return
    while not state.stop_event.is_set():
        state.stop_event.wait(HEARTBEAT_SECS)
        with state.lock:
            import time
            idle = time.time() - state.last_interaction
        if idle > CACHE_TTL_SECS and not state.stop_event.is_set():
            warm_cache()


def _setup_signal_handlers():
    """Ctrl+C interrupts the current turn; SIGTERM exits.

    SIGINT keeps Python's default (KeyboardInterrupt) so chat() can abandon
    the turn, kill the CLI agent and return to the prompt; at the prompt it
    exits the loop. start_chat()'s finally block sets stop_event on the way
    out, so neither signal needs to touch it.
    """
    def _terminate(signum, frame):
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, _terminate)
    except Exception:
        # Signal handling might not be available on all platforms
        pass


def _index_ready_for_query() -> bool:
    """Decide whether this query can use the RAG index, waiting if it must.

    A cold index build embeds every chunk on CPU (314 s for a 64-file repo
    in testing). In OAuth mode the Claude CLI agent can Read/Grep the code
    itself, so the query goes ahead without retrieval. In API-key mode the
    model has no tools and would answer blind, so the query waits.
    """
    if state.indexing_done.is_set():
        return True
    if AUTH_MODE == "OAUTH":
        return False
    print(f"  {_T_SYS}  Indexing…", flush=True)
    state.indexing_done.wait()
    return True


def start_chat(auto_apply=False):
    print_startup_header(MODEL, API_KEY_SOURCE)
    _setup_signal_handlers()

    # Kick off indexing in background so the prompt appears immediately.
    # Any query submitted before indexing completes will wait at that point.
    state.indexing_done.clear()

    def _bg_refresh():
        try:
            full_refresh(quiet=True)
        finally:
            state.indexing_done.set()

    _refresh_thread = threading.Thread(target=_bg_refresh, daemon=True, name="claude-light-refresh")
    _refresh_thread.start()
    # If threading.Thread is mocked (test environment), is_alive() returns a
    # MagicMock rather than bool. Run _bg_refresh() inline so indexing_done
    # gets set and the query loop doesn't block indefinitely.
    if not isinstance(_refresh_thread.is_alive(), bool):
        _bg_refresh()

    # Watchdog can start immediately — it doesn't need the index to be ready.
    observer = None
    try:
        observer = Observer()
        observer.schedule(SourceHandler(), path=".", recursive=True)
        observer.start()
    except Exception as e:
        print(f"{_T_ERR} Failed to start file watcher: {e}", file=sys.stderr)

    if AUTH_MODE == "API_KEY":
        threading.Thread(target=heartbeat, daemon=True).start()

    if _PROMPTTK_AVAILABLE:
        from prompt_toolkit import HTML
        from prompt_toolkit.formatted_text import to_formatted_text

        def _get_status_html():
            project = os.path.basename(os.getcwd())
            if not state.indexing_done.is_set():
                left = f' <ansicyan>◆</ansicyan> {project}  ·  <ansiyellow>indexing…</ansiyellow>'
            else:
                with state.lock:
                    total_in = state.session_tokens["input"] + state.session_tokens["cache_write"] + state.session_tokens["cache_read"]
                    saved = state.session_tokens["cache_read"]
                    cost = state.session_cost
                ratio = (saved / total_in * 100) if total_in > 0 else 0.0
                cost_label = "cost" if ECONOMY_MODE == "USD" else "API equiv."
                left = (
                    f' <ansicyan>◆</ansicyan> {project}'
                    f'  ·  {total_in:,} tok  ·  {ratio:.0f}% cached'
                    f'  ·  {cost_label} ${cost:.4f}'
                )
            return left + '  ·  / for commands  ·  alt+enter new line'

        def get_status_bar():
            return HTML(_get_status_html())

        bindings = _KeyBindings()

        @bindings.add("escape", "enter")  # alt+enter on most terminals
        def _newline(event):
            event.current_buffer.insert_text("\n")

        CACHE_DIR.mkdir(exist_ok=True)
        _session = _PromptSession(
            history=_FileHistory(str(CACHE_DIR / "history.txt")),
            auto_suggest=_AutoSuggest(),
            completer=_SlashCompleter(),
            complete_while_typing=True,
            key_bindings=bindings,
            bottom_toolbar=get_status_bar,
            refresh_interval=1.0,  # the toolbar shows when indexing ends
            prompt_continuation="  ",
            style=_PTStyle.from_dict({
                "bottom-toolbar": "noreverse #888888",
                "prompt": "ansicyan bold",
                "completion-menu.completion": "bg:#303030 #dddddd",
                "completion-menu.completion.current": "bg:#005f87 #ffffff",
                "completion-menu.meta.completion": "bg:#303030 #888888",
                "completion-menu.meta.completion.current": "bg:#005f87 #dddddd",
            }),
        )
        _prompt = [("", "\n"), ("class:prompt", "❯ " if _UNICODE else "> ")]

        def _get_input():
            return _session.prompt(_prompt).strip()
    else:
        def _get_input():
            with state.lock:
                total_in = state.session_tokens["input"] + state.session_tokens["cache_write"] + state.session_tokens["cache_read"]
                saved = state.session_tokens["cache_read"]
                cost = state.session_cost
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            cost_label = "Cost" if ECONOMY_MODE == "USD" else "API equiv."
            stat = f"  {_ANSI_DIM}◆  {os.path.basename(os.getcwd())}  ·  {total_in:,} tok  ({ratio:.1f}% cached)  ·  {cost_label} ${cost:.4f}{_ANSI_RESET}"
            print(f"\n{stat}")
            return input("> ").strip()

    try:
        while True:
            try:
                query = _get_input()
            except (KeyboardInterrupt, EOFError):
                break

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                break
            if query == "/clear":
                state.conversation_history.clear()
                state.cli_session_id = None
                print(f"  {_T_SYS}  Conversation cleared.")
                continue
            if query == "/compact":
                try:
                    with ToolExecutionManager("Summarising the conversation"):
                        turns = compact_history()
                except KeyboardInterrupt:
                    print(f"  {_T_SYS}  Cancelled; the conversation is unchanged.")
                    continue
                except Exception as e:
                    print(f"  {_T_ERR}  Could not summarise ({e}); the conversation is unchanged.")
                    continue
                if turns:
                    print(f"  {_T_SYS}  {turns} turn{'' if turns == 1 else 's'} replaced by a summary.")
                else:
                    print(f"  {_T_SYS}  Nothing to compact yet.")
                continue
            if query == "/cost":
                print_session_summary()
                continue
            if query == "/help":
                _print_help()
                continue
            if query == "/why":
                from claude_light.retrieval import explain_last_retrieval
                print(f"\n{explain_last_retrieval()}\n")
                continue
            if query == "/run" or query.startswith("/run "):
                cmd = query[4:].strip()
                if not cmd:
                    print(f"{_T_ERR} Usage: /run <shell command>\n")
                    continue
                from claude_light.executor import _run_command
                transcript = _run_command(cmd)
                chat(f"I ran the following command and got this output. Please help me understand or fix any issues:\n\n```\n{transcript}\n```",
                     auto_apply=auto_apply, use_index=_index_ready_for_query())
                continue
            if query == "/undo":
                from claude_light import git_manager
                if not git_manager.is_git_repo():
                    print(f"{_T_ERR} Not in a git repository. Cannot undo.\n")
                    continue
                
                last_commit = git_manager.get_last_commit_message()
                if not last_commit:
                    print(f"{_T_ERR} No commits to undo.\n")
                    continue
                
                msg_preview = last_commit.strip().split('\n')[0][:60]
                print(f"  {_T_SYS}  Last commit: {_ANSI_DIM}{msg_preview}{_ANSI_RESET}")
                print(f"  Undo? [{_ANSI_GREEN}y{_ANSI_RESET}/{_ANSI_RED}n{_ANSI_RESET}] ", end="", flush=True)
                try:
                    confirm = input().strip().lower() in ("y", "yes")
                except (KeyboardInterrupt, EOFError):
                    confirm = False
                
                if confirm:
                    if git_manager.undo_last_commit():
                        print(f"  {_T_SYS}  Reverted.\n")
                    else:
                        print(f"  {_T_ERR}  Undo failed.\n")
                else:
                    print(f"  {_T_SYS}  Cancelled.\n")
                continue

            if query.startswith("/") and " " not in query:
                # A mistyped command would otherwise go to Claude as a prompt.
                print(f"  {_T_ERR}  Unknown command {query}. /help lists them.")
                continue

            # Standard Chat Query
            use_index = _index_ready_for_query()
            try:
                chat(query, auto_apply=auto_apply, use_index=use_index)
            except ClaudeNotLoggedIn:
                print(f"\n  {_T_ERR}  Claude CLI is not logged in.")
                print(f"  {_T_SYS}  Set up a long-lived Automation Token? [{_ANSI_GREEN}y{_ANSI_RESET}/{_ANSI_RED}n{_ANSI_RESET}] ", end="", flush=True)
                try:
                    choice = input().strip().lower()
                except (KeyboardInterrupt, EOFError):
                    choice = "n"
                
                if choice in ("y", "yes"):
                    import subprocess
                    import shutil
                    from pathlib import Path
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
                                from claude_light import config
                                config.API_KEY, config.AUTH_MODE, config.API_KEY_SOURCE, config.AUTH_TOKEN = config._resolve_api_key()
                                full_refresh()
                                chat(query, auto_apply=auto_apply, use_index=use_index)
                            else:
                                print(f"  {_T_ERR}  Token must start with 'sk-ant-oat01'.")
                        except (KeyboardInterrupt, EOFError):
                            print(f"\n  {_T_SYS}  Cancelled.")
                    else:
                        print(f"  {_T_ERR}  Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
                else:
                    print(f"  {_T_SYS}  Proceeding without auth. Requests will likely fail.")
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=5)  # Wait max 5 seconds for observer to stop
            except Exception as e:
                print(f"{_T_ERR} Error stopping file watcher: {e}", file=sys.stderr)
        print_session_footer()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Light RAG Chat")
    parser.add_argument("--test-mode", choices=["small", "medium", "large", "extra-large"],
                        help="Run in test mode with a synthetic codebase and mocked API.")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically apply changes without asking.")
    parser.add_argument("--allow-command", action="append", default=[], metavar="PREFIX",
                        help="Let the Claude CLI agent (subscription mode) run shell commands starting "
                             "with PREFIX, e.g. \"python -m pytest\". Repeatable. Also "
                             "CLAUDE_LIGHT_ALLOW_COMMANDS=\"a; b\".")
    parser.add_argument("query", nargs="*", help="Optional query for one-shot mode.")
    args, unknown = parser.parse_known_args()

    import claude_light.config as _config
    for prefix in args.allow_command:
        if prefix.strip() and prefix.strip() not in _config.ALLOWED_COMMANDS:
            _config.ALLOWED_COMMANDS.append(prefix.strip())

    if args.test_mode:
        from tests.utilities.test_mocks import MockManager
        manager = MockManager(args.test_mode)
        manager.start()

    # Kick off LLMLingua-2 model load in background as early as possible so
    # indexing + skeleton-building run in parallel with the model load. No-op
    # if disabled or if the package isn't installed.
    _start_llmlingua_load()

    query_str = " ".join(args.query).strip()
    if not query_str and sys.stdin.isatty():
        start_chat(auto_apply=args.yes)
        return
    try:
        one_shot(query_str or sys.stdin.read().strip(), auto_apply=args.yes)
    except KeyboardInterrupt:
        print(f"\n  {_T_ERR}  Interrupted.", file=sys.stderr)
        raise SystemExit(130)

if __name__ == "__main__":
    main()
