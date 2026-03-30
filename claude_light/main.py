import sys
import os

# Fix for Windows: prevent Intel Fortran runtime from hijacking Ctrl+C
if os.name == 'nt':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import threading
import signal
from pathlib import Path
from watchdog.observers import Observer

import claude_light.state as state
from claude_light.config import MODEL, CACHE_DIR, HEARTBEAT_SECS, CACHE_TTL_SECS, AUTH_MODE, ECONOMY_MODE, API_KEY_SOURCE
from claude_light.ui import _ANSI_BOLD, _ANSI_RESET, _ANSI_CYAN, _ANSI_DIM, _ANSI_GREEN, _ANSI_RED, _T_SYS, _T_ERR, print_session_summary
from claude_light.llm import full_refresh, chat, one_shot, warm_cache, ClaudeNotLoggedIn
from claude_light.indexer import SourceHandler

try:
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.history import FileHistory as _FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _AutoSuggest
    from prompt_toolkit.completion import WordCompleter as _WordCompleter
    _PROMPTTK_AVAILABLE = True
except ImportError:
    _PROMPTTK_AVAILABLE = False


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
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        # Signal to stop everything gracefully
        state.stop_event.set()
    
    # Handle SIGINT (Ctrl+C) and SIGTERM
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except Exception:
        # Signal handling might not be available on all platforms
        pass


def start_chat(auto_apply=False):
    full_refresh()
    _setup_signal_handlers()  # Setup signal handlers before starting threads

    observer = None
    try:
        observer = Observer()
        observer.schedule(SourceHandler(), path=".", recursive=True)
        observer.start()
    except Exception as e:
        print(f"{_T_ERR} Failed to start file watcher: {e}", file=sys.stderr)
        # Continue without file watching if it fails
    
    if AUTH_MODE == "API_KEY":
        threading.Thread(target=heartbeat, daemon=True).start()

    print(f"│ {_ANSI_CYAN}{MODEL}{_ANSI_RESET}  |  Auth: {_ANSI_BOLD}{API_KEY_SOURCE}{_ANSI_RESET}  |  RAG: {_ANSI_BOLD}top-{state.TOP_K}{_ANSI_RESET}")
    print(f"{_ANSI_DIM}Commands: /compact  /cost  /run <cmd>  /help  exit{_ANSI_RESET}\n")

    if _PROMPTTK_AVAILABLE:
        from prompt_toolkit import HTML

        def get_status_text():
            with state.lock:
                total_in = state.session_tokens["input"] + state.session_tokens["cache_write"] + state.session_tokens["cache_read"]
                saved = state.session_tokens["cache_read"]
                cost = state.session_cost
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            
            text = (
                f' <b>Repo:</b> <ansicyan>{os.path.basename(os.getcwd())}</ansicyan>  |  '
                f'<b>Tokens:</b> {total_in:,} '
                f'(<ansigreen>{saved:,}</ansigreen> saved, <ansigreen>{ratio:.1f}%</ansigreen>)'
            )
            cost_label = "Cost" if ECONOMY_MODE == "USD" else "API equiv."
            text += f'  |  <b>{cost_label}:</b> <ansiyellow>${cost:.4f}</ansiyellow>'
            return text

        def get_status_bar():
            return HTML(get_status_text())

        CACHE_DIR.mkdir(exist_ok=True)
        _slash_completer = _WordCompleter(
            ["/compact", "/clear", "/cost", "/help", "/run", "/undo", "exit", "quit"],
            sentence=True,
        )
        _session = _PromptSession(
            history=_FileHistory(str(CACHE_DIR / "history.txt")),
            auto_suggest=_AutoSuggest(),
            completer=_slash_completer,
            complete_while_typing=False,
            # bottom_toolbar is unstable on Windows console host, switching to in-line stats
            bottom_toolbar=None if os.name == 'nt' else get_status_bar
        )
        def _get_input():
            if os.name == 'nt':
                # Manually print status line before prompt on Windows to avoid ghosting.
                # Remove HTML tags for plain text print.
                import re
                raw_text = get_status_text()
                clean_text = re.sub(r'<[^>]+>', '', raw_text)
                print(f"\n{clean_text}")
            return _session.prompt("> ").strip()
    else:
        def _get_input():
            with state.lock:
                total_in = state.session_tokens["input"] + state.session_tokens["cache_write"] + state.session_tokens["cache_read"]
                saved = state.session_tokens["cache_read"]
                cost = state.session_cost
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            cost_label = "Cost" if ECONOMY_MODE == "USD" else "API equiv."
            stat_line = f"\n[{os.path.basename(os.getcwd())}] Tokens: {total_in:,} ({saved:,} saved, {ratio:.1f}%) | {cost_label}: ${cost:.4f}"
            print(stat_line)
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
            if query in ("/clear", "/compact"):
                state.conversation_history.clear()
                state.cli_session_id = None  # force a fresh CLI session
                print(f"{_T_SYS} Conversation history compacted.\n")
                continue
            if query == "/cost":
                print_session_summary()
                continue
            if query == "/help":
                print(
                    "  /compact      — reset conversation history\n"
                    "  /cost         — show session spend so far\n"
                    "  /run <cmd>    — run a shell command and feed output to Claude\n"
                    "  /undo         — undo the last commit (revert AI changes)\n"
                    "  exit          — quit\n"
                )
                continue
            if query == "/run" or query.startswith("/run "):
                cmd = query[4:].strip()
                if not cmd:
                    print(f"{_T_ERR} Usage: /run <shell command>\n")
                    continue
                from claude_light.executor import _run_command
                transcript = _run_command(cmd)
                chat(f"I ran the following command and got this output. Please help me understand or fix any issues:\n\n```\n{transcript}\n```", auto_apply=auto_apply)
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
                print(f"Last commit: {msg_preview}")
                print(f"Undo this commit? [{_ANSI_GREEN}y{_ANSI_RESET}/{_ANSI_RED}n{_ANSI_RESET}] ", end="", flush=True)
                try:
                    confirm = input().strip().lower() in ("y", "yes")
                except (KeyboardInterrupt, EOFError):
                    confirm = False
                
                if confirm:
                    if git_manager.undo_last_commit():
                        print(f"{_T_SYS} Undo complete.\n")
                    else:
                        print(f"{_T_ERR} Failed to undo.\n")
                else:
                    print(f"{_T_SYS} Cancelled.\n")
                continue

            # Standard Chat Query
            try:
                chat(query, auto_apply=auto_apply)
            except ClaudeNotLoggedIn:
                print(f"\n{_T_ERR} Claude CLI is not logged in.")
                print(f"{_T_SYS} This often happens on Windows when the CLI keyring is restricted.")
                print(f"[?] Perform a ONE-TIME setup of a long-lived 'Automation Token'? [y/n] ", end="", flush=True)
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
                        print(f"{_T_SYS} Launching 'claude setup-token'...")
                        print(f"{_ANSI_DIM}Follow the browser instructions. Once finished, the CLI will display your token.{_ANSI_RESET}")
                        subprocess.run([claude_bin, "setup-token"])
                        
                        print(f"\n{_ANSI_BOLD}Paste your new Automation Token here:{_ANSI_RESET} ", end="", flush=True)
                        try:
                            new_token = input().strip()
                            if new_token.startswith("sk-ant-oat01"):
                                token_path = Path.home() / ".claude_light_automation_token"
                                token_path.write_text(new_token, encoding="utf-8")
                                print(f"{_T_SYS} Token saved to {token_path}")
                                
                                # Reload config/state and retry
                                from claude_light import config
                                config.API_KEY, config.AUTH_MODE, config.API_KEY_SOURCE, config.AUTH_TOKEN = config._resolve_api_key()
                                full_refresh()
                                chat(query, auto_apply=auto_apply)
                            else:
                                print(f"{_T_ERR} Invalid token format. It should start with 'sk-ant-oat01'.")
                        except (KeyboardInterrupt, EOFError):
                            print(f"\n{_T_SYS} Cancelled.")
                    else:
                        print(f"{_T_ERR} Claude CLI binary not found. Please install first.")
                else:
                    print(f"{_T_SYS} Proceeding without login. Results may fail.")
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
        print_session_summary()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Light RAG Chat")
    parser.add_argument("--test-mode", choices=["small", "medium", "large", "extra-large"],
                        help="Run in test mode with a synthetic codebase and mocked API.")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically apply changes without asking.")
    parser.add_argument("query", nargs="*", help="Optional query for one-shot mode.")
    args, unknown = parser.parse_known_args()

    if args.test_mode:
        from tests.utilities.test_mocks import MockManager
        manager = MockManager(args.test_mode)
        manager.start()

    query_str = " ".join(args.query).strip()
    if query_str:
        one_shot(query_str, auto_apply=args.yes)
    elif not sys.stdin.isatty():
        one_shot(sys.stdin.read().strip(), auto_apply=args.yes)
    else:
        start_chat(auto_apply=args.yes)

if __name__ == "__main__":
    main()
