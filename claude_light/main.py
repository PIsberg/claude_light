import sys
import os
import threading
from pathlib import Path
from watchdog.observers import Observer

import claude_light.state as state
from claude_light.config import MODEL, CACHE_DIR, HEARTBEAT_SECS, CACHE_TTL_SECS
from claude_light.ui import _ANSI_BOLD, _ANSI_RESET, _ANSI_CYAN, _ANSI_DIM, _T_SYS, _T_ERR, print_session_summary
from claude_light.llm import full_refresh, chat, one_shot, warm_cache
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
    while not state.stop_event.is_set():
        state.stop_event.wait(HEARTBEAT_SECS)
        with state.lock:
            import time
            idle = time.time() - state.last_interaction
        if idle > CACHE_TTL_SECS and not state.stop_event.is_set():
            warm_cache()


def start_chat():
    full_refresh()

    observer = Observer()
    observer.schedule(SourceHandler(), path=".", recursive=True)
    observer.start()
    threading.Thread(target=heartbeat, daemon=True).start()

    print(f"\n╭── {_ANSI_BOLD}❖ Claude Light{_ANSI_RESET} ──╮")
    print(f"│ {_ANSI_CYAN}{MODEL}{_ANSI_RESET}  |  "
          f"RAG top-{_ANSI_BOLD}{state.TOP_K}{_ANSI_RESET}  |  "
          f"Embed: {state.EMBED_MODEL}")
    print(f"╰{'─'*20}╯")
    print(f"{_ANSI_DIM}Commands: /compact  /cost  /run <cmd>  /help  exit{_ANSI_RESET}\n")

    if _PROMPTTK_AVAILABLE:
        from prompt_toolkit import HTML

        def get_status_bar():
            total_in = state.session_tokens["input"] + state.session_tokens["cache_write"] + state.session_tokens["cache_read"]
            saved = state.session_tokens["cache_read"]
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            cost = state.session_cost
            repo = os.path.basename(os.getcwd())
            
            return HTML(
                f' <b>Repo:</b> <ansicyan>{repo}</ansicyan>  |  '
                f'<b>Tokens:</b> {total_in:,} '
                f'(<ansigreen>{saved:,}</ansigreen> saved, <ansigreen>{ratio:.1f}%</ansigreen>)  |  '
                f'<b>Cost:</b> <ansiyellow>${cost:.4f}</ansiyellow>'
            )

        CACHE_DIR.mkdir(exist_ok=True)
        _slash_completer = _WordCompleter(
            ["/compact", "/clear", "/cost", "/help", "/run", "exit", "quit"],
            sentence=True,
        )
        _session = _PromptSession(
            history=_FileHistory(str(CACHE_DIR / "history.txt")),
            auto_suggest=_AutoSuggest(),
            completer=_slash_completer,
            complete_while_typing=False,
            bottom_toolbar=get_status_bar
        )
        def _get_input():
            return _session.prompt("> ").strip()
    else:
        def _get_input():
            total_in = state.session_tokens["input"] + state.session_tokens["cache_write"] + state.session_tokens["cache_read"]
            saved = state.session_tokens["cache_read"]
            ratio = (saved / total_in * 100) if total_in > 0 else 0.0
            print(f"\n[{os.path.basename(os.getcwd())}] Tokens: {total_in:,} ({saved:,} saved, {ratio:.1f}%) | Cost: ${state.session_cost:.4f}")
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
                chat(f"I ran the following command and got this output. Please help me understand or fix any issues:\n\n```\n{transcript}\n```")
                continue

            chat(query)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        observer.stop()
        observer.join()
        print_session_summary()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Light RAG Chat")
    parser.add_argument("--test-mode", choices=["small", "medium", "large", "extra-large"],
                        help="Run in test mode with a synthetic codebase and mocked API.")
    parser.add_argument("query", nargs="*", help="Optional query for one-shot mode.")
    args, unknown = parser.parse_known_args()

    if args.test_mode:
        from tests.test_mocks import MockManager
        manager = MockManager(args.test_mode)
        manager.start()

    query_str = " ".join(args.query).strip()
    if query_str:
        one_shot(query_str)
    elif not sys.stdin.isatty():
        one_shot(sys.stdin.read().strip())
    else:
        start_chat()

if __name__ == "__main__":
    main()
