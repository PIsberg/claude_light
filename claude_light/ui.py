import sys
import time
import threading
import difflib
from pathlib import Path

from claude_light.config import PRICE_WRITE, PRICE_READ, PRICE_INPUT, PRICE_OUTPUT, ECONOMY_MODE
from claude_light.testing import TEST_MODE_TAG
import claude_light.state as state

try:
    from rich.console import Console as _RichConsole
    from rich.markdown import Markdown as _RichMarkdown
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

console = _RichConsole() if _RICH_AVAILABLE else None

_ANSI_RED     = "\033[31m"
_ANSI_GREEN   = "\033[32m"
_ANSI_YELLOW  = "\033[33m"
_ANSI_BLUE    = "\033[34m"
_ANSI_MAGENTA = "\033[35m"
_ANSI_CYAN    = "\033[36m"
_ANSI_BOLD    = "\033[1m"
_ANSI_DIM     = "\033[2m"
_ANSI_RESET   = "\033[0m"

# Detect Unicode capability (falls back to ASCII on legacy Windows consoles)
def _stdout_is_utf8() -> bool:
    try:
        enc = (getattr(sys.stdout, 'encoding', '') or '').lower().replace('-', '')
        return enc in ('utf8', 'utf16', 'utf32')
    except Exception:
        return False

_UNICODE = _stdout_is_utf8()

# Symbols — Unicode when supported, ASCII fallback otherwise
_SYM_MARK  = "◆" if _UNICODE else ">"
_SYM_TOOL  = "⏺" if _UNICODE else "*"
_SYM_EDIT  = "⊕" if _UNICODE else "+"
_SYM_ERR   = "✗" if _UNICODE else "!"
_SYM_RESP  = "◆" if _UNICODE else ">"
_SPIN_FRAMES = "⣾⣽⣻⢿⡿⣟⣯⣷" if _UNICODE else r"-\|/"

# Claude Code-style indicators (symbols instead of [BRACKETS])
_T_RAG    = f"{_ANSI_CYAN}{_SYM_TOOL}{_ANSI_RESET}"
_T_CACHE  = f"{_ANSI_DIM}{_SYM_MARK}{_ANSI_RESET}"
_T_SYS    = f"{_ANSI_DIM}{_SYM_MARK}{_ANSI_RESET}"
_T_EDIT   = f"{_ANSI_GREEN}{_SYM_EDIT}{_ANSI_RESET}"
_T_ERR    = f"{_ANSI_RED}{_SYM_ERR}{_ANSI_RESET}"
_T_TEST   = TEST_MODE_TAG
_T_ROUTE  = f"{_ANSI_DIM}{_SYM_MARK}{_ANSI_RESET}"


class _Spinner:
    """Animated spinner for long-running operations."""

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        ellipsis = "…" if _UNICODE else "..."
        while not self._stop.is_set():
            frame = _SPIN_FRAMES[i % len(_SPIN_FRAMES)]
            print(f"\r\033[K  {_ANSI_GREEN}{frame}{_ANSI_RESET}  {_ANSI_DIM}{self.label}{ellipsis}{_ANSI_RESET}",
                  end="", flush=True)
            i += 1
            time.sleep(0.1)
        print("\r\033[K", end="", flush=True)

    def update(self, label: str):
        self.label = label

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


def print_banner(model: str, top_k: int, access_mode: str):
    """Print the Claude Code-style welcome banner."""
    cwd = Path.cwd()
    home = Path.home()
    try:
        display_path = "~/" + str(cwd.relative_to(home)).replace("\\", "/")
    except ValueError:
        display_path = str(cwd).replace("\\", "/")

    model_short = model.replace("claude-", "").replace("-20251001", "")

    print(f"\n  {_ANSI_CYAN}{_ANSI_BOLD}{_SYM_MARK} claude light{_ANSI_RESET}")
    print(f"  {_ANSI_DIM}{display_path}{_ANSI_RESET}\n")
    print(f"  {_ANSI_DIM}{model_short}  ·  RAG top-{top_k}  ·  {access_mode}{_ANSI_RESET}")
    print(f"  {_ANSI_DIM}/help for commands  ·  Ctrl+C to exit{_ANSI_RESET}\n")


def calculate_cost(usage):
    """Always returns the API-equivalent cost regardless of auth mode."""
    write = (getattr(usage, "cache_creation_input_tokens", 0) / 1_000_000) * PRICE_WRITE
    read  = (getattr(usage, "cache_read_input_tokens",      0) / 1_000_000) * PRICE_READ
    inp   = (usage.input_tokens  / 1_000_000) * PRICE_INPUT
    out   = (usage.output_tokens / 1_000_000) * PRICE_OUTPUT
    return write + read + inp + out


def print_stats(usage, label="Stats", file=sys.stdout):
    write_tokens = getattr(usage, "cache_creation_input_tokens", 0)
    read_tokens  = getattr(usage, "cache_read_input_tokens",      0)
    total_input  = usage.input_tokens + write_tokens + read_tokens

    actual_cost  = calculate_cost(usage)
    nocache_cost = (total_input         / 1_000_000) * PRICE_INPUT \
                 + (usage.output_tokens / 1_000_000) * PRICE_OUTPUT

    savings     = nocache_cost - actual_cost
    savings_pct = (savings / nocache_cost * 100) if nocache_cost > 0 else 0.0
    hit_pct     = (read_tokens / total_input * 100) if total_input > 0 else 0.0

    with state.lock:
        session_cost = state.session_cost

    cost_label = "Cost" if ECONOMY_MODE == "USD" else "API equiv."

    sep = "·"
    print(
        f"  {_ANSI_DIM}{_SYM_MARK} {label}"
        f"  {sep}  {total_input:,} tokens"
        f"  {sep}  cached {read_tokens:,} ({hit_pct:.1f}%)"
        f"  {sep}  saved ${savings:.4f} ({savings_pct:.1f}%)"
        f"  {sep}  {cost_label} ${actual_cost:.4f}{_ANSI_RESET}",
        file=file,
    )


def print_session_summary():
    with state.lock:
        inp   = state.session_tokens["input"]
        cw    = state.session_tokens["cache_write"]
        cr    = state.session_tokens["cache_read"]
        out   = state.session_tokens["output"]
        cost  = state.session_cost
        turns = len(state.conversation_history) // 2

    total = inp + cw + cr + out

    cost_inp = (inp / 1_000_000) * PRICE_INPUT
    cost_cw  = (cw  / 1_000_000) * PRICE_WRITE
    cost_cr  = (cr  / 1_000_000) * PRICE_READ
    cost_out = (out / 1_000_000) * PRICE_OUTPUT
    cost_tot = cost_inp + cost_cw + cost_cr + cost_out

    def pct(n):
        return f"{n / total * 100:.1f}%" if total else "—"

    _B = _ANSI_CYAN
    _R = _ANSI_RESET
    _H = _ANSI_BOLD

    input_base = inp + cw + cr
    hit_str    = f"{_ANSI_GREEN}{cr / input_base * 100:.1f}%{_R}" if input_base else "—"

    cost_col_hdr = "Cost" if ECONOMY_MODE == "USD" else "API equiv."
    col_w = [22, 12, 8, 9]

    def row(label, tokens, cost_val, color=""):
        cost_str = f"${cost_val:.4f}"
        return (f"│ {color}{label:<{col_w[0]}}{_R} │ {tokens:>{col_w[1]},} │"
                f" {pct(tokens):>{col_w[2]}} │ {_ANSI_YELLOW}{cost_str:>{col_w[3]}}{_R} │")

    print(f"\n{_B}┌{'─'*62}┐{_R}")
    print(f"{_B}│{_R}{_H}{'Session Token Summary':^62}{_R}{_B}│{_R}")
    print(f"{_B}├{'─'*24}┬{'─'*14}┬{'─'*10}┬{'─'*11}┤{_R}")
    print(f"{_B}│{_R} {'Type':<{col_w[0]}} {_B}│{_R} {'Tokens':>{col_w[1]}} {_B}│{_R} {'%':>{col_w[2]}} {_B}│{_R} {cost_col_hdr:>{col_w[3]}} {_B}│{_R}")
    print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*11}┤{_R}")
    print(row("Input (uncached)",  inp,  cost_inp))
    print(row("Cache write",       cw,   cost_cw))
    print(row("Cache read",        cr,   cost_cr,  _ANSI_GREEN))
    print(row("Output",            out,  cost_out))
    print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*11}┤{_R}")
    print(row("TOTAL",             total, cost_tot, _ANSI_BOLD))
    print(f"{_B}└{'─'*24}┴{'─'*14}┴{'─'*10}┴{'─'*11}┘{_R}")

    print(f"  Turns: {_ANSI_BOLD}{turns}{_R}  ·  Cache hit rate: {hit_str}")

    # --- Global Lifetime Savings ---
    gs = state.global_stats
    full_t  = gs.get("total_tokens_full", 0)
    saved_t = gs.get("total_tokens_saved", 0)
    total_t = full_t + saved_t
    saved_d  = gs.get("total_dollars_saved", 0.0)
    sessions = gs.get("total_sessions", 0)

    overall_pct = (saved_t / total_t * 100) if total_t > 0 else 0.0

    print(f"\n{_B}┌{'─'*57}┐{_R}")
    print(f"{_B}│{_R}{_H}{'Global Lifetime Savings':^57}{_R}{_B}│{_R}")
    print(f"{_B}├{'─'*28}┬{'─'*28}┤{_R}")
    dollars_label = "Total Dollars Saved:" if ECONOMY_MODE == "USD" else "Total API-Equiv. Saved:"
    print(f"{_B}│{_R} {dollars_label:<29}{_B}│{_R} {_ANSI_GREEN}${saved_d:<26.2f}{_R} {_B}│{_R}")
    token_str = f"{saved_t:,} ({overall_pct:.1f}%)"
    print(f"{_B}│{_R} Total Tokens Saved:          {_B}│{_R} {_ANSI_GREEN}{token_str:<26}{_R} {_B}│{_R}")
    print(f"{_B}├{'─'*28}┴{'─'*28}┤{_R}")
    print(f"{_B}│{_R} Sessions: {sessions:<47} {_B}│{_R}")
    print(f"{_B}└{'─'*57}┘{_R}")


def _colorize_diff(lines):
    result = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            result.append(f"{_ANSI_GREEN}{line}{_ANSI_RESET}")
        elif line.startswith("-") and not line.startswith("---"):
            result.append(f"{_ANSI_RED}{line}{_ANSI_RESET}")
        elif line.startswith("@@"):
            result.append(f"{_ANSI_CYAN}{line}{_ANSI_RESET}")
        else:
            result.append(line)
    return result


def show_diff(path, new_content, old_content=None):
    p = Path(path)
    if old_content is None:
        old_content = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if old_content:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        ))
        if diff:
            print("".join(_colorize_diff(line + "\n" for line in diff)))
        else:
            print(f"  {_ANSI_DIM}(no changes detected in {path}){_ANSI_RESET}\n")
    else:
        preview = new_content[:600] + ("…" if len(new_content) > 600 else "")
        print(f"{_ANSI_GREEN}  [NEW FILE]{_ANSI_RESET} {path}\n{preview}\n")


def _print_reply(text):
    """Render Claude's reply — rich Markdown when available, plain text otherwise."""
    if _RICH_AVAILABLE:
        console.print()
        console.print(_RichMarkdown(text))
        console.print()
    else:
        print(f"\n{text}\n")
