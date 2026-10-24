import sys
import time
import threading
import difflib
import re
from pathlib import Path

from claude_light.config import PRICE_WRITE, ECONOMY_MODE, model_prices
from claude_light.testing import TEST_MODE_TAG
import claude_light.state as state

try:
    from rich.console import Console as _RichConsole
    from rich.markdown import Markdown as _RichMarkdown
    from rich.status import Status as _RichStatus
    from rich.syntax import Syntax as _RichSyntax
    from rich.panel import Panel as _RichPanel
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

if _RICH_AVAILABLE:
    from rich.theme import Theme as _RichTheme
    # rich draws inline code bold cyan on black, a box per identifier; a
    # reply naming many functions read as a wall of boxes.
    _REPLY_THEME = _RichTheme({"markdown.code": "cyan"})
console = _RichConsole(theme=_REPLY_THEME) if _RICH_AVAILABLE else None

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


def _display_path() -> str:
    cwd = Path.cwd()
    home = Path.home()
    try:
        return "~/" + str(cwd.relative_to(home)).replace("\\", "/")
    except ValueError:
        return str(cwd).replace("\\", "/")


# Display labels for config.API_KEY_SOURCE. The source string comes back from
# _resolve_api_key() in the same tuple as the key, so it is mapped through
# this table rather than printed: nothing from that tuple reaches the screen.
_ACCESS_LABELS = {
    "Claude CLI (Pro Subscription)": "Claude subscription",
    "Environment": "API key from ANTHROPIC_API_KEY",
    "Dotfile (.anthropic)": "API key from ~/.anthropic",
    "Dotfile (.env)": "API key from .env",
    "Test": "test mode",
}


def access_label(source: str) -> str:
    """A readable label for an auth source, or "" for one not in the table."""
    return _ACCESS_LABELS.get(source, "")


def print_startup_header(model: str | None = None, access_mode: str | None = None):
    """The welcome box: name, project, model and access, how to get help.

    Printed before indexing starts, so it appears at once; the index status
    lives in the toolbar under the prompt.
    """
    help_line = "/help for commands  ·  ctrl+c cancels a turn  ·  exit to quit"
    model_line = ""
    if model:
        model_line = model.replace("claude-", "") + (f"  ·  {access_mode}" if access_mode else "")
    if _RICH_AVAILABLE:
        from rich import box
        from rich.text import Text as _T
        body = _T()
        body.append(f"{_SYM_MARK} claude light", style="bold cyan")
        body.append(f"\n\n{_display_path()}")
        if model_line:
            body.append(f"\n{model_line}", style="dim")
        body.append(f"\n{help_line}", style="dim")
        con = _RichConsole(file=sys.stdout)
        con.print()
        con.print(_RichPanel(body, box=box.ROUNDED, border_style="cyan", expand=False, padding=(0, 2)))
    else:
        print(f"\n  {_ANSI_CYAN}{_ANSI_BOLD}{_SYM_MARK} claude light{_ANSI_RESET}")
        print(f"  {_display_path()}")
        if model_line:
            print(f"  {_ANSI_DIM}{model_line}{_ANSI_RESET}")
        print(f"  {_ANSI_DIM}{help_line}{_ANSI_RESET}", flush=True)


def _count(value) -> float:
    """A token count from a usage field, or 0 for anything that isn't one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return value


def _one_hour_writes(usage) -> float:
    """Cache-write tokens made with the 1-hour TTL, when the usage says so."""
    split = getattr(usage, "cache_creation", None)
    if isinstance(split, dict):
        return _count(split.get("ephemeral_1h_input_tokens"))
    return _count(getattr(split, "ephemeral_1h_input_tokens", 0))


def cost_breakdown(usage, model=None) -> dict:
    """API-equivalent USD per token category, priced at `model`'s rates.

    When the usage carries a cost reported by the Claude CLI (`cost_usd`),
    the categories are scaled to that exact total: the CLI may run more than
    one model in a turn.
    """
    p = model_prices(model)
    writes = _count(getattr(usage, "cache_creation_input_tokens", 0))
    writes_1h = min(_one_hour_writes(usage), writes)
    costs = {
        "input":       _count(usage.input_tokens) / 1e6 * p["input"],
        "cache_write": ((writes - writes_1h) * p["write"] + writes_1h * p["write_1h"]) / 1e6,
        "cache_read":  _count(getattr(usage, "cache_read_input_tokens", 0)) / 1e6 * p["read"],
        "output":      _count(usage.output_tokens) / 1e6 * p["output"],
    }
    reported = getattr(usage, "cost_usd", None)
    if _count(reported) and sum(costs.values()) > 0:
        scale = reported / sum(costs.values())
        costs = {k: v * scale for k, v in costs.items()}
    return costs


def calculate_cost(usage, model=None):
    """Always returns the API-equivalent cost regardless of auth mode."""
    reported = getattr(usage, "cost_usd", None)
    if _count(reported):
        return float(reported)
    return sum(cost_breakdown(usage, model).values())


def print_stats(usage, label="Stats", file=sys.stdout, model=None):
    write_tokens = getattr(usage, "cache_creation_input_tokens", 0)
    read_tokens  = getattr(usage, "cache_read_input_tokens",      0)
    total_input  = usage.input_tokens + write_tokens + read_tokens

    p = model_prices(model)
    actual_cost  = calculate_cost(usage, model)
    nocache_cost = (total_input         / 1_000_000) * p["input"] \
                 + (usage.output_tokens / 1_000_000) * p["output"]

    savings     = nocache_cost - actual_cost
    savings_pct = (savings / nocache_cost * 100) if nocache_cost > 0 else 0.0
    hit_pct     = (read_tokens / total_input * 100) if total_input > 0 else 0.0

    with state.lock:
        session_cost = state.session_cost

    cost_label = "Cost" if ECONOMY_MODE == "USD" else "API equiv."

    sep = "·"
    if savings >= 0:
        savings_str = f"saved ${savings:.4f} ({savings_pct:.1f}%)"
    else:
        savings_str = f"overhead ${-savings:.4f} ({-savings_pct:.1f}%)"
    print(
        f"  {_ANSI_DIM}{_SYM_MARK} {label}"
        f"  {sep}  {total_input:,} tokens"
        f"  {sep}  cached {read_tokens:,} ({hit_pct:.1f}%)"
        f"  {sep}  {savings_str}"
        f"  {sep}  {cost_label} ${actual_cost:.4f}{_ANSI_RESET}",
        file=file,
    )


def print_session_summary():
    with state.lock:
        inp          = state.session_tokens["input"]
        cw           = state.session_tokens["cache_write"]
        cr           = state.session_tokens["cache_read"]
        out          = state.session_tokens["output"]
        compress_pre  = state.session_tokens.get("compress_pre", 0)
        compress_post = state.session_tokens.get("compress_post", 0)
        cost  = state.session_cost
        turns = state.turn_count
        # Accumulated per call at the rate of the model that ran it.
        cost_inp = state.session_costs["input"]
        cost_cw  = state.session_costs["cache_write"]
        cost_cr  = state.session_costs["cache_read"]
        cost_out = state.session_costs["output"]

    total = inp + cw + cr + out
    cost_tot = cost_inp + cost_cw + cost_cr + cost_out

    def pct(n):
        return f"{n / total * 100:.1f}%" if total else "—"

    _B = _ANSI_CYAN
    _R = _ANSI_RESET
    _H = _ANSI_BOLD

    input_base = inp + cw + cr
    hit_str    = f"{_ANSI_GREEN}{cr / input_base * 100:.1f}%{_R}" if input_base else "—"

    cost_col_hdr = "Cost" if ECONOMY_MODE == "USD" else "API equiv."
    col_w = [22, 12, 8, 10]

    def row(label, tokens, cost_val, color=""):
        cost_str = f"${cost_val:.4f}"
        return (f"│ {color}{label:<{col_w[0]}}{_R} │ {tokens:>{col_w[1]},} │"
                f" {pct(tokens):>{col_w[2]}} │ {_ANSI_YELLOW}{cost_str:>{col_w[3]}}{_R} │")

    print(f"\n{_B}┌{'─'*63}┐{_R}")
    print(f"{_B}│{_R}{_H}{'Session Token Summary':^63}{_R}{_B}│{_R}")
    print(f"{_B}├{'─'*24}┬{'─'*14}┬{'─'*10}┬{'─'*12}┤{_R}")
    print(f"{_B}│{_R} {'Type':<{col_w[0]}} {_B}│{_R} {'Tokens':>{col_w[1]}} {_B}│{_R} {'%':>{col_w[2]}} {_B}│{_R} {cost_col_hdr:>{col_w[3]}} {_B}│{_R}")
    print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*12}┤{_R}")
    print(row("Input (uncached)",  inp,  cost_inp))
    print(row("Cache write",       cw,   cost_cw))
    print(row("Cache read",        cr,   cost_cr,  _ANSI_GREEN))
    print(row("Output",            out,  cost_out))
    print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*12}┤{_R}")
    print(row("TOTAL",             total, cost_tot, _ANSI_BOLD))

    comp_delta = compress_pre - compress_post if compress_pre > compress_post else 0
    if comp_delta > 0:
        comp_cost     = (comp_delta / 1_000_000) * PRICE_WRITE
        comp_tok_str  = f"-{comp_delta:,}"
        comp_cost_str = f"-${comp_cost:.4f}"
        print(f"{_B}├{'─'*24}┼{'─'*14}┼{'─'*10}┼{'─'*12}┤{_R}")
        print(
            f"{_B}│{_R} {_ANSI_GREEN}{'Compressed away':<{col_w[0]}}{_R}"
            f" {_B}│{_R} {_ANSI_GREEN}{comp_tok_str:>{col_w[1]}}{_R}"
            f" {_B}│{_R} {'':>{col_w[2]}}"
            f" {_B}│{_R} {_ANSI_GREEN}{comp_cost_str:>{col_w[3]}}{_R} {_B}│{_R}"
        )

    print(f"{_B}└{'─'*24}┴{'─'*14}┴{'─'*10}┴{'─'*12}┘{_R}")

    with state.lock:
        cost_without_cache = state.session_costs["without_cache"]
    saved_label, saved_str = "Saved", "—"
    if cost_without_cache > 0:
        saved_pct = (cost_without_cache - cost_tot) / cost_without_cache * 100
        if saved_pct >= 0:
            saved_str = f"{_ANSI_GREEN}{saved_pct:.1f}%{_R}"
        else:  # cache writes cost more than they saved so far
            saved_label, saved_str = "Overhead", f"{_ANSI_YELLOW}{-saved_pct:.1f}%{_R}"

    print(f"  Turns: {_ANSI_BOLD}{turns}{_R}  ·  Cache hit rate: {hit_str}  ·  {saved_label}: {saved_str}")

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
    print(f"{_B}│{_R} {dollars_label:<27}{_B}│{_R} {_ANSI_GREEN}${saved_d:<25.2f}{_R} {_B}│{_R}")
    token_str = f"{saved_t:,} ({overall_pct:.1f}%)"
    print(f"{_B}│{_R} Total Tokens Saved:        {_B}│{_R} {_ANSI_GREEN}{token_str:<27}{_R}{_B}│{_R}")
    print(f"{_B}├{'─'*28}┴{'─'*28}┤{_R}")
    print(f"{_B}│{_R} Sessions: {sessions:<45} {_B}│{_R}")
    print(f"{_B}└{'─'*57}┘{_R}")


def print_session_footer():
    """One line on exit: turns, tokens, cache hits, cost. /cost has the tables."""
    with state.lock:
        inp, cw = state.session_tokens["input"], state.session_tokens["cache_write"]
        cr, out = state.session_tokens["cache_read"], state.session_tokens["output"]
        cost = sum(state.session_costs[k] for k in ("input", "cache_write", "cache_read", "output"))
        without = state.session_costs["without_cache"]
        turns = state.turn_count
    total = inp + cw + cr + out
    if not total:
        return
    parts = [f"{turns} turn{'' if turns == 1 else 's'}", f"{total:,} tokens"]
    if inp + cw + cr:
        parts.append(f"{cr / (inp + cw + cr) * 100:.1f}% cached")
    parts.append(f"{'Cost' if ECONOMY_MODE == 'USD' else 'API equiv.'} ${cost:.4f}")
    if without > 0:
        pct = (without - cost) / without * 100
        parts.append(f"saved {pct:.1f}%" if pct >= 0 else f"overhead {-pct:.1f}%")
    print(f"\n  {_ANSI_DIM}{_SYM_MARK} " + "  ·  ".join(parts) + _ANSI_RESET)


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

    if not old_content:
        if _RICH_AVAILABLE:
            # Use a per-call Console pointing at the current sys.stdout so that
            # output respects any sys.stdout patches active during tests.
            _con = _RichConsole(file=sys.stdout)
            preview = new_content[:800] + ("…" if len(new_content) > 800 else "")
            _con.print(
                _RichPanel(
                    _RichSyntax(preview, Path(path).suffix.lstrip(".") or "text",
                                theme="monokai", line_numbers=True),
                    title=f"[green][NEW FILE][/green]  {path}",
                    border_style="dim green",
                )
            )
        else:
            preview = new_content[:600] + ("…" if len(new_content) > 600 else "")
            print(f"{_ANSI_GREEN}  [NEW FILE]{_ANSI_RESET} {path}\n{preview}\n")
        return

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    ))
    if not diff:
        print(f"  {_ANSI_DIM}(no changes detected in {path}){_ANSI_RESET}\n")
        return

    # Content lines keep their own newline; joined with another, every diff
    # line was followed by a blank one.
    show_unified_diff(path, "\n".join(line.rstrip("\r\n") for line in diff))


def show_unified_diff(path, diff_text, max_lines=80):
    """A unified diff in a panel titled with the path.

    The file headers (diff --git, index, ---/+++) before the first hunk are
    dropped: the title already names the file. Longer diffs are cut at
    max_lines with a count of what was left out.
    """
    lines = diff_text.splitlines()
    first_hunk = next((i for i, l in enumerate(lines) if l.startswith("@@")), 0)
    lines = lines[first_hunk:]
    hidden = len(lines) - max_lines
    lines = lines[:max_lines]
    more = f"{hidden} more lines" if hidden > 0 else None
    if _RICH_AVAILABLE:
        _con = _RichConsole(file=sys.stdout)
        _con.print(
            _RichPanel(
                _RichSyntax("\n".join(lines), "diff", theme="monokai", line_numbers=False),
                title=f"[cyan]{path}[/cyan]",
                title_align="left",
                subtitle=f"[dim]{more}[/dim]" if more else None,
                border_style="dim blue",
            )
        )
    else:
        print(f"  {_ANSI_CYAN}{path}{_ANSI_RESET}")
        print("\n".join(_colorize_diff(lines)))
        if more:
            print(f"  {_ANSI_DIM}… {more}{_ANSI_RESET}")


class ToolExecutionManager:
    """Context manager that shows a rich status spinner while a tool runs,
    then prints a concise success/failure summary line on exit."""

    def __init__(self, label: str):
        self._label = label
        self._start = 0.0
        self._status = None

    def __enter__(self):
        self._start = time.monotonic()
        if _RICH_AVAILABLE and console is not None:
            self._status = _RichStatus(
                f"[dim]{self._label}…[/dim]",
                console=console,
                spinner="dots",
            )
            self._status.__enter__()
        else:
            print(f"  {_T_SYS}  {self._label}…", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.monotonic() - self._start
        if self._status is not None:
            self._status.__exit__(exc_type, exc_val, exc_tb)
        sym = _SYM_ERR if exc_type else ("✔" if _UNICODE else "+")
        color = _ANSI_RED if exc_type else _ANSI_GREEN
        print(
            f"  {color}{sym}{_ANSI_RESET}  "
            f"{_ANSI_DIM}{self._label} ({elapsed:.1f}s){_ANSI_RESET}"
        )
        return False  # don't suppress exceptions


NOTICE_SECS = 8


def notify(text: str, error: bool = False) -> None:
    """Post a notice from a background thread; the toolbar shows it for
    NOTICE_SECS. It never prints: a print from another thread lands inside
    a streaming reply, or over the prompt the user is typing."""
    with state.lock:
        state.notice = (text, time.monotonic() + NOTICE_SECS, error)


def current_notice() -> str:
    """The notice to show now, or "" when there is none or it has expired."""
    with state.lock:
        notice = state.notice
    if not notice or time.monotonic() > notice[1]:
        return ""
    return f"{_SYM_ERR} {notice[0]}" if notice[2] else notice[0]


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _split_committable(text: str) -> tuple[str, str]:
    """Split streamed markdown into (finished blocks, unfinished tail).

    A block is finished once a blank line follows it, or its code fence has
    closed. Inside an open fence nothing is finished: a blank line there is
    part of the code.
    """
    cut = pos = 0
    fence = None
    for line in text.split("\n")[:-1]:  # the last piece has no newline yet
        pos += len(line) + 1
        m = _FENCE_RE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                    and not line.strip().strip(fence[0]):
                fence = None
                cut = pos
        elif m:
            fence = m.group(1)
        elif not line.strip():
            cut = pos
    return text[:cut], text[cut:]


if _RICH_AVAILABLE:
    from rich.markdown import Heading as _RichHeading
    from rich.segment import Segment as _Segment
    from rich.style import Style as _Style
    from rich.text import Text as _Text

    class _LeftHeading(_RichHeading):
        # rich centres h1, which reads oddly in a chat transcript.
        LEVEL_ALIGN = {**_RichHeading.LEVEL_ALIGN, "h1": "left"}

    class _ReplyMarkdown(_RichMarkdown):
        elements = {**_RichMarkdown.elements, "heading_open": _LeftHeading}

    class _Gutter:
        """Indent a renderable by two columns, with an optional marker in the
        first line's gutter. With max_lines, keep only the last lines: the
        part of a long block still streaming that fits on screen."""

        def __init__(self, renderable, marker=None, max_lines=None):
            self.renderable = renderable
            self.marker = marker
            self.max_lines = max_lines

        @staticmethod
        def _blank(line) -> bool:
            return all(not seg.text.strip() and not (seg.style and seg.style.bgcolor)
                       for seg in line)

        def __rich_console__(self, console, options):
            width = max(options.max_width - 2, 10)
            lines = console.render_lines(self.renderable, options.update(width=width), pad=False)
            while lines and self._blank(lines[0]):
                lines.pop(0)
            while lines and self._blank(lines[-1]):
                lines.pop()
            marker = self.marker
            if self.max_lines and len(lines) > self.max_lines:
                lines = lines[-self.max_lines:]
                marker = None  # its line scrolled off
            for i, line in enumerate(lines):
                if i == 0 and marker:
                    yield _Segment(f"{marker} ", _Style(color="cyan", bold=True))
                else:
                    yield _Segment("  ")
                yield from line
                yield _Segment.line()


def _reply_markdown(text):
    return _ReplyMarkdown(text)


_SPINNER_DOTS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _UNICODE else r"-\|/"


class ResponseView:
    """One assistant turn on screen: a spinner while waiting, the reply as
    rendered markdown while it streams, and a line per tool call.

    Finished markdown blocks are printed once, in full; only the block still
    arriving is redrawn, so a reply taller than the terminal never leaves
    duplicated frames in the scrollback. Without a terminal on stdout the
    reply is written there as plain text and progress goes to stderr, so a
    redirected one-shot answer holds nothing else.
    """

    def __init__(self, label="Thinking", console=None, animate=True):
        # animate=False: no spinner and no redrawn tail, blocks only.
        self.label = label
        self._animate = animate
        if console is None and _RICH_AVAILABLE and _isatty(sys.stdout):
            console = _RichConsole(file=sys.stdout, theme=_REPLY_THEME)
        self._console = console
        self._live = None
        self._heartbeat = None
        self._pending = ""
        self._in_segment = False   # a text segment has started since the last tool
        self._blocks = 0           # blocks printed in the current segment
        self._wrote_plain = False
        self._closed = False
        self._start = time.monotonic()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        self._start = time.monotonic()
        if self._console is not None:
            if self._animate:
                from rich.live import Live
                self._live = Live(self, console=self._console, transient=True,
                                  refresh_per_second=10)
                self._live.start()
        else:
            self._start_heartbeat()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        """Print what is still pending and remove the spinner. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._console is not None:
            pending, self._pending = self._pending, ""
            self._commit(pending)
            if self._live is not None:
                self._live.stop()
        else:
            self._stop_heartbeat()
            if self._wrote_plain:
                sys.stdout.write("\n")
                sys.stdout.flush()

    # -- events ------------------------------------------------------------

    def text(self, delta: str):
        if not delta:
            return
        if self._console is None:
            self._stop_heartbeat()
            if not self._in_segment and self._wrote_plain:
                sys.stdout.write("\n\n")
            elif not self._wrote_plain and _isatty(sys.stdout):
                sys.stdout.write(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ")
            self._in_segment = True
            self._wrote_plain = True
            sys.stdout.write(delta)
            sys.stdout.flush()
            return
        self._pending += delta
        done, self._pending = _split_committable(self._pending)
        self._commit(done)

    def tool(self, label: str):
        """A tool call: finish the text so far and show the call on its own line."""
        line = f"{_SYM_TOOL} {label}"
        if self._console is None:
            self._stop_heartbeat()
            if _isatty(sys.stdout):
                if self._wrote_plain and self._in_segment:
                    sys.stdout.write("\n")
                print(f"  {_ANSI_DIM}{line}{_ANSI_RESET}", flush=True)
            else:
                print(f"  {line}", file=sys.stderr, flush=True)
            self._in_segment = False
            self._start_heartbeat()
            return
        pending, self._pending = self._pending, ""
        self._commit(pending)
        if self._in_segment:
            self._console.print()
        self._console.print(_Text(f"  {line}", style="dim"), highlight=False)
        self._in_segment = False
        self._blocks = 0

    def status(self, label: str):
        """Change the spinner's label, e.g. to "Thinking"."""
        self.label = label

    # -- rendering ---------------------------------------------------------

    def _commit(self, block: str):
        if not block.strip():
            return
        if self._blocks or not self._in_segment:
            self._console.print()
        marker = None if self._in_segment else _SYM_RESP
        self._console.print(_Gutter(_reply_markdown(block.strip("\n")), marker=marker))
        self._in_segment = True
        self._blocks += 1

    def __rich_console__(self, console, options):
        # Called by Live on every refresh (another thread): read fields once.
        pending = self._pending
        if pending.strip():
            room = max(console.size.height - 4, 3)
            marker = None if self._in_segment else _SYM_RESP
            if self._blocks or not self._in_segment:
                yield _Text("")
            yield _Gutter(_reply_markdown(pending), marker=marker, max_lines=room)
        elapsed = int(time.monotonic() - self._start)
        frame = _SPINNER_DOTS[int(time.monotonic() * 10) % len(_SPINNER_DOTS)]
        ellipsis = "…" if _UNICODE else "..."
        status = _Text()
        status.append(f"\n{frame} ", style="cyan")
        status.append(f"{self.label}{ellipsis} ", style="")
        status.append(f"({elapsed}s · ctrl+c to interrupt)", style="dim")
        yield status

    # -- plain-mode spinner ------------------------------------------------

    def _start_heartbeat(self):
        if self._animate and _isatty(sys.stderr):
            self._heartbeat = _StderrHeartbeat(self.label)
            self._heartbeat.start()

    def _stop_heartbeat(self):
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None


class NullView:
    """A ResponseView that shows nothing, for calls whose output is not a reply."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, delta):
        pass

    def tool(self, label):
        pass

    def status(self, label):
        pass

    def close(self):
        pass


class _StderrHeartbeat:
    """A one-line '⏺ Thinking… (Ns)' ticker on stderr for terminals without rich."""

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = time.monotonic()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.5)
        print("\r\033[K", end="", file=sys.stderr, flush=True)

    def _run(self):
        ellipsis = "…" if _UNICODE else "..."
        while True:
            elapsed = int(time.monotonic() - self._start)
            print(f"\r\033[K  {_ANSI_DIM}{_SYM_TOOL}  {self.label}{ellipsis} ({elapsed}s){_ANSI_RESET}",
                  end="", file=sys.stderr, flush=True)
            if self._stop.wait(1.0):
                return


def _print_reply(text):
    """Render Claude's reply — rich Markdown when available, plain text otherwise."""
    if _RICH_AVAILABLE:
        console.print()
        console.print(_RichMarkdown(text))
        console.print()
    else:
        print(f"\n{text}\n")
