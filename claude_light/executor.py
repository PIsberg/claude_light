import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests.*",
    category=UserWarning,
    module="huggingface_hub.*",
)

from sentence_transformers import SentenceTransformer
from claude_light.config import TARGET_RETRIEVED_TOKENS
from claude_light.ui import _Spinner, _T_RAG, _T_ERR, _ANSI_GREEN, _ANSI_BOLD, _ANSI_RESET, _ANSI_DIM, _ANSI_YELLOW
import claude_light.state as state

_RUN_HEAD_LINES   = 20
_RUN_TAIL_LINES   = 80
_RUN_MAX_CHARS    = 8_000

def _run_command(cmd: str) -> str:
    import subprocess, shlex
    _T_RUN = f"{_ANSI_YELLOW}[Run]{_ANSI_RESET}"
    print(f"\n{_T_RUN} {_ANSI_DIM}$ {cmd}{_ANSI_RESET}")
    print(f"{_ANSI_DIM}{'-'*60}{_ANSI_RESET}")
    try:
        proc = subprocess.run(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception as exc:
        msg = f"[Run] Failed to start process: {exc}"
        print(f"{_T_ERR} {msg}")
        return msg

    raw = proc.stdout or ""
    exit_code = proc.returncode

    print(raw, end="")
    status_color = _ANSI_GREEN if exit_code == 0 else "\033[31m" # RED
    print(f"{_ANSI_DIM}{'-'*60}{_ANSI_RESET}")
    print(f"{_T_RUN} Exit code: {status_color}{exit_code}{_ANSI_RESET}\n")

    lines = raw.splitlines()
    if len(lines) <= _RUN_HEAD_LINES + _RUN_TAIL_LINES:
        trimmed = raw
    else:
        head = "\n".join(lines[:_RUN_HEAD_LINES])
        tail = "\n".join(lines[-_RUN_TAIL_LINES:])
        omitted = len(lines) - _RUN_HEAD_LINES - _RUN_TAIL_LINES
        trimmed = f"{head}\n\n... ({omitted} lines omitted) ...\n\n{tail}"

    if len(trimmed) > _RUN_MAX_CHARS:
        trimmed = trimmed[-_RUN_MAX_CHARS:]
        trimmed = f"... (truncated to last {_RUN_MAX_CHARS} chars) ...\n" + trimmed

    return f"$ {cmd}\n(exit {exit_code})\n{trimmed}"

def auto_tune(source_files, chunks=None, quiet=False):
    n = len(source_files)
    if n < 50:
        chosen_model = "all-MiniLM-L6-v2"
    elif n < 200:
        chosen_model = "all-mpnet-base-v2"
    else:
        chosen_model = "nomic-ai/nomic-embed-text-v1.5"

    if chosen_model != state.EMBED_MODEL or state.embedder is None:
        state.EMBED_MODEL = chosen_model
        if quiet:
            state.embedder = SentenceTransformer(state.EMBED_MODEL, trust_remote_code=True)
        else:
            with _Spinner(f"{_T_RAG} Loading {_ANSI_BOLD}{chosen_model}{_ANSI_RESET}"):
                state.embedder = SentenceTransformer(state.EMBED_MODEL, trust_remote_code=True)
            print(f"{_T_RAG} {_ANSI_GREEN}Loaded{_ANSI_RESET} {chosen_model}")

    if chunks:
        n_units     = len(chunks)
        total_chars = sum(len(c["text"]) for c in chunks)
    else:
        n_units     = n
        total_chars = sum(f.stat().st_size for f in source_files if f.exists())

    avg_tokens = max(1, (total_chars // n_units) // 4) if n_units else 1
    state.TOP_K = max(2, min(15, round(TARGET_RETRIEVED_TOKENS / avg_tokens)))

    unit_label = f"{n_units} chunks" if chunks else f"{n} files"
    if not quiet:
        print(
            f"{_T_RAG} Auto-tune → {_ANSI_BOLD}{n} files{_ANSI_RESET} → {unit_label} | "
            f"~{avg_tokens} tok/chunk | TOP_K={_ANSI_BOLD}{state.TOP_K}{_ANSI_RESET} | model={state.EMBED_MODEL}"
        )
