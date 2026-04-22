import os
import sys
import io
import threading
import contextlib
import warnings
import logging
from typing import Optional

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests.*",
    category=UserWarning,
    module="huggingface_hub.*",
)

from sentence_transformers import SentenceTransformer
from pathlib import Path
from claude_light.config import TARGET_RETRIEVED_TOKENS
from claude_light.ui import _Spinner, _T_RAG, _T_ERR, _ANSI_GREEN, _ANSI_BOLD, _ANSI_RESET, _ANSI_DIM, _ANSI_YELLOW
import claude_light.state as state

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None

_RUN_HEAD_LINES   = 20
_RUN_TAIL_LINES   = 80
_RUN_MAX_CHARS    = 8_000

# Model sizes in MB (approximate for common models)
_MODEL_SIZES = {
    "all-MiniLM-L6-v2": 22,
    "all-mpnet-base-v2": 420,
    "nomic-ai/nomic-embed-text-v1.5": 550,
}


def _get_model_cache_dir() -> Path:
    """Get the huggingface cache directory."""
    cache_home = os.environ.get("HF_HOME")
    if cache_home:
        return Path(cache_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _check_model_cached(model_name: str) -> bool:
    """Check if model is already cached locally."""
    cache_dir = _get_model_cache_dir()
    # Model names are sanitized by huggingface (slashes become hyphens)
    sanitized_name = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{sanitized_name}"
    return model_path.exists()


def _detect_device() -> tuple[str, str]:
    """Return (device, hint). Hint is non-empty when torch is CPU-only
    on a machine that has CUDA-capable hardware — the user can install
    the CUDA wheel for a large speedup."""
    try:
        import torch
    except ImportError:
        return "cpu", ""

    if torch.cuda.is_available():
        return "cuda", ""

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", ""

    hint = ""
    try:
        import shutil
        if getattr(torch.version, "cuda", None) is None and shutil.which("nvidia-smi"):
            hint = (
                "NVIDIA GPU detected but PyTorch is CPU-only — "
                "embedding will run on CPU. For a 20-100x speedup:\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
    except Exception:
        pass
    return "cpu", hint


def _load_embedding_model(model_name: str, quiet: bool = False) -> SentenceTransformer:
    """
    Load embedding model with optional download progress display.

    Args:
        model_name: HuggingFace model identifier
        quiet: If True, suppress progress output

    Returns:
        Loaded SentenceTransformer instance
    """
    if state.device is None:
        state.device, _device_hint = _detect_device()
        if _device_hint and not quiet:
            print(f"{_T_RAG} {_ANSI_YELLOW}{_device_hint}{_ANSI_RESET}")

    is_cached = _check_model_cached(model_name)
    model_size_mb = _MODEL_SIZES.get(model_name, 100)

    if quiet or is_cached:
        with contextlib.redirect_stderr(io.StringIO()):
            return SentenceTransformer(model_name, trust_remote_code=True, device=state.device)
    
    # Show download progress for first-time model loads
    if _TQDM_AVAILABLE:
        print(f"\n{_T_RAG} Downloading {_ANSI_BOLD}{model_name}{_ANSI_RESET} ({model_size_mb} MB)...")
        
        # Use tqdm to track download
        with tqdm(
            total=100,
            desc=f"  {model_name}",
            unit="%",
            ncols=80,
            bar_format="{desc}: {percentage:3.0f}% {bar}",
        ) as pbar:
            with contextlib.redirect_stderr(io.StringIO()):
                model = SentenceTransformer(model_name, trust_remote_code=True, device=state.device)
            pbar.update(100)  # Mark as complete
        
        print(f"{_T_RAG} {_ANSI_GREEN}Downloaded{_ANSI_RESET} {model_name}\n")
    else:
        # Fallback without tqdm
        print(f"\n{_T_RAG} Downloading {_ANSI_BOLD}{model_name}{_ANSI_RESET} ({model_size_mb} MB)...")
        print(f"  (This may take a minute on first run...)")
        with contextlib.redirect_stderr(io.StringIO()):
            model = SentenceTransformer(model_name, trust_remote_code=True, device=state.device)
        print(f"{_T_RAG} {_ANSI_GREEN}Downloaded{_ANSI_RESET} {model_name}\n")
    
    return model


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

def auto_tune(source_files, chunks=None, quiet=False, load_model=True):
    n = len(source_files)
    if n < 50:
        chosen_model = "all-MiniLM-L6-v2"
    elif n < 200:
        chosen_model = "all-mpnet-base-v2"
    else:
        chosen_model = "nomic-ai/nomic-embed-text-v1.5"

    # Only load model if explicitly requested AND (model changed or not loaded)
    # This prevents loading on warm cache hits
    if load_model and (chosen_model != state.EMBED_MODEL or state.embedder is None):
        state.EMBED_MODEL = chosen_model
        state.embedder = _load_embedding_model(state.EMBED_MODEL, quiet=quiet)
        state.embedder_ready.set()
        if not quiet:
            print(f"{_T_RAG} {_ANSI_GREEN}Loaded{_ANSI_RESET} {chosen_model}")
    elif load_model:
        # Already loaded — just mark ready (idempotent for callers)
        state.embedder_ready.set()
    else:
        # Just update the model choice without loading
        state.EMBED_MODEL = chosen_model

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


def start_embedder_background_load(quiet: bool = False) -> threading.Thread:
    """Load state.EMBED_MODEL in a daemon thread; set state.embedder_ready when done.

    Used on full cache hits: the model is only needed for query encoding, so we
    return control to the caller immediately and let retrieve() block on the
    event if the user's first query arrives before the load finishes.
    """
    state.embedder_ready.clear()
    state.embedder_load_error = None
    model_name = state.EMBED_MODEL

    def _load():
        try:
            state.embedder = _load_embedding_model(model_name, quiet=True)
        except Exception as exc:
            state.embedder_load_error = exc
        finally:
            state.embedder_ready.set()

    t = threading.Thread(target=_load, daemon=True, name="claude-light-embedder")
    t.start()
    return t
