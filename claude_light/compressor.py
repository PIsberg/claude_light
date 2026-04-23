"""LLMLingua-2 prompt compression for retrieved RAG context.

Optional dependency. If `llmlingua` is not installed, `compress_context()`
is a no-op that returns the input unchanged with a diagnostic `info` dict.
Compression failures (model load errors, runtime errors) are swallowed and
the original text is returned — this must never raise into the hot path.
"""

from __future__ import annotations

import threading
import time

from claude_light import config

_compressor = None
_load_error: Exception | None = None
_load_lock = threading.Lock()
_load_attempted = False
_load_thread: threading.Thread | None = None
_load_done = threading.Event()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _noop(text: str, reason: str) -> tuple[str, dict]:
    est = _estimate_tokens(text)
    return text, {
        "tokens_before": est,
        "tokens_after": est,
        "ratio": 1.0,
        "elapsed_ms": 0.0,
        "skipped": True,
        "reason": reason,
    }


def get_compressor():
    """Lazy-load the LLMLingua-2 PromptCompressor. Thread-safe singleton.

    Returns None if llmlingua isn't installed or the model fails to load.
    The load is attempted at most once per process; subsequent calls return
    the cached result (including cached None on failure).

    WARNING: this call can block for seconds (model load) to minutes (first-
    run model download). Do NOT call from the synchronous query path — call
    start_background_load() instead and let compress_context() skip
    compression until _load_done is set.
    """
    global _compressor, _load_error, _load_attempted
    if _load_attempted:
        return _compressor
    with _load_lock:
        if _load_attempted:
            return _compressor
        _load_attempted = True
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except ImportError as e:
            _load_error = e
            _load_done.set()
            return None
        try:
            _compressor = PromptCompressor(
                model_name=config.LLMLINGUA_MODEL,
                use_llmlingua2=True,
            )
        except Exception as e:
            _load_error = e
            _compressor = None
        _load_done.set()
        return _compressor


def start_background_load() -> None:
    """Kick off model load in a daemon thread. Idempotent.

    Call this once at startup (main.py) so the model is ready by the time
    the first query fires. If the user fires a query before loading
    completes, compress_context() returns a noop rather than blocking.
    """
    global _load_thread
    if _load_thread is not None or _load_done.is_set():
        return
    if not config.LLMLINGUA_ENABLED:
        return

    def _worker():
        try:
            get_compressor()
        except Exception:
            _load_done.set()

    _load_thread = threading.Thread(
        target=_worker, daemon=True, name="llmlingua-loader"
    )
    _load_thread.start()


def compress_context(
    text: str,
    rate: float | None = None,
    force_tokens: list[str] | None = None,
) -> tuple[str, dict]:
    """Compress retrieved-context text with LLMLingua-2.

    Never raises. On any failure (missing dep, model load error, runtime
    error) returns the original text with `skipped=True` in `info`.

    Returns (compressed_text, info) where info contains:
      tokens_before, tokens_after, ratio, elapsed_ms, skipped, reason?
    """
    if not text:
        return _noop(text, "empty")
    if not config.LLMLINGUA_ENABLED:
        return _noop(text, "disabled")

    est_before = _estimate_tokens(text)
    if est_before < config.LLMLINGUA_MIN_TOKENS:
        return _noop(text, "below_min_tokens")

    # Never block the query path on model load. If the background loader
    # hasn't finished (first run: ~280 MB download; subsequent runs: ~2–5 s
    # from disk cache), skip compression this turn.
    if not _load_done.is_set():
        start_background_load()  # idempotent — ensures a loader exists
        return _noop(text, "still_loading")

    comp = get_compressor()
    if comp is None:
        reason = "llmlingua_not_installed" if isinstance(_load_error, ImportError) else "load_failed"
        return _noop(text, reason)

    effective_rate = rate if rate is not None else config.LLMLINGUA_TARGET_RATE
    ft = force_tokens if force_tokens is not None else config.LLMLINGUA_FORCE_TOKENS

    t0 = time.perf_counter()
    try:
        result = comp.compress_prompt(text, rate=effective_rate, force_tokens=ft)
    except Exception as e:
        return _noop(text, f"runtime_error:{type(e).__name__}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    compressed = result.get("compressed_prompt", text) if isinstance(result, dict) else text
    tokens_before = int(result.get("origin_tokens", est_before)) if isinstance(result, dict) else est_before
    tokens_after = int(result.get("compressed_tokens", _estimate_tokens(compressed))) if isinstance(result, dict) else _estimate_tokens(compressed)
    ratio = (tokens_after / tokens_before) if tokens_before else 1.0

    # Sanity: if LLMLingua somehow returned a longer output, use the original.
    if tokens_after >= tokens_before:
        return _noop(text, "no_gain")

    return compressed, {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "ratio": ratio,
        "elapsed_ms": elapsed_ms,
        "skipped": False,
    }
