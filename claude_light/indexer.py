import json
import pickle
import threading
import concurrent.futures
from pathlib import Path
from watchdog.events import FileSystemEventHandler

from claude_light.config import (
    CACHE_DIR, CACHE_INDEX, CACHE_MANIFEST, INDEXABLE_EXTENSIONS, _DOC_PREFIX
)
from claude_light.ui import _T_RAG, _T_CACHE, _T_ERR, _ANSI_GREEN, _ANSI_YELLOW, _ANSI_BOLD, _ANSI_RESET, _ANSI_CYAN
import claude_light.state as state

from claude_light.skeleton import _file_hash, _file_hash_parallel, _get_cached_paths, _invalidate_path_cache
from claude_light.parsing import chunk_file
from claude_light.executor import auto_tune, start_embedder_background_load

def _is_skipped(p):
    from claude_light.skeleton import _is_skipped
    return _is_skipped(p)

def _load_cache(source_files: list, embed_model: str, file_hashes: dict, quiet=False) -> tuple:
    try:
        manifest      = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
        cached_model  = manifest.get("embed_model")
        old_hashes    = manifest["files"]
        cached_index  = pickle.loads(CACHE_INDEX.read_bytes())
        
        # Check if model changed - if so, all files are stale
        # If embed_model is None/empty, use cached model (first startup with existing cache)
        if embed_model and cached_model and cached_model != embed_model:
            if (CACHE_MANIFEST.exists() or CACHE_INDEX.exists()) and not quiet:
                print(f"{_T_CACHE} Miss (embed model changed); re-indexing everything.")
            return {}, list(source_files)
    except Exception as exc:
        if (CACHE_MANIFEST.exists() or CACHE_INDEX.exists()) and not quiet:
            print(f"{_T_CACHE} Miss ({exc}); re-indexing everything.")
        return {}, list(source_files)

    cached_store: dict = {}
    stale: list        = []
    for f in source_files:
        key    = str(f)
        f_hash = file_hashes.get(key) or _file_hash(f)
        if f_hash == old_hashes.get(key):
            prefix = key + "::"
            for cid, val in cached_index.items():
                if cid == key or cid.startswith(prefix):
                    cached_store[cid] = val
        else:
            stale.append(f)

    hit  = len(source_files) - len(stale)
    miss = len(stale)
    if not quiet:
        print(f"{_T_CACHE} {_ANSI_GREEN}{hit}{_ANSI_RESET} files hit, "
              f"{_ANSI_YELLOW}{miss}{_ANSI_RESET} stale/new.")
    return cached_store, stale


def _save_cache(embed_model: str) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        manifest = {
            "embed_model": embed_model,
            "files": dict(state._file_hashes),
        }
        with state.lock:
            store_snapshot = dict(state.chunk_store)
        CACHE_INDEX.write_bytes(pickle.dumps(store_snapshot, protocol=pickle.HIGHEST_PROTOCOL))
        CACHE_MANIFEST.write_text(json.dumps(manifest), encoding="utf-8")
    except Exception as e:
        print(f"[Cache] Failed to save: {e}")

def index_files(quiet=False):
    if not quiet:
        print(f"{_T_RAG} Scanning project files...", end="", flush=True)
    
    # Use cached paths from skeleton module to avoid duplicate rglob
    all_paths = _get_cached_paths()
    source_files = [
        p for p in all_paths
        if p.is_file() and p.suffix.lower() in INDEXABLE_EXTENSIONS
    ]
    if not source_files:
        if not quiet:
            print(f"\r{_T_RAG} No supported source files found.")
        return
    if not quiet:
        print(f"\r{_T_RAG} Found {_ANSI_BOLD}{len(source_files)}{_ANSI_RESET} source files.")

    state._source_files = source_files
    # Parallel hash computation for all source files
    state._file_hashes = _file_hash_parallel(source_files) if source_files else {}

    # Check cache status BEFORE loading model - pass hashes so auto_tune can check
    cached_store, stale_files = _load_cache(source_files, state.EMBED_MODEL, state._file_hashes, quiet=quiet)

    if stale_files:
        # Must load model synchronously — we're about to embed new chunks.
        auto_tune(source_files, quiet=quiet, load_model=True)
    else:
        # Full cache hit. Pick the model + set TOP_K synchronously, then
        # defer the actual model load to a background thread. Query encoding
        # (state.embedder.encode in retrieve()) will wait on embedder_ready.
        auto_tune(source_files, quiet=quiet, load_model=False)
        start_embedder_background_load(quiet=quiet)

    new_chunks: list = []

    def process_file_chunks(f):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            return chunk_file(str(f), text)
        except OSError:
            return []

    if stale_files:
        if not quiet:
            print(f"{_T_RAG} Chunking {len(stale_files)} file(s)...", end="", flush=True)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunks in executor.map(process_file_chunks, stale_files):
                new_chunks.extend(chunks)
        if not quiet:
            print(f"\r{_T_RAG} Chunked → {len(new_chunks)} chunks.")
    else:
        pass

    if new_chunks:
        # Model must be loaded at this point since we have new chunks to embed
        doc_prefix = _DOC_PREFIX.get(state.EMBED_MODEL, "")
        show_bar   = (len(new_chunks) > 100) and not quiet
        if not show_bar and not quiet:
            print(f"{_T_RAG} Embedding {len(new_chunks)} chunk(s)...", end="", flush=True)
        embeddings = state.embedder.encode(
            [doc_prefix + c["text"] for c in new_chunks],
            normalize_embeddings=True,
            show_progress_bar=show_bar,
        )
        if not show_bar and not quiet:
            print(f"\r{_T_RAG} Embedded  {len(new_chunks)} chunk(s).  ")
        new_store = {
            c["id"]: {"text": c["text"], "emb": emb}
            for c, emb in zip(new_chunks, embeddings)
        }
    else:
        new_store = {}

    merged = {**cached_store, **new_store}

    all_chunk_texts = [{"text": v["text"]} for v in merged.values()]
    # Model already loaded above if needed; just update TOP_K based on final chunk count
    auto_tune(source_files, chunks=all_chunk_texts, quiet=quiet, load_model=False)

    with state.lock:
        state.chunk_store.clear()
        state.chunk_store.update(merged)

    if new_store:
        _save_cache(state.EMBED_MODEL)

    cached_n = len(source_files) - len(stale_files)
    if not quiet:
        print(
            f"{_T_RAG} {_ANSI_GREEN}Index ready{_ANSI_RESET} — "
            f"{_ANSI_BOLD}{len(source_files)}{_ANSI_RESET} files → "
            f"{_ANSI_BOLD}{len(merged)}{_ANSI_RESET} chunks "
            f"({_ANSI_GREEN}{cached_n} cached{_ANSI_RESET}, {len(stale_files)} re-embedded)"
        )


def _chunks_for_file(filepath):
    prefix = filepath + "::"
    return [k for k in state.chunk_store if k == filepath or k.startswith(prefix)]

def reindex_file(path):
    if state.embedder is None:
        return
    try:
        p      = Path(path)
        text   = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_file(path, text)

        doc_prefix = _DOC_PREFIX.get(state.EMBED_MODEL, "")
        embeddings = state.embedder.encode(
            [doc_prefix + c["text"] for c in chunks],
            normalize_embeddings=True,
        )

        with state.lock:
            for k in _chunks_for_file(path):
                del state.chunk_store[k]
            for chunk, emb in zip(chunks, embeddings):
                state.chunk_store[chunk["id"]] = {"text": chunk["text"], "emb": emb}

        state._file_hashes[path] = _file_hash(p)
        _save_cache(state.EMBED_MODEL)
        print(f"{_T_RAG} Re-indexed {_ANSI_CYAN}{path}{_ANSI_RESET} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"{_T_ERR} Failed to re-index {path}: {e}")

def _chunk_label(chunk_id):
    if "::" in chunk_id:
        path, method = chunk_id.rsplit("::", 1)
        return f"{Path(path).name}::{method}"
    return Path(chunk_id).name


_file_timers: dict = {}

def _debounce(src: str, fn, delay: float = 1.5) -> None:
    if src in _file_timers:
        _file_timers[src].cancel()
    t = threading.Timer(delay, fn)
    _file_timers[src] = t
    t.start()


def _remove_file_from_index(path: str) -> None:
    with state.lock:
        for k in _chunks_for_file(path):
            del state.chunk_store[k]
    state._file_hashes.pop(path, None)
    if state.EMBED_MODEL:
        _save_cache(state.EMBED_MODEL)


class SourceHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if ext == ".md":
            from claude_light.llm import refresh_md_file
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
            _debounce(src, lambda s=src: reindex_file(s))

    def on_created(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if ext == ".md":
            from claude_light.llm import refresh_md_file
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
            from claude_light.llm import refresh_skeleton_only
            _debounce(src, lambda s=src: reindex_file(s))
            refresh_skeleton_only()

    def on_deleted(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if src in _file_timers:
            _file_timers.pop(src).cancel()
        if ext in INDEXABLE_EXTENSIONS:
            _remove_file_from_index(src)
        if ext == ".md":
            from claude_light.llm import refresh_md_file
            refresh_md_file(src)
        else:
            from claude_light.llm import refresh_skeleton_only
            refresh_skeleton_only()

    def on_moved(self, event):
        if event.is_directory:
            return
        src, dest   = event.src_path, event.dest_path
        src_ext     = Path(src).suffix.lower()
        dest_ext    = Path(dest).suffix.lower()
        for p in (src, dest):
            if p in _file_timers:
                _file_timers.pop(p).cancel()
        if src_ext in INDEXABLE_EXTENSIONS:
            _remove_file_from_index(src)
        if dest_ext in INDEXABLE_EXTENSIONS:
            _debounce(dest, lambda d=dest: reindex_file(d))
        if src_ext == ".md":
            state._skeleton_md_hashes.pop(src, None)
            state._skeleton_md_parts.pop(src, None)
        if dest_ext == ".md":
            from claude_light.llm import refresh_md_file
            _debounce(dest, lambda d=dest: refresh_md_file(d))
        else:
            from claude_light.llm import refresh_skeleton_only
            refresh_skeleton_only()
