import os
import io
import json
import threading
import concurrent.futures
from pathlib import Path
from watchdog.events import FileSystemEventHandler

import claude_light.config as config
from claude_light.config import (
    CACHE_DIR, CACHE_INDEX, CACHE_MANIFEST, INDEXABLE_EXTENSIONS, _DOC_PREFIX
)
from claude_light.ui import _T_RAG, _T_CACHE, _T_ERR, _ANSI_GREEN, _ANSI_YELLOW, _ANSI_BOLD, _ANSI_RESET, _ANSI_CYAN, notify
import claude_light.state as state

from claude_light.skeleton import _file_hash, _file_hash_parallel, _get_cached_paths, _invalidate_path_cache, _stat_tuple
from claude_light.parsing import chunk_file
from claude_light.executor import (
    auto_tune, start_embedder_background_load, set_top_k, _load_embedding_model,
)

def _is_skipped(p):
    from claude_light.skeleton import _is_skipped
    return _is_skipped(p)


def _project_path(src) -> str:
    """The key the index uses for a file: project-relative, no './' prefix.

    Watchdog reports './a.py' (with a backslash on Windows) while the index keys
    files as 'a.py'; with the raw path a re-index added a second copy of the
    file's chunks instead of replacing the old ones.
    """
    s = str(src)
    if os.path.isabs(s):
        try:
            return str(Path(s).relative_to(Path.cwd()))
        except ValueError:
            return s
    # Only the prefix goes: the separators must stay as the caller wrote them.
    while s.startswith(("./", ".\\")):
        s = s[2:]
    return s

def _load_cache(source_files: list, embed_model: str, file_hashes: dict, quiet=False) -> tuple:
    try:
        manifest      = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
        cached_model  = manifest.get("embed_model")
        old_hashes    = manifest["files"]
        cached_index  = _decode_index(CACHE_INDEX.read_bytes())
        
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


_LEGACY_INDEX = CACHE_DIR / "index.pkl"
_VECTOR_FIELDS = ("emb", "emb_fast")


def _encode_index(store: dict) -> bytes:
    """The chunk store as .npz bytes: ids and texts as JSON, each vector kind
    as one float32 matrix plus the rows that have it.

    Not a pickle: the cache lives in the project directory, and a cloned repo
    could ship its own. Unpickling it ran whatever code it contained.
    """
    import numpy as np
    ids = list(store)
    meta = json.dumps({"ids": ids, "texts": [store[cid]["text"] for cid in ids]}).encode("utf-8")
    arrays = {"meta": np.frombuffer(meta, dtype=np.uint8)}
    for field in _VECTOR_FIELDS:
        rows = [n for n, cid in enumerate(ids) if store[cid].get(field) is not None]
        if rows:
            arrays[field] = np.stack([np.asarray(store[ids[n]][field], dtype=np.float32) for n in rows])
            arrays[f"{field}_rows"] = np.array(rows, dtype=np.int64)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _decode_index(data: bytes) -> dict:
    """Inverse of _encode_index. allow_pickle=False: object arrays raise."""
    import numpy as np
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode("utf-8"))
        ids = meta["ids"]
        store = {cid: {"text": text} for cid, text in zip(ids, meta["texts"])}
        for field in _VECTOR_FIELDS:
            if field in z.files:
                for row, vec in zip(z[f"{field}_rows"], z[field]):
                    store[ids[int(row)]][field] = vec
    return store


def _atomic_write(path: Path, data) -> None:
    """Write bytes/str to a sibling temp file then os.replace() into place.

    os.replace is atomic on POSIX and Windows, so a crash mid-write leaves the
    previous cache intact instead of a truncated/corrupt file.
    """
    import os
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _save_cache(embed_model: str) -> None:
    # cache_io_lock serializes concurrent writers (multiple reindex timers);
    # combined with _atomic_write the on-disk cache can never be left partial.
    with state.cache_io_lock:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            manifest = {
                "embed_model": embed_model,
                "fast_model": config.FAST_EMBED_MODEL if config.HYBRID_RETRIEVAL else None,
                "files": dict(state._file_hashes),
                "file_stats": dict(state._file_stats),
            }
            with state.lock:
                store_snapshot = dict(state.chunk_store)
            _atomic_write(CACHE_INDEX, _encode_index(store_snapshot))
            _LEGACY_INDEX.unlink(missing_ok=True)  # never read: see _decode_index
            _atomic_write(CACHE_MANIFEST, json.dumps(manifest))
        except Exception as e:
            print(f"[Cache] Failed to save: {e}")

def index_files(quiet=False, dense="background"):
    """Build or refresh the index. With config.HYBRID_RETRIEVAL it returns as
    soon as the static index is ready; `dense` says what happens to the dense
    vectors: "background" (a thread fills them in), "skip" (only those already
    cached are used: one-shot runs, which exit before a pass would finish), or
    "sync" (wait for them)."""
    if config.HYBRID_RETRIEVAL:
        return _index_files_hybrid(quiet, dense)
    if not quiet:
        print(f"{_T_RAG} Scanning project files...", flush=True)

    # Use cached paths from skeleton module to avoid duplicate rglob
    all_paths = _get_cached_paths()
    source_files = [
        p for p in all_paths
        if p.is_file() and p.suffix.lower() in INDEXABLE_EXTENSIONS
    ]
    if not source_files:
        if not quiet:
            print(f"{_T_RAG} No supported source files found.")
        return
    if not quiet:
        print(f"{_T_RAG} Found {_ANSI_BOLD}{len(source_files)}{_ANSI_RESET} source files.")

    state._source_files = source_files
    # Parallel hash computation — read previous manifest first so we can skip
    # re-hashing files whose (mtime, size) hasn't changed since last run.
    try:
        prev_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        prev_manifest = {}
    prev_hashes = prev_manifest.get("files", {})
    prev_stats  = prev_manifest.get("file_stats", {})
    if source_files:
        state._file_hashes, state._file_stats = _file_hash_parallel(
            source_files, prev_hashes, prev_stats
        )
    else:
        state._file_hashes, state._file_stats = {}, {}

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
            print(f"{_T_RAG} Chunking {len(stale_files)} file(s)...", flush=True)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunks in executor.map(process_file_chunks, stale_files):
                new_chunks.extend(chunks)
        if not quiet:
            print(f"{_T_RAG} Chunked → {len(new_chunks)} chunks.")

    if new_chunks:
        # Model must be loaded at this point since we have new chunks to embed
        doc_prefix = _DOC_PREFIX.get(state.EMBED_MODEL, "")
        show_bar   = (len(new_chunks) > 100) and not quiet
        if not show_bar and not quiet:
            print(f"{_T_RAG} Embedding {len(new_chunks)} chunk(s)...", flush=True)
        with state.embedder_lock:
            embeddings = state.embedder.encode(
                [doc_prefix + c["text"] for c in new_chunks],
                normalize_embeddings=True,
                show_progress_bar=show_bar,
                batch_size=256,
            )
        if not show_bar and not quiet:
            print(f"{_T_RAG} Embedded  {len(new_chunks)} chunk(s).")
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


# ---------------------------------------------------------------------------
# Hybrid index (#61): static vectors first ("emb_fast"), dense ones ("emb")
# filled in afterwards. See config.HYBRID_RETRIEVAL.
# ---------------------------------------------------------------------------

_DENSE_BATCH = 64  # chunks per locked encode: queries wait at most one batch


def _scan_sources(quiet):
    all_paths = _get_cached_paths()
    source_files = [p for p in all_paths if p.is_file() and p.suffix.lower() in INDEXABLE_EXTENSIONS]
    state._source_files = source_files
    try:
        prev = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        prev = {}
    if source_files:
        state._file_hashes, state._file_stats = _file_hash_parallel(
            source_files, prev.get("files", {}), prev.get("file_stats", {}))
    else:
        state._file_hashes, state._file_stats = {}, {}
    return source_files


def _load_hybrid_cache(source_files, quiet):
    """(cached entries for unchanged files, stale files). Vectors from a model
    that is no longer configured are dropped, not the whole entry: a new
    dense model re-embeds in the background without re-chunking anything."""
    try:
        manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
        cached_index = _decode_index(CACHE_INDEX.read_bytes())
        old_hashes = manifest["files"]
    except Exception:
        return {}, list(source_files)
    keep_fast = manifest.get("fast_model") == config.FAST_EMBED_MODEL
    keep_dense = manifest.get("embed_model") == config.DENSE_EMBED_MODEL
    by_file: dict = {}
    for cid, entry in cached_index.items():
        by_file.setdefault(cid.split("::", 1)[0], []).append((cid, entry))
    cached, stale = {}, []
    for f in source_files:
        key = str(f)
        if (state._file_hashes.get(key) or _file_hash(f)) != old_hashes.get(key) or key not in by_file:
            stale.append(f)
            continue
        for cid, entry in by_file[key]:
            entry = {"text": entry["text"],
                     **({"emb_fast": entry["emb_fast"]} if keep_fast and "emb_fast" in entry else {}),
                     **({"emb": entry["emb"]} if keep_dense and "emb" in entry else {})}
            cached[cid] = entry
    if not quiet:
        print(f"{_T_CACHE} {_ANSI_GREEN}{len(source_files) - len(stale)}{_ANSI_RESET} files hit, "
              f"{_ANSI_YELLOW}{len(stale)}{_ANSI_RESET} stale/new.")
    return cached, stale


def _ensure_fast_embedder(quiet):
    if state.fast_embedder is None:
        state.fast_embedder = _load_embedding_model(config.FAST_EMBED_MODEL, quiet=quiet)
    return state.fast_embedder


def _index_files_hybrid(quiet, dense):
    if not quiet:
        print(f"{_T_RAG} Scanning project files...", flush=True)
    source_files = _scan_sources(quiet)
    if not source_files:
        if not quiet:
            print(f"{_T_RAG} No supported source files found.")
        return
    state.EMBED_MODEL = config.DENSE_EMBED_MODEL
    store, stale = _load_hybrid_cache(source_files, quiet)

    def _chunks(f):
        try:
            return chunk_file(str(f), f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            return []

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for chunks in pool.map(_chunks, stale):
            for c in chunks:
                store[c["id"]] = {"text": c["text"]}

    missing = [cid for cid, e in store.items() if "emb_fast" not in e]
    fast = _ensure_fast_embedder(quiet)
    if missing:
        vecs = fast.encode([store[cid]["text"] for cid in missing], normalize_embeddings=True, batch_size=256)
        for cid, v in zip(missing, vecs):
            store[cid]["emb_fast"] = v

    set_top_k(sum(len(e["text"]) for e in store.values()), len(store))
    with state.lock:
        state.chunk_store.clear()
        state.chunk_store.update(store)
    if missing or stale:
        _save_cache(state.EMBED_MODEL)

    need_dense = sum(1 for e in store.values() if "emb" not in e)
    if not quiet:
        print(f"{_T_RAG} {_ANSI_GREEN}Index ready{_ANSI_RESET} — {_ANSI_BOLD}{len(source_files)}{_ANSI_RESET} "
              f"files → {_ANSI_BOLD}{len(store)}{_ANSI_RESET} chunks ({len(missing)} embedded with the "
              f"static model; {need_dense} still need dense vectors)")

    state.dense_ready.clear()
    if need_dense == 0:
        start_embedder_background_load(quiet=True)  # only query encoding needs it
        state.dense_ready.set()
    elif dense == "sync":
        _complete_dense_index()
    elif dense == "background":
        threading.Thread(target=_complete_dense_index, daemon=True, name="claude-light-dense").start()


def _complete_dense_index():
    """Give every chunk its dense vector, a batch at a time, then set
    state.dense_ready. Runs in the background; retrieval uses the static
    vectors alone until it finishes."""
    try:
        if state.embedder is None or not state.embedder_ready.is_set():
            state.embedder = _load_embedding_model(state.EMBED_MODEL, quiet=True)
            state.embedder_ready.set()
        prefix = _DOC_PREFIX.get(state.EMBED_MODEL, "")
        while not state.stop_event.is_set():
            with state.lock:
                todo = [(cid, e["text"]) for cid, e in state.chunk_store.items() if "emb" not in e][:_DENSE_BATCH]
            if not todo:
                break
            with state.embedder_lock:
                vecs = state.embedder.encode([prefix + t for _, t in todo], normalize_embeddings=True)
            with state.lock:
                for (cid, text), v in zip(todo, vecs):
                    entry = state.chunk_store.get(cid)
                    if entry is not None and entry["text"] == text:  # not re-chunked meanwhile
                        entry["emb"] = v
        else:
            return  # shutting down: the next run finishes from the cache
        _save_cache(state.EMBED_MODEL)
        state.dense_ready.set()
        notify("dense index ready")
    except Exception as e:
        notify(f"dense index failed ({e}); using the static model only", error=True)


def _chunks_for_file(filepath):
    prefix = filepath + "::"
    return [k for k in state.chunk_store if k == filepath or k.startswith(prefix)]

def reindex_file(path):
    hybrid = config.HYBRID_RETRIEVAL and state.fast_embedder is not None
    if state.embedder is None and not hybrid:
        return
    path = _project_path(path)
    try:
        p      = Path(path)
        text   = p.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_file(path, text)
        entries = [{"text": c["text"]} for c in chunks]

        if hybrid:
            fast = state.fast_embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
            for entry, v in zip(entries, fast):
                entry["emb_fast"] = v
        # In hybrid mode the dense model may still be loading; the background
        # pass then fills these chunks in with the rest.
        if state.embedder is not None and (not hybrid or state.embedder_ready.is_set()):
            doc_prefix = _DOC_PREFIX.get(state.EMBED_MODEL, "")
            with state.embedder_lock:
                embeddings = state.embedder.encode(
                    [doc_prefix + c["text"] for c in chunks],
                    normalize_embeddings=True,
                    batch_size=256,
                )
            for entry, emb in zip(entries, embeddings):
                entry["emb"] = emb

        with state.lock:
            for k in _chunks_for_file(path):
                del state.chunk_store[k]
            for chunk, entry in zip(chunks, entries):
                state.chunk_store[chunk["id"]] = entry

        state._file_hashes[path] = _file_hash(p)
        stat = _stat_tuple(p)
        if stat is not None:
            state._file_stats[path] = list(stat)
        _save_cache(state.EMBED_MODEL)
        notify(f"re-indexed {path} ({len(chunks)} chunks)")
    except Exception as e:
        notify(f"could not re-index {path}: {e}", error=True)

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
    path = _project_path(path)
    with state.lock:
        for k in _chunks_for_file(path):
            del state.chunk_store[k]
    state._file_hashes.pop(path, None)
    state._file_stats.pop(path, None)
    if state.EMBED_MODEL:
        _save_cache(state.EMBED_MODEL)


class SourceHandler(FileSystemEventHandler):
    @staticmethod
    def _watched(src):
        """The event's project path, or None for files the index skips
        (.git, .venv, node_modules, the cache directory, ...)."""
        path = _project_path(src)
        return None if _is_skipped(Path(path)) else path

    def on_modified(self, event):
        if event.is_directory:
            return
        src = self._watched(event.src_path)
        if src is None:
            return
        ext = Path(src).suffix.lower()
        if ext == ".md":
            from claude_light.llm import refresh_md_file
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
            _debounce(src, lambda s=src: reindex_file(s))

    def on_created(self, event):
        if event.is_directory:
            return
        src = self._watched(event.src_path)
        if src is None:
            return
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
        src = self._watched(event.src_path)
        if src is None:
            return
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
        src, dest   = self._watched(event.src_path), self._watched(event.dest_path)
        if src is None and dest is None:
            return
        # One side is None when the file moved into or out of a skipped dir.
        src_ext     = Path(src).suffix.lower() if src else ""
        dest_ext    = Path(dest).suffix.lower() if dest else ""
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
