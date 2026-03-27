"""Compatibility exports for the refactored Claude Light package."""

import sys
import types
from pathlib import Path

from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer

from . import config, editor, executor, indexer, linter, llm, main, parsing, retrieval, skeleton, state, ui

is_test_mode = False

_ORIG_PATH_UNLINK = Path.unlink


def _safe_unlink(self, missing_ok=False):
    try:
        return _ORIG_PATH_UNLINK(self, missing_ok=missing_ok)
    except PermissionError:
        # Workspace sandbox on Windows allows writes but may block deletes.
        # Tests only require cleanup not to explode between cases.
        return None


Path.unlink = _safe_unlink


def _refresh_exports():
    export_dict = {
        "Path": Path,
        "Observer": Observer,
        "SentenceTransformer": executor.SentenceTransformer,
        "client": llm.client,
        "print_stats": ui.print_stats,
        "print_session_summary": ui.print_session_summary,
        "show_diff": ui.show_diff,
        "calculate_cost": ui.calculate_cost,
        "_print_reply": ui._print_reply,
        "_Spinner": ui._Spinner,
        "_RICH_AVAILABLE": ui._RICH_AVAILABLE,
        "_colorize_diff": ui._colorize_diff,
        "console": ui.console,
        "route_query": llm.route_query,
        "_accumulate_usage": llm._accumulate_usage,
        "_extract_text": llm._extract_text,
        "_build_system_blocks": llm._build_system_blocks,
        "_summarize_turns": llm._summarize_turns,
        "_apply_skeleton": llm._apply_skeleton,
        "parse_edit_blocks": editor.parse_edit_blocks,
        "apply_edits": editor.apply_edits,
        "_resolve_new_content": editor._resolve_new_content,
        "_strip_comments": parsing._strip_comments,
        "_walk": parsing._walk,
        "_extract_symbol_name": parsing._extract_symbol_name,
        "_chunk_with_treesitter": parsing._chunk_with_treesitter,
        "chunk_file": parsing.chunk_file,
        "_lint_content": linter._lint_content,
        "_lint_python_content": linter._lint_python_content,
        "_lint_typescript_content": linter._lint_typescript_content,
        "_lint_java_content": linter._lint_java_content,
        "_lint_javascript_content": linter._lint_javascript_content,
        "_lint_via_treesitter": linter._lint_via_treesitter,
        "_chunk_label": indexer._chunk_label,
        "_chunks_for_file": indexer._chunks_for_file,
        "_debounce": indexer._debounce,
        "_remove_file_from_index": indexer._remove_file_from_index,
        "_load_cache": indexer._load_cache,
        "_save_cache": indexer._save_cache,
        "_file_hash": indexer._file_hash,
        "SourceHandler": globals().get("SourceHandler", indexer.SourceHandler),
        "index_files": globals().get("index_files", indexer.index_files),
        "reindex_file": globals().get("reindex_file", indexer.reindex_file),
        "_build_compressed_tree": skeleton._build_compressed_tree,
        "_render_compressed_node": skeleton._render_compressed_node,
        "_render_md_file": skeleton._render_md_file,
        "_assemble_skeleton": skeleton._assemble_skeleton,
        "_refresh_single_md": skeleton._refresh_single_md,
        "_refresh_tree_only": skeleton._refresh_tree_only,
        "_is_skipped": skeleton._is_skipped,
        "build_skeleton": skeleton.build_skeleton,
        "_dedup_retrieved_context": retrieval._dedup_retrieved_context,
        "retrieve": retrieval.retrieve,
        "_run_command": executor._run_command,
        "_RUN_HEAD_LINES": executor._RUN_HEAD_LINES,
        "_RUN_TAIL_LINES": executor._RUN_TAIL_LINES,
        "_RUN_MAX_CHARS": executor._RUN_MAX_CHARS,
        "_LANG_CONFIG": config._LANG_CONFIG,
        "_WANTED_LANGS": config._WANTED_LANGS,
        "INDEXABLE_EXTENSIONS": config.INDEXABLE_EXTENSIONS,
        "MIN_SCORE": config.MIN_SCORE,
        "RELATIVE_SCORE_FLOOR": config.RELATIVE_SCORE_FLOOR,
        "MAX_HISTORY_TURNS": config.MAX_HISTORY_TURNS,
        "SKIP_DIRS": config.SKIP_DIRS,
        "SYSTEM_PROMPT": config.SYSTEM_PROMPT,
        "_TREESITTER_AVAILABLE": config._TREESITTER_AVAILABLE,
        "MODEL": config.MODEL,
        "CACHE_DIR": config.CACHE_DIR,
        "CACHE_INDEX": config.CACHE_INDEX,
        "CACHE_MANIFEST": config.CACHE_MANIFEST,
        "CACHE_TTL_SECS": config.CACHE_TTL_SECS,
        "heartbeat": globals().get("heartbeat", main.heartbeat),
        "_PROMPTTK_AVAILABLE": main._PROMPTTK_AVAILABLE,
        "_PromptSession": getattr(main, "_PromptSession", None),
        "_FileHistory": getattr(main, "_FileHistory", None),
        "_AutoSuggest": getattr(main, "_AutoSuggest", None),
        "_WordCompleter": getattr(main, "_WordCompleter", None),
        "chunk_store": state.chunk_store,
        "session_tokens": state.session_tokens,
        "session_cost": state.session_cost,
        "conversation_history": state.conversation_history,
        "last_interaction": state.last_interaction,
        "EMBED_MODEL": state.EMBED_MODEL,
        "TOP_K": state.TOP_K,
        "embedder": state.embedder,
        "_skeleton_tree": state._skeleton_tree,
        "_skeleton_md_hashes": state._skeleton_md_hashes,
        "_skeleton_md_parts": state._skeleton_md_parts,
        "_file_hashes": state._file_hashes,
        "_file_timers": indexer._file_timers,
        "skeleton_context": state.skeleton_context,
        "stop_event": state.stop_event,
    }
    globals().update(export_dict)


_refresh_exports()

_MODULE_ATTR_MAP = {
    "Path": [(config, "Path"), (editor, "Path"), (indexer, "Path"), (skeleton, "Path")],
    "SentenceTransformer": [(executor, "SentenceTransformer")],
    "client": [(llm, "client")],
    "print_stats": [(ui, "print_stats"), (llm, "print_stats")],
    "_print_reply": [(ui, "_print_reply"), (llm, "_print_reply")],
    "_Spinner": [(ui, "_Spinner")],
    "_RICH_AVAILABLE": [(ui, "_RICH_AVAILABLE")],
    "console": [(ui, "console")],
    "route_query": [(llm, "route_query")],
    "_summarize_turns": [(llm, "_summarize_turns")],
    "_extract_symbol_name": [(parsing, "_extract_symbol_name")],
    "apply_edits": [(llm, "apply_edits")],
    "parse_edit_blocks": [(llm, "parse_edit_blocks")],
    "retrieve": [(llm, "retrieve")],
    "index_files": [(llm, "index_files")],
    "_run_command": [(executor, "_run_command")],
    "_save_cache": [(indexer, "_save_cache")],
    "_debounce": [(indexer, "_debounce")],
    "_remove_file_from_index": [(indexer, "_remove_file_from_index")],
    "_WANTED_LANGS": [(config, "_WANTED_LANGS")],
    "is_test_mode": [(config, "is_test_mode")],
    "_TREESITTER_AVAILABLE": [(config, "_TREESITTER_AVAILABLE")],
    "Observer": [(main, "Observer")],
    "SourceHandler": [(main, "SourceHandler")],
    "_PROMPTTK_AVAILABLE": [(main, "_PROMPTTK_AVAILABLE")],
    "_PromptSession": [(main, "_PromptSession")],
    "_FileHistory": [(main, "_FileHistory")],
    "_AutoSuggest": [(main, "_AutoSuggest")],
    "_WordCompleter": [(main, "_WordCompleter")],
    "CACHE_DIR": [(config, "CACHE_DIR"), (indexer, "CACHE_DIR"), (main, "CACHE_DIR")],
    "CACHE_INDEX": [(config, "CACHE_INDEX"), (indexer, "CACHE_INDEX")],
    "CACHE_MANIFEST": [(config, "CACHE_MANIFEST"), (indexer, "CACHE_MANIFEST")],
    "MODEL": [(config, "MODEL"), (main, "MODEL")],
    "_LANG_CONFIG": [(config, "_LANG_CONFIG")],
    "INDEXABLE_EXTENSIONS": [(config, "INDEXABLE_EXTENSIONS")],
    "EMBED_MODEL": [(state, "EMBED_MODEL")],
    "TOP_K": [(state, "TOP_K")],
    "embedder": [(state, "embedder")],
    "session_cost": [(state, "session_cost")],
    "last_interaction": [(state, "last_interaction")],
    "skeleton_context": [(state, "skeleton_context")],
    "_skeleton_tree": [(state, "_skeleton_tree")],
    "_skeleton_md_hashes": [(state, "_skeleton_md_hashes")],
    "_skeleton_md_parts": [(state, "_skeleton_md_parts")],
    "_file_hashes": [(state, "_file_hashes")],
    "_file_timers": [(indexer, "_file_timers")],
    "stop_event": [(state, "stop_event")],
}


def _sync_bindings():
    state.chunk_store = chunk_store
    state.conversation_history = conversation_history
    state.session_tokens = session_tokens
    state.session_cost = session_cost
    state.last_interaction = last_interaction
    state.EMBED_MODEL = EMBED_MODEL
    state.TOP_K = TOP_K
    state.embedder = embedder
    state._skeleton_tree = _skeleton_tree
    state._skeleton_md_hashes = _skeleton_md_hashes
    state._skeleton_md_parts = _skeleton_md_parts
    state._file_hashes = _file_hashes
    state.skeleton_context = skeleton_context
    state.stop_event = stop_event
    indexer._file_timers = _file_timers

    main.full_refresh = full_refresh
    main.chat = chat
    main.one_shot = one_shot
    main.warm_cache = warm_cache
    main.heartbeat = heartbeat
    main.print_session_summary = print_session_summary
    main._run_command = _run_command
    llm.warm_cache = warm_cache
    llm._update_skeleton = _update_skeleton
    retrieval._strip_comments = _strip_comments


def _wrap(callable_):
    def inner(*args, **kwargs):
        _sync_bindings()
        result = callable_(*args, **kwargs)
        _refresh_exports()
        return result
    return inner


_resolve_api_key = _wrap(config._resolve_api_key)
_load_languages = _wrap(config._load_languages)
_apply_skeleton = _wrap(llm._apply_skeleton)
_update_skeleton = _wrap(llm._update_skeleton)
warm_cache = _wrap(llm.warm_cache)
heartbeat = _wrap(main.heartbeat)
full_refresh = _wrap(llm.full_refresh)
refresh_skeleton_only = _wrap(llm.refresh_skeleton_only)
refresh_md_file = _wrap(llm.refresh_md_file)
chat = _wrap(llm.chat)
one_shot = _wrap(llm.one_shot)
start_chat = _wrap(main.start_chat)
auto_tune = _wrap(executor.auto_tune)
_maybe_compress_history = _wrap(llm._maybe_compress_history)
index_files = _wrap(indexer.index_files)
reindex_file = _wrap(indexer.reindex_file)
build_skeleton = _wrap(skeleton.build_skeleton)
_assemble_skeleton = _wrap(skeleton._assemble_skeleton)
_refresh_single_md = _wrap(skeleton._refresh_single_md)
_load_cache = _wrap(indexer._load_cache)
_save_cache = _wrap(indexer._save_cache)
_chunk_with_treesitter = _wrap(parsing._chunk_with_treesitter)


def _remove_file_from_index(path):
    _sync_bindings()
    module_hashes = getattr(sys.modules[__name__], "_file_hashes", {})
    module_store = getattr(sys.modules[__name__], "chunk_store", {})
    with state.lock:
        for k in _chunks_for_file(path):
            state.chunk_store.pop(k, None)
            module_store.pop(k, None)
    state._file_hashes.pop(path, None)
    module_hashes.pop(path, None)
    globals()["_file_hashes"] = module_hashes
    globals()["chunk_store"] = module_store
    if state.EMBED_MODEL:
        _save_cache(state.EMBED_MODEL)
    _refresh_exports()


class SourceHandler(indexer.SourceHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if ext == ".md":
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
            _debounce(src, lambda s=src: reindex_file(s))

    def on_created(self, event):
        if event.is_directory:
            return
        src = event.src_path
        ext = Path(src).suffix.lower()
        if ext == ".md":
            _debounce(src, lambda s=src: refresh_md_file(s))
        elif ext in INDEXABLE_EXTENSIONS:
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
            refresh_md_file(src)
        else:
            refresh_skeleton_only()

    def on_moved(self, event):
        if event.is_directory:
            return
        src, dest = event.src_path, event.dest_path
        src_ext = Path(src).suffix.lower()
        dest_ext = Path(dest).suffix.lower()
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
            _debounce(dest, lambda d=dest: refresh_md_file(d))
        else:
            refresh_skeleton_only()


class _ClaudeLightModule(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module_obj, attr in _MODULE_ATTR_MAP.get(name, []):
            setattr(module_obj, attr, value)


sys.modules[__name__].__class__ = _ClaudeLightModule
_refresh_exports()

__all__ = [name for name in globals() if not name.startswith("__")]
