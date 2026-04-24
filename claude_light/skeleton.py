import hashlib
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from claude_light.config import SKIP_DIRS
import claude_light.state as state

_FULL_MD_NAMES = {"claude.md", "agents.md"}
_FULL_MD_LIMIT = 5_000
_COMPACT_MD_LIMIT = 1_200
_SMALL_MD_LIMIT = 1_500

# Cached directory walk result to avoid duplicate rglob calls
_cached_paths = None
_cached_cwd = None

def _file_hash(path: Path) -> str:
    """MD5 of file bytes — fast change detection, not cryptographic."""
    return hashlib.md5(path.read_bytes()).hexdigest()

def _stat_tuple(path: Path):
    """Return (mtime, size) or None if the file is unreadable."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime, st.st_size)

def _file_hash_parallel(
    paths: list[Path],
    prev_hashes: dict | None = None,
    prev_stats: dict | None = None,
) -> tuple[dict, dict]:
    """Compute file hashes in parallel, reusing cached hashes when (mtime, size) matches.

    Returns (hashes, stats) where stats is {path: [mtime, size]} for persistence.
    """
    prev_hashes = prev_hashes or {}
    prev_stats = prev_stats or {}

    def hash_one(p: Path):
        key = str(p)
        stat = _stat_tuple(p)
        if stat is None:
            return (key, None, None)
        prev_stat = prev_stats.get(key)
        if prev_stat and tuple(prev_stat) == stat and key in prev_hashes:
            return (key, prev_hashes[key], list(stat))
        return (key, _file_hash(p), list(stat))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(hash_one, paths))

    hashes = {k: h for k, h, _ in results if h is not None}
    stats = {k: s for k, _, s in results if s is not None}
    return hashes, stats

def _is_skipped(path):
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)

def _get_cached_paths():
    """Get cached directory paths, computing them once per session if needed."""
    global _cached_paths, _cached_cwd
    
    current_cwd = os.getcwd()
    
    # Invalidate cache if working directory changed
    if _cached_cwd != current_cwd:
        _cached_paths = None
        _cached_cwd = current_cwd
    
    if _cached_paths is None:
        all_paths = []
        for path in sorted(Path(".").rglob("*")):
            if not _is_skipped(path):
                all_paths.append(path)
        _cached_paths = all_paths
    
    return _cached_paths

def _invalidate_path_cache():
    """Invalidate the cached paths, forcing a fresh walk on next access."""
    global _cached_paths
    _cached_paths = None

def _build_compressed_tree(paths):
    """
    Build a compressed directory tree from an already-filtered list of Paths.
    """
    root_node = {}
    root = Path(".")
    for path in paths:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            # Path is not relative to root (e.g., "." or absolute paths)
            continue
        if not parts:
            # Skip empty paths (e.g., "." relative to ".")
            continue
        node  = root_node
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        last = parts[-1]
        if path.is_dir():
            node.setdefault(last, {})
        else:
            node.setdefault(last, None)

    lines = []
    _render_compressed_node(root_node, lines, "")
    return "Directory structure:\n" + "\n".join(lines)

def _render_compressed_node(node, lines, indent):
    dirs  = {k: v for k, v in node.items() if isinstance(v, dict)}
    files = [k for k, v in node.items() if v is None]

    for dirname in sorted(dirs):
        children   = dirs[dirname]
        compressed = dirname
        current    = children
        while len(current) == 1:
            only_key, only_val = next(iter(current.items()))
            if isinstance(only_val, dict):
                compressed += "/" + only_key
                current = only_val
            else:
                break
        lines.append(f"{indent}{compressed}/")
        _render_compressed_node(current, lines, indent + "  ")

    by_ext = {}
    no_ext = []
    for fname in files:
        ext = Path(fname).suffix.lower()
        if ext:
            by_ext.setdefault(ext, []).append(Path(fname).stem)
        else:
            no_ext.append(fname)

    for fname in sorted(no_ext):
        lines.append(f"{indent}{fname}")
    for ext in sorted(by_ext):
        stems = sorted(by_ext[ext])
        if len(stems) == 1:
            lines.append(f"{indent}{stems[0]}{ext}")
        else:
            lines.append(f"{indent}{{{','.join(stems)}}}{ext}")

def _render_md_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if path.name.lower() in _FULL_MD_NAMES:
            pass
        elif len(text) > _FULL_MD_LIMIT and not _looks_structured_md(text):
            text = text[:_FULL_MD_LIMIT] + "\n\n... [TRUNCATED due to length]"
        elif len(text) > _SMALL_MD_LIMIT:
            text = _compact_md_text(text, limit=_COMPACT_MD_LIMIT)
        return f"<!-- {path} -->\n{text}" if text else ""
    except OSError:
        return ""


def _looks_structured_md(text: str) -> bool:
    return any(marker in text for marker in ("\n# ", "\n##", "\n###", "\n- ", "\n* ", "\n1. "))


def _compact_md_text(text: str, limit: int = _COMPACT_MD_LIMIT) -> str:
    lines = text.splitlines()
    kept = []
    saw_body = False
    blank_pending = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            blank_pending = bool(kept)
            continue

        keep = False
        if stripped.startswith(("#", "##", "###")):
            keep = True
        elif stripped.startswith(("- ", "* ", "1. ", "2. ", "3. ")):
            keep = True
        elif stripped.startswith("```"):
            continue
        elif not saw_body:
            keep = True
            saw_body = True
        elif len(kept) < 18:
            keep = True

        if not keep:
            continue

        if blank_pending and kept:
            kept.append("")
            blank_pending = False
        kept.append(line)

        if len("\n".join(kept)) >= limit:
            break

    compact = "\n".join(kept).strip()
    if len(compact) < len(text):
        compact += "\n\n... [COMPACTED markdown excerpt]"
    return compact

def _assemble_skeleton() -> str:
    docs = "\n\n".join(v for v in state._skeleton_md_parts.values() if v)
    return state._skeleton_tree + ("\n\n" + docs if docs else "")

def _refresh_single_md(path_str: str) -> bool:
    p = Path(path_str)
    if not p.exists():
        changed = path_str in state._skeleton_md_parts
        state._skeleton_md_hashes.pop(path_str, None)
        state._skeleton_md_parts.pop(path_str, None)
        return changed
    h = _file_hash(p)
    if state._skeleton_md_hashes.get(path_str) == h:
        return False
    state._skeleton_md_hashes[path_str] = h
    state._skeleton_md_parts[path_str] = _render_md_file(p)
    return True

def _refresh_tree_only() -> str:
    all_paths = _get_cached_paths()
    state._skeleton_tree = _build_compressed_tree(all_paths)
    return _assemble_skeleton()

def build_skeleton() -> str:
    all_paths = _get_cached_paths()
    
    # Separate files by type for skeleton tree and markdown processing
    md_files = [p for p in all_paths if p.suffix == ".md" and p.is_file()]

    state._skeleton_tree = _build_compressed_tree(all_paths)

    # Parallel hash computation for markdown files (skip re-read when stat matches)
    if md_files:
        new_hashes, new_stats = _file_hash_parallel(
            md_files, state._skeleton_md_hashes, state._skeleton_md_stats
        )
    else:
        new_hashes, new_stats = {}, {}

    new_parts = {}
    for path in md_files:
        path_str = str(path)
        h = new_hashes[path_str]
        if state._skeleton_md_hashes.get(path_str) == h and path_str in state._skeleton_md_parts:
            new_parts[path_str] = state._skeleton_md_parts[path_str]
        else:
            new_parts[path_str] = _render_md_file(path)

    state._skeleton_md_hashes = new_hashes
    state._skeleton_md_stats  = new_stats
    state._skeleton_md_parts  = new_parts

    return _assemble_skeleton()
