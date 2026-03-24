import hashlib
from pathlib import Path
from claude_light.config import SKIP_DIRS
import claude_light.state as state

def _file_hash(path: Path) -> str:
    """MD5 of file bytes — fast change detection, not cryptographic."""
    return hashlib.md5(path.read_bytes()).hexdigest()

def _is_skipped(path):
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)

def _build_compressed_tree(paths):
    """
    Build a compressed directory tree from an already-filtered list of Paths.
    """
    root_node = {}
    root = Path(".")
    for path in paths:
        parts = path.relative_to(root).parts
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
        if len(text) > 5000 and path.name.lower() not in ("claude.md", "agents.md"):
            text = text[:5000] + "\n\n... [TRUNCATED due to length]"
        return f"<!-- {path} -->\n{text}" if text else ""
    except OSError:
        return ""

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
    all_paths = [p for p in sorted(Path(".").rglob("*")) if not _is_skipped(p)]
    state._skeleton_tree = _build_compressed_tree(all_paths)
    return _assemble_skeleton()

def build_skeleton() -> str:
    all_paths, md_files = [], []
    for path in sorted(Path(".").rglob("*")):
        if _is_skipped(path):
            continue
        all_paths.append(path)
        if path.suffix == ".md" and path.is_file():
            md_files.append(path)

    state._skeleton_tree = _build_compressed_tree(all_paths)

    new_hashes, new_parts = {}, {}
    for path in md_files:
        path_str = str(path)
        h = _file_hash(path)
        new_hashes[path_str] = h
        if state._skeleton_md_hashes.get(path_str) == h and path_str in state._skeleton_md_parts:
            new_parts[path_str] = state._skeleton_md_parts[path_str]
        else:
            new_parts[path_str] = _render_md_file(path)
    state._skeleton_md_hashes = new_hashes
    state._skeleton_md_parts  = new_parts

    return _assemble_skeleton()
