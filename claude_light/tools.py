"""
Read-only workspace tools the model can call in API-key mode.

In OAuth mode the Claude CLI brings its own tools; in API-key mode a turn used
to be one Messages call with only the retrieved chunks, so the model could not
open a file that retrieval missed. These three tools close that gap without
giving the model any way to change or execute anything: edits still go through
SEARCH/REPLACE blocks, which claude_light lint-checks, previews and commits.

Every path is resolved against the project root and refused if it escapes it.
Output is capped so a single call cannot flood the context window.
"""

import fnmatch
import os
import re
from pathlib import Path

from claude_light.config import SKIP_DIRS

_MAX_READ_LINES = 400
_MAX_OUTPUT_CHARS = 40_000
_MAX_GREP_MATCHES = 100
_MAX_LIST_ENTRIES = 300
_MAX_SCAN_BYTES = 1_000_000  # skip larger files when searching
_MAX_FETCH_IDS = 5
_SKIP = set(SKIP_DIRS) | {".claude_light_cache", ".venv", "venv", "dist"}

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a text file from the project, with line numbers. Use it when the "
            "retrieved context does not contain the code you need. Paths are relative "
            f"to the project root. Returns at most {_MAX_READ_LINES} lines; pass "
            "start_line/end_line (1-based, inclusive) to read another range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root"},
                "start_line": {"type": "integer", "description": "First line to read (1-based)"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "eager_input_streaming": True,
    },
    {
        "name": "fetch_chunk",
        "description": (
            "Fetch indexed code by chunk id: one function or method body, without the "
            "rest of its file. Use it before read_file when you know the symbol you "
            "need, since it costs a fraction of the tokens. An id is 'path::symbol' as "
            "shown in the retrieved context (e.g. 'claude_light/llm.py::route_query'); a "
            "bare symbol name also works when it is unambiguous. Each result lists the "
            f"chunks the symbol calls and is called by. Takes at most {_MAX_FETCH_IDS} ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk ids ('path::symbol') or bare symbol names",
                },
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
        "eager_input_streaming": True,
    },
    {
        "name": "grep",
        "description": (
            "Search the project's text files for a Python regular expression and return "
            "matching lines as path:line: text. Use it to find where something is defined "
            "or used. Version-control, dependency and build directories are skipped. "
            f"Returns at most {_MAX_GREP_MATCHES} matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "glob": {"type": "string", "description": "Only search paths matching this glob, e.g. '*.py' or 'src/*'"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        "eager_input_streaming": True,
    },
    {
        "name": "list_files",
        "description": (
            "List the files under a project directory, recursively, as paths relative to "
            f"the project root. Returns at most {_MAX_LIST_ENTRIES} entries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to the project root (default: the root)"},
            },
            "additionalProperties": False,
        },
        "eager_input_streaming": True,
    },
]


class _ToolError(Exception):
    """A problem to report back to the model as an error result."""


def run_tool(name, tool_input, root=None):
    """Run one tool call. Returns (output_text, is_error); never raises."""
    root = Path(root or os.getcwd()).resolve()
    try:
        if not isinstance(tool_input, dict):
            raise _ToolError("tool input must be a JSON object")
        if name == "read_file":
            return _read_file(root, tool_input), False
        if name == "fetch_chunk":
            return _fetch_chunk(tool_input), False
        if name == "grep":
            return _grep(root, tool_input), False
        if name == "list_files":
            return _list_files(root, tool_input), False
        raise _ToolError(f"unknown tool {name!r}")
    except _ToolError as e:
        return f"Error: {e}", True
    except Exception as e:  # unexpected I/O problems still go back to the model
        return f"Error: {type(e).__name__}: {e}", True


def _str_arg(tool_input, key, required=True):
    value = tool_input.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _ToolError(f"'{key}' must be a non-empty string")
    return value.strip()


def _int_arg(tool_input, key):
    value = tool_input.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _ToolError(f"'{key}' must be a positive integer")
    return value


def _inside(root, rel):
    """Resolve `rel` under root; refuse anything that escapes it."""
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise _ToolError(f"{rel!r} is outside the project")
    return target


def _rel(root, path):
    return path.relative_to(root).as_posix()


def _skipped(rel_parts):
    return any(p in _SKIP or (p.startswith(".") and p not in (".", "..")) for p in rel_parts)


def _walk_files(root, base):
    # os.walk does not descend into symlinked directories, but it does list
    # symlinked files; one pointing outside the project (say at ~/.ssh) must
    # not be readable through grep or visible through list_files.
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = Path(dirpath).relative_to(root).parts
        dirnames[:] = sorted(d for d in dirnames if not _skipped(rel_dir + (d,)))
        for fn in sorted(filenames):
            if _skipped(rel_dir + (fn,)):
                continue
            path = Path(dirpath) / fn
            try:
                real = path.resolve()
            except OSError:
                continue
            if root in real.parents:
                yield path


def _cap(text):
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n… (output truncated)"


def _read_file(root, tool_input):
    rel = _str_arg(tool_input, "path")
    start = _int_arg(tool_input, "start_line") or 1
    end = _int_arg(tool_input, "end_line")
    path = _inside(root, rel)
    if not path.is_file():
        raise _ToolError(f"{rel!r} not found")
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise _ToolError(f"{rel!r} is a binary file")
    lines = data.decode("utf-8", errors="replace").splitlines()
    if end is None or end > start + _MAX_READ_LINES - 1:
        end = min(len(lines), start + _MAX_READ_LINES - 1)
    if start > len(lines):
        return f"{_rel(root, path)} has {len(lines)} lines"
    body = "\n".join(f"{n}\t{lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1))
    more = f"\n… ({len(lines) - end} more lines; pass start_line={end + 1})" if end < len(lines) else ""
    return _cap(body + more)


def _find_chunks(wanted, ids):
    """Chunk ids matching `wanted`: an exact id, else any id ending '::wanted'
    or '/wanted' (a symbol name, or a path suffix plus symbol)."""
    if wanted in ids:
        return [wanted]
    tail = wanted.rsplit("::", 1)[-1] if "::" in wanted else None
    found = [c for c in ids if c.endswith("::" + wanted) or c.endswith("/" + wanted)
             or c.endswith("\\" + wanted)]
    if not found and tail is not None:
        # 'dir/file.py::name' written with another path form: match file suffix + symbol
        head = wanted.rsplit("::", 1)[0].replace("\\", "/")
        found = [c for c in ids if "::" in c and c.rsplit("::", 1)[1] == tail
                 and c.split("::", 1)[0].replace("\\", "/").endswith(head)]
    return sorted(found)


def _fetch_chunk(tool_input):
    from claude_light import callgraph
    import claude_light.state as state

    wanted = tool_input.get("ids")
    if isinstance(wanted, str):
        wanted = [wanted]
    if not isinstance(wanted, list) or not wanted or not all(isinstance(w, str) and w.strip() for w in wanted):
        raise _ToolError("'ids' must be a non-empty list of strings")
    wanted = [w.strip() for w in wanted[:_MAX_FETCH_IDS]]

    with state.lock:
        texts = {cid: e["text"] for cid, e in state.chunk_store.items()}
    if not texts:
        raise _ToolError("the code index is not ready; use read_file instead")

    parts = []
    for w in wanted:
        found = _find_chunks(w, list(texts))
        if not found:
            parts.append(f"## {w}\nNo such chunk. Use grep to find the symbol, or read_file.")
            continue
        if len(found) > 1:
            parts.append(f"## {w}\nAmbiguous; pass one of: " + ", ".join(found[:10]))
            continue
        cid = found[0]
        text = texts[cid]
        body = text.split(callgraph._SEP, 1)[1] if callgraph._SEP in text else text
        callees, callers = callgraph.related_chunks(cid, limit=8)
        links = []
        if callees:
            links.append("Calls: " + ", ".join(callees))
        if callers:
            links.append("Called by: " + ", ".join(callers))
        parts.append("\n".join([f"## {cid}", body.strip()] + links))
    return _cap("\n\n".join(parts))


def _grep(root, tool_input):
    pattern = _str_arg(tool_input, "pattern")
    glob = _str_arg(tool_input, "glob", required=False)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise _ToolError(f"invalid regular expression: {e}")
    matches = []
    for path in _walk_files(root, root):
        rel = _rel(root, path)
        if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(path.name, glob)):
            continue
        try:
            if path.stat().st_size > _MAX_SCAN_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                matches.append(f"{rel}:{n}: {line.strip()[:200]}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    return _cap("\n".join(matches) + f"\n… (stopped at {_MAX_GREP_MATCHES} matches)")
    return _cap("\n".join(matches)) if matches else "No matches."


def _list_files(root, tool_input):
    rel = _str_arg(tool_input, "path", required=False) or "."
    base = _inside(root, rel)
    if not base.is_dir():
        raise _ToolError(f"{rel!r} is not a directory")
    entries = []
    for path in _walk_files(root, base):
        entries.append(_rel(root, path))
        if len(entries) >= _MAX_LIST_ENTRIES:
            entries.append(f"… (stopped at {_MAX_LIST_ENTRIES} entries)")
            break
    return "\n".join(entries) if entries else "(empty)"
