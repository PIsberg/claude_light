"""
Call edges between indexed chunks, read from the chunk text.

A question like "why does X fail" often needs the function that calls X, and
embedding similarity rarely ranks that caller highly. After retrieval picks its
best chunk, `related_chunks()` names the chunks it calls and the chunks that
call it, so retrieval can add them within the token budget.

Edges come from a name match, not from a type-aware resolver: a call `foo(` in
chunk A points at the chunk(s) whose symbol is `foo`. To keep that honest:
  - a definition in the same file wins over one elsewhere;
  - a name defined in more than `_MAX_DEFS` places is treated as too generic
    to follow (`get`, `run`, `__init__`);
  - a chunk never points at itself.

Nothing is stored on disk. The names a chunk calls are cached per chunk id and
keyed on the hash of its text, so an edited chunk is re-read and the index
cache format does not change.
"""

import re

import claude_light.state as state

_SEP = "\n    // ...\n"
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_MAX_DEFS = 3
_KEYWORDS = frozenset({
    "if", "elif", "else", "for", "while", "switch", "case", "catch", "try", "with",
    "return", "yield", "await", "async", "def", "class", "function", "func", "fn",
    "lambda", "new", "throw", "throws", "sizeof", "typeof", "instanceof", "assert",
    "not", "and", "or", "in", "is", "print", "len", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "range", "isinstance", "super", "self", "this",
})

# chunk id -> (hash of its text, names it calls)
_names_cache = {}


def _symbol(chunk_id):
    return chunk_id.rsplit("::", 1)[1] if "::" in chunk_id else None


def _file(chunk_id):
    return chunk_id.split("::", 1)[0]


def called_names(text):
    """Names that appear as `name(` in a chunk's body, minus keywords and builtins."""
    body = text.split(_SEP, 1)[1] if _SEP in text else text
    return frozenset(n for n in _CALL.findall(body) if n not in _KEYWORDS)


def _names_for(chunk_id, text):
    h = hash(text)
    hit = _names_cache.get(chunk_id)
    if hit is not None and hit[0] == h:
        return hit[1]
    names = called_names(text)
    _names_cache[chunk_id] = (h, names)
    return names


def _snapshot():
    with state.lock:
        return {cid: e["text"] for cid, e in state.chunk_store.items()}


def _definitions(texts):
    """symbol name -> chunk ids that define it."""
    defs = {}
    for cid in texts:
        sym = _symbol(cid)
        if sym:
            defs.setdefault(sym, []).append(cid)
    return defs


def _resolve(name, source_id, defs):
    """Chunk ids a call to `name` made from `source_id` can refer to."""
    candidates = [c for c in defs.get(name, ()) if c != source_id]
    if not candidates:
        return []
    same_file = [c for c in candidates if _file(c) == _file(source_id)]
    if same_file:
        return same_file
    return candidates if len(candidates) <= _MAX_DEFS else []


def related_chunks(chunk_id, limit=6):
    """(callees, callers) of `chunk_id`, each a sorted list of chunk ids."""
    texts = _snapshot()
    if chunk_id not in texts or "::" not in chunk_id:
        return [], []
    defs = _definitions(texts)

    callees = set()
    for name in _names_for(chunk_id, texts[chunk_id]):
        callees.update(_resolve(name, chunk_id, defs))

    sym = _symbol(chunk_id)
    callers = set()
    for cid, text in texts.items():
        if cid == chunk_id or "::" not in cid:
            continue
        if sym in _names_for(cid, text) and chunk_id in _resolve(sym, cid, defs):
            callers.add(cid)

    return sorted(callees)[:limit], sorted(callers)[:limit]


def interleaved(chunk_id, limit=6):
    """Callees and callers of `chunk_id` alternating, callee first, no repeats."""
    callees, callers = related_chunks(chunk_id, limit)
    out = []
    for i in range(max(len(callees), len(callers))):
        for group in (callees, callers):
            if i < len(group) and group[i] not in out:
                out.append(group[i])
    return out[:limit]


def prune_cache():
    """Drop cache entries for chunks that no longer exist."""
    with state.lock:
        live = set(state.chunk_store)
    for cid in [c for c in _names_cache if c not in live]:
        _names_cache.pop(cid, None)
