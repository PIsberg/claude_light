# Claude Light: Token Usage Architecture

This document outlines the primary mechanisms and architectural decisions implemented in `claude_light.py` to massively reduce LLM token consumption. The strategies are ordered by their relative significance and impact on token cost.

---

## 1. RAG Method-Level Chunking & Retrieval
**Significance:** Absolute highest impact. Reduces the baseline context payload from "entire source system" to "only relevant methods".

### Mechanism
Rather than feeding Claude an entire codebase for every question, `claude_light` uses **Tree-sitter** to parse ASTs of source files. It splits the code strictly into method-level chunks while retaining the file's imports and class headers. When the user asks a question, only the highest-scoring chunks mathematically similar to the query are retrieved and injected as context.

### Code: `_chunk_with_treesitter` (line 622)
This is the core chunker. For each source file it runs the tree-sitter parser, walks the AST for function/method nodes, and emits one chunk per symbol. Every chunk is prefixed with a `// {filepath}` comment and the file preamble (package declarations, imports, class header) so it is fully self-contained for embedding — the model never sees a method body without knowing which class it belongs to.

```python
def _chunk_with_treesitter(filepath, source, language, node_types):
    src_bytes = bytes(source, "utf-8")
    parser    = TSParser(language)
    tree      = parser.parse(src_bytes)

    symbols = []
    _walk(tree.root_node, node_types, symbols)

    if not symbols:
        return [{"id": filepath, "text": source}]   # whole-file fallback

    lines          = source.splitlines(keepends=True)
    first_sym_line = symbols[0].start_point[0]
    preamble       = "".join(lines[:first_sym_line]).rstrip()

    chunks = []
    seen: dict = {}
    for node in symbols:
        name = _extract_symbol_name(node)
        # numeric suffix deduplicates overloaded method names
        if name in seen:
            seen[name] += 1
            uid = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
            uid = name

        node_text  = source[node.start_byte:node.end_byte]
        chunk_text = (
            f"// {filepath}\n"
            + preamble + "\n"
            + "    // ...\n"
            + node_text.strip() + "\n"
        )
        chunks.append({"id": f"{filepath}::{uid}", "text": chunk_text})

    return chunks
```

Chunk IDs use `filepath::methodName`. The `::` separator is guaranteed not to appear in real source identifiers, making it safe to split on later. If no symbols are found (e.g. a file with only top-level constants) the function falls back to a single whole-file chunk.

### Code: `auto_tune` (line 681)
Called twice during startup — once before the cache is loaded (to select the embedding model by file count) and once after all chunks are built (to set `TOP_K` from actual average chunk size). This avoids hard-coding a `TOP_K` that would be wrong for both tiny and enormous repos.

```python
def auto_tune(source_files, chunks=None):
    n = len(source_files)
    if n < 50:
        chosen_model = "all-MiniLM-L6-v2"       # 22 MB  — fast startup
    elif n < 200:
        chosen_model = "all-mpnet-base-v2"       # 420 MB — better depth
    else:
        chosen_model = "nomic-ai/nomic-embed-text-v1.5"  # best for large repos

    # TOP_K = how many chunks fit inside TARGET_RETRIEVED_TOKENS
    avg_tokens = max(1, (total_chars // n_units) // 4)
    TOP_K      = max(2, min(15, round(TARGET_RETRIEVED_TOKENS / avg_tokens)))
```

### Code: `retrieve` (line 987)
Retrieval is a single matrix multiply — no per-chunk Python loop. Two filtering stages run after the scores are computed.

```python
def retrieve(query, token_budget=None):
    budget = token_budget or TARGET_RETRIEVED_TOKENS
    k = max(2, round(base_k * budget / TARGET_RETRIEVED_TOKENS))  # scale k to budget

    embs   = np.stack([chunk_store[cid]["emb"] for cid in ids])
    q_emb  = embedder.encode(query_prefix + query, normalize_embeddings=True)
    scores = embs @ q_emb   # (n_chunks,) — one vectorised op

    # Stage 1: absolute floor — discard anything below MIN_SCORE (0.45)
    top_pairs = [(ids[i], float(scores[i]))
                 for i in np.argsort(-scores)
                 if scores[i] >= MIN_SCORE][:k]

    # Stage 2: relative floor — discard stragglers far below the best match
    if top_pairs:
        threshold = top_pairs[0][1] * RELATIVE_SCORE_FLOOR   # e.g. 0.91 × 0.60 = 0.546
        top_pairs = [(cid, s) for cid, s in top_pairs if s >= threshold]
```

The `token_budget` parameter is set by `route_query()` before `retrieve()` is called — so a simple "list X" query routed to `low` effort retrieves at most 1,500 tokens of chunks rather than the full 6,000. `k` is scaled proportionally from the auto-tuned `TOP_K` so the ratio of chunks-to-budget stays consistent regardless of effort level.

### Estimated Token Savings
- **90% – 99%+ per query**
- **Example:** In an "extra-large" project of 1,000 files yielding ~900,000 tokens, injecting the top 15 relevant functional chunks takes the query context down to ~1,300 tokens. This alone prevents the query cost from exceeding $2.75 *per turn*, dropping it to $0.0004.

---

## 2. Dynamic Prefix Caching Re-Architecture
**Significance:** Extreme compounding impact over long, multi-turn conversations.

### Mechanism
Anthropic's Prompt Caching reads tokens sequentially from the start of the `messages` chain and breaks the cache the moment it hits a modified sequence. The architecture places three `ephemeral` cache breakpoints at stable boundaries so that per-turn cost only covers the truly new tokens.

| Breakpoint | Location | Stable across |
|---|---|---|
| ① | End of `skeleton` system block | Entire session |
| ② | End of `conversation_history` (last stored turn) | All but the newest turn |
| ③ | End of retrieved RAG chunks in the current user message | Consecutive turns retrieving the same chunks |

### Code: `_build_system_blocks` (line 1032) — Breakpoint ①
The `SYSTEM_PROMPT` and the skeleton (directory tree + `.md` files) are packed into a two-element system array. The `cache_control` marker on the skeleton block tells Anthropic to cache everything up to and including that block.

```python
def _build_system_blocks(skeleton):
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton, "cache_control": {"type": "ephemeral"}},
    ]
    return blocks
```

`SYSTEM_PROMPT` has no marker of its own — it doesn't need one because it sits before the skeleton's breakpoint and is therefore covered by it automatically.

### Code: `chat()` history cache control (line 1430) — Breakpoint ②
Before assembling the final messages list, the last entry in the trimmed conversation history is converted from a plain string into a content-block array with an `ephemeral` marker. This creates a stable cache boundary at the end of all previously-seen history, so the entire conversation context (minus the current turn) is read from cache on the next call.

```python
if trimmed:
    last_msg = trimmed[-1]
    if isinstance(last_msg["content"], str):
        trimmed[-1] = {
            "role": last_msg["role"],
            "content": [{"type": "text", "text": last_msg["content"],
                         "cache_control": {"type": "ephemeral"}}]
        }
```

### Code: `chat()` retrieved-chunk cache control (line 1442) — Breakpoint ③
The current user message is structured as a two-element content-block array rather than a plain string. The retrieved chunks occupy the first block (with `cache_control`); the bare question text is the second block (no marker). On the next turn, if `retrieve()` returns identical chunks, the entire chunk block is a cache hit and only the new question pays full input price.

```python
current_blocks: list = []
if retrieved_ctx:
    current_blocks.append({
        "type": "text",
        "text": f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n",
        "cache_control": {"type": "ephemeral"},
    })
current_blocks.append({"type": "text", "text": f"Question:\n{query}"})
messages = trimmed + [{"role": "user", "content": current_blocks}]
```

### Estimated Token Savings
- **50% - 95% discount per turn for historical conversation tokens**
- **Example:** Re-sending 10,000 tokens of conversational history previously cost full input pricing ($3.00 / 1M). With a stabilized cache prefix hitting 100% of the conversational history and skeleton, those tokens are billed at the cached rate ($0.30 / 1M) — a 90% cost drop.

---

## 3. Scrubbing Assistant-Generated Code Blocks from History
**Significance:** Critical for any session employing edit instructions.

### Mechanism
When the user asks Claude to edit a file, the assistant replies with full-file code blocks. Storing those raw blocks in `conversation_history` would permanently bloat every subsequent turn. The script strips them before committing to history and replaces them with a compact marker.

### Code: `_ANY_BLOCK` regex (line 194)
A single compiled pattern matches any fenced block that carries a file-path tag — both the SEARCH/REPLACE edit format and plain new-file blocks.

```python
_ANY_BLOCK = re.compile(r"```[a-zA-Z]*:([^\n`]+)\n(.*?)```", re.DOTALL)
```

The pattern captures the file path in group 1 and the block body in group 2. The `re.DOTALL` flag is essential — without it `.` would not match newlines and the pattern would fail on any multi-line code block.

### Code: `clean_reply` assembly in `chat()` (line 1470)
After the API response arrives, `parse_edit_blocks()` extracts the edit instructions for `apply_edits()`. The reply stored in `conversation_history` is then built separately: all fenced blocks are removed by `_ANY_BLOCK.sub("", reply)`, and a single `[Files edited: ...]` line is appended in their place.

```python
edits = parse_edit_blocks(reply)
if edits:
    labels     = ", ".join(e["path"] for e in edits)
    clean_reply = _ANY_BLOCK.sub("", reply).strip()
    clean_reply = (clean_reply + f"\n[Files edited: {labels}]").strip()
else:
    clean_reply = reply
conversation_history.append({"role": "assistant", "content": clean_reply})
```

The full `reply` is still used for `apply_edits()` and for printing to the terminal — only the stored copy is stripped. This means the user sees the full diff output but future API calls never carry the code payload.

### Estimated Token Savings
- **1,000 – 10,000+ tokens saved sequentially per edit turn**
- **Example:** Editing a 2,000 token file 5 times would normally compound into passing 10,000 useless tokens to the LLM on the 6th turn. This prevents exponential bloat during refactoring loops.

---

## 4. Bounded Markdown Ingestion (`build_skeleton`)
**Significance:** Highly effective for repositories with large documentation bases.

### Mechanism
`build_skeleton()` recursively explores the source tree, reads `.md` files, and includes them in the cached skeleton so Claude has high-level architectural context. Without a size cap, a single verbose changelog or vendor manual could balloon the skeleton by tens of thousands of tokens.

### Code: `_render_md_file` (line 525)
Every `.md` file passes through this function before entering the skeleton. Files over 5,000 characters are hard-truncated — except `CLAUDE.md` and `agents.md`, which hold operational rules that must never be trimmed.

```python
def _render_md_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > 5000 and path.name.lower() not in ("claude.md", "agents.md"):
            text = text[:5000] + "\n\n... [TRUNCATED due to length]"
        return f"<!-- {path} -->\n{text}" if text else ""
    except OSError:
        return ""
```

The exemption list is checked by `path.name.lower()` — it matches regardless of platform case conventions.

### Code: `build_skeleton` (line 567)
A single `rglob("*")` pass collects all non-skipped paths and all `.md` files. Each `.md` file is hash-checked against `_skeleton_md_hashes`; if the MD5 is unchanged, the previously rendered string is reused from `_skeleton_md_parts` without re-reading or re-rendering the file.

```python
def build_skeleton() -> str:
    for path in sorted(Path(".").rglob("*")):
        if _is_skipped(path):
            continue
        all_paths.append(path)
        if path.suffix == ".md" and path.is_file():
            md_files.append(path)

    _skeleton_tree = _build_compressed_tree(all_paths)

    for path in md_files:
        h = _file_hash(path)
        if _skeleton_md_hashes.get(path_str) == h and path_str in _skeleton_md_parts:
            new_parts[path_str] = _skeleton_md_parts[path_str]  # cache hit
        else:
            new_parts[path_str] = _render_md_file(path)         # re-render
```

This means that after the first startup, a watchdog-triggered skeleton refresh only re-renders the specific `.md` file that changed — all others are retrieved from the in-memory part cache.

### Estimated Token Savings
- **Variable (10,000 – 200,000+ tokens) depending on repository hygiene**
- **Example:** A project with 100,000 tokens worth of `.md` documentation is bounded to ~1,000 tokens max per file, immediately capping the potential skeleton leak.

---

## 5. "Heartbeat" Auto-Warmer
**Significance:** Prevents expensive "cold-start" query penalties when working intermittently.

### Mechanism
Anthropic's token caching system retains context ephemerally for approximately 5 minutes after the last query. Exceeding this window forces the very next query to be treated as a cold-start, billing the entire skeleton and conversation history at the full (10× higher) input token price.

### Code: `warm_cache` (line 1054)
Sends a minimal API call — `max_tokens=1`, throwaway message `"ok"` — carrying the full system blocks with cache control. This costs only the cache-write tokens for the skeleton on the first call, then essentially nothing on subsequent calls (the cache is already warm). The call uses `MODEL` (Sonnet) rather than Haiku because Anthropic's cache is keyed per model; warming with the wrong model would not help subsequent Sonnet/Opus calls.

```python
def warm_cache():
    with lock:
        skeleton = skeleton_context
    response = client.messages.create(
        model=MODEL,
        max_tokens=1,
        system=_build_system_blocks(skeleton),
        messages=[{"role": "user", "content": "ok"}],
    )
```

### Code: `heartbeat` (line 1190)
A daemon thread that wakes every `HEARTBEAT_SECS` (30 s). If it detects `idle > CACHE_TTL_SECS` (240 s = 4 min), it calls `warm_cache()`. The 4-minute threshold gives a 60-second safety margin before Anthropic's 5-minute TTL would expire.

```python
def heartbeat():
    while not stop_event.is_set():
        stop_event.wait(HEARTBEAT_SECS)   # sleep 30 s
        with lock:
            idle = time.time() - last_interaction
        if idle > CACHE_TTL_SECS and not stop_event.is_set():
            warm_cache()
```

The thread is started as a `daemon=True` thread so it is killed automatically when the main process exits — no explicit cleanup is needed.

### Estimated Token Savings
- **90% discount retained continuously between idle periods**
- **Example:** Returning after 15 minutes natively forces a re-read of a 50,000 token skeleton ($0.15). The heartbeat prevents this expiration entirely, keeping the subsequent interaction at the cached price ($0.015).

---

## 6. Smart Skeleton Compression
**Significance:** High — reduces the skeleton itself before it ever reaches the cache.

### Mechanism
The directory tree emitted into the skeleton system prompt applies two lossless compressions before the string is built.

### Code: `_build_compressed_tree` (line 453)
Converts the flat list of `Path` objects from `build_skeleton()` into a nested dict tree, then delegates rendering to `_render_compressed_node`.

```python
def _build_compressed_tree(paths):
    root_node = {}
    for path in paths:
        parts = path.relative_to(root).parts
        node  = root_node
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        last = parts[-1]
        if path.is_dir():
            node.setdefault(last, {})
        else:
            node.setdefault(last, None)   # None marks a file leaf

    lines = []
    _render_compressed_node(root_node, lines, "")
    return "Directory structure:\n" + "\n".join(lines)
```

Files are stored as `None` values; directories are stored as nested dicts. This distinction drives the two compressions in the renderer.

### Code: `_render_compressed_node` (line 485)
**Chain collapse:** while a directory has exactly one child that is also a directory, it appends that child's name with a `/` separator instead of adding a new indented line. This turns `main/ → java/ → com/ → example/` into a single `main/java/com/example/` entry.

**Brace grouping:** sibling files are bucketed by extension. If a bucket has more than one stem, they are joined into a single `{Stem1,Stem2}.ext` line.

```python
def _render_compressed_node(node, lines, indent):
    dirs  = {k: v for k, v in node.items() if isinstance(v, dict)}
    files = [k for k, v in node.items() if v is None]

    for dirname in sorted(dirs):
        children   = dirs[dirname]
        compressed = dirname
        current    = children
        # collapse single-child directory chains
        while len(current) == 1:
            only_key, only_val = next(iter(current.items()))
            if isinstance(only_val, dict):
                compressed += "/" + only_key
                current = only_val
            else:
                break
        lines.append(f"{indent}{compressed}/")
        _render_compressed_node(current, lines, indent + "  ")

    # brace-group sibling files by extension
    for ext in sorted(by_ext):
        stems = sorted(by_ext[ext])
        if len(stems) == 1:
            lines.append(f"{indent}{stems[0]}{ext}")
        else:
            lines.append(f"{indent}{{{','.join(stems)}}}{ext}")
```

Both transforms are purely cosmetic — no information is lost and Claude reads brace/path notation fluently.

### Estimated Token Savings
- **30–50% reduction in skeleton tree tokens**
- **Example:** A typical Java microservice with 5 deeply-nested packages and 40 source files might produce a 600-token tree naively. After compression the same tree fits in ~300 tokens.

---

## 7. Retrieved-Chunk Deduplication
**Significance:** Medium-High — eliminates repeated preamble when multiple methods from the same file rank highly.

### Mechanism
Each method-level chunk embeds a full file preamble to keep it self-contained for embedding quality. When several chunks from the same file are retrieved, naïvely concatenating them repeats the identical preamble for every method.

### Code: `_dedup_retrieved_context` (line 933)
Groups the ranked chunk pairs by source file. For each file it emits the preamble exactly once, then appends each retrieved method body beneath a `// methodName` comment. Whole-file fallback chunks (no `::` in their ID) are included verbatim.

```python
def _dedup_retrieved_context(top_pairs):
    _SEP = "\n    // ...\n"   # separator between preamble and body in each chunk

    file_order, file_data, whole_files = [], {}, []

    for chunk_id, _score in top_pairs:
        text = chunk_store[chunk_id]["text"]
        if "::" not in chunk_id:
            whole_files.append(_strip_comments(text, ext))
            continue
        filepath, method_name = chunk_id.rsplit("::", 1)
        if _SEP in text:
            header, body = text.split(_SEP, 1)
        else:
            header, body = f"// {filepath}", text
        if filepath not in file_data:
            file_order.append(filepath)
            file_data[filepath] = {"header": header, "methods": []}
        file_data[filepath]["methods"].append(
            (method_name, _strip_comments(body, ext).strip())
        )

    parts = whole_files[:]
    for filepath in file_order:
        d = file_data[filepath]
        method_blocks = [f"    // {name}\n{body}" for name, body in d["methods"]]
        parts.append(d["header"] + "\n" + "\n\n".join(method_blocks))

    return "\n\n".join(parts)
```

Comments are stripped from method bodies (but not from the preamble) via `_strip_comments()`. The preamble is preserved verbatim because it contains structural information — package names, imports, class signature — that Claude needs for accurate reasoning.

### Estimated Token Savings
- **5–20% reduction in retrieved-context tokens per query**
- **Example:** Three methods from a 30-line preamble file each scoring above `MIN_SCORE` — old approach sent the preamble three times (~90 tokens wasted); deduplication sends it once.

---

## 8. Dynamic Model & Effort Routing
**Significance:** High — prevents paying Sonnet/Opus rates for trivial lookups, and unlocks Opus extended thinking only when it genuinely helps.

### Mechanism
`route_query()` runs a zero-API-cost heuristic classifier before every user turn. It selects a `(model, effort, max_tokens)` triple and — critically — runs before `retrieve()` so the retrieval budget is already known at fetch time.

| Effort | Model | `max_tokens` | Retrieval budget | Thinking |
|---|---|---|---|---|
| `low` | `claude-haiku-4-5` | 2 048 | 1 500 tokens | off |
| `medium` | `claude-sonnet-4-6` | 4 096 | 3 000 tokens | off |
| `high` | `claude-sonnet-4-6` | 8 192 | 6 000 tokens | off |
| `max` | `claude-opus-4-6` | 16 000 | 9 000 tokens | on (`budget_tokens=10 000`) |

### Code: `route_query` (line 320)
Three keyword signal sets are matched against the lowercased query. Hit counts and word count drive a top-down decision tree — conservatively defaulting to `high/sonnet` when uncertain.

```python
def route_query(query: str) -> tuple[str, str, int]:
    q = query.lower()
    word_count = len(q.split())

    _LOW_SIGNALS  = {"list", "show", "where", "what is", "how many", ...}
    _HIGH_SIGNALS = {"implement", "write", "create", "refactor", "fix", ...}
    _MAX_SIGNALS  = {"architect", "optimize", "trade-off", "evaluate", "comprehensive", ...}

    low_hits  = sum(1 for s in _LOW_SIGNALS  if s in q)
    high_hits = sum(1 for s in _HIGH_SIGNALS if s in q)
    max_hits  = sum(1 for s in _MAX_SIGNALS  if s in q)

    if max_hits >= 2 or (max_hits >= 1 and word_count > 35):
        effort, model, max_tokens = "max",    MODEL_OPUS,    16_000
    elif high_hits >= 1 or word_count > 30:
        effort, model, max_tokens = "high",   MODEL_SONNET,   8_192
    elif low_hits >= 1 and high_hits == 0 and word_count <= 20:
        effort, model, max_tokens = "low",    MODEL_HAIKU,    2_048
    else:
        effort, model, max_tokens = "medium", MODEL_SONNET,   4_096
```

The result is printed to stderr immediately so the user always knows which model is being used before the API call is made.

### Code: `_extract_text` (line 384)
When `effort == "max"`, the `thinking` parameter is added to the API call. The response content then contains both `thinking` blocks and `text` blocks. This helper filters to text only, keeping the rest of the pipeline model-agnostic.

```python
def _extract_text(content_blocks) -> str:
    return "".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )
```

### Code: call site in `chat()` (line 1413)
`route_query()` is called before `retrieve()` specifically so that `_RETRIEVAL_BUDGET[effort]` can be passed straight into the fetch — the routing and retrieval decisions are coupled.

```python
routed_model, effort, max_tok = route_query(query)
retrieved_ctx, hits = retrieve(query, token_budget=_RETRIEVAL_BUDGET[effort])
# ...
create_kwargs = dict(model=routed_model, max_tokens=max_tok, ...)
if effort == "max":
    create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}
response = client.messages.create(**create_kwargs)
reply = _extract_text(response.content)
```

### Estimated Token Savings
- Routing a simple lookup to Haiku instead of Sonnet cuts per-turn cost by ~8× (Haiku: $0.80/M vs Sonnet: $3.00/M input).
- The reduced retrieval budget for `low`/`medium` effort compounds this: fewer chunks are fetched and the third cache breakpoint (section 2) pays off more consistently when the same small chunk set is reused across turns.

---

## 9. Adaptive Retrieval: Token Budget & Relative Score Floor
**Significance:** Medium — eliminates noise chunks and right-sizes the retrieval payload per query, compounding the savings from sections 1 and 8.

### Mechanism
Two independent filters work together inside `retrieve()` to ensure the returned context is as compact and relevant as possible.

**Token budget scaling:** `TOP_K` is fixed by `auto_tune()` for the `high` effort level (6,000-token budget). For other effort levels, `k` is scaled proportionally: `k = max(2, round(TOP_K × budget / TARGET_RETRIEVED_TOKENS))`. A `low` query therefore fetches at most `TOP_K × 1500/6000 = TOP_K/4` chunks.

**Relative score floor:** the absolute `MIN_SCORE = 0.45` floor is necessary but insufficient — it admits chunks that are technically above the threshold but are far below the best match and would only add noise. A second filter drops any chunk whose score falls below `RELATIVE_SCORE_FLOOR × top_score` (0.60 × top score). This means a chunk scoring 0.47 is kept when the top chunk scores 0.55 (both are meaningful), but dropped when the top chunk scores 0.92 (a 0.47 is noise in that context).

### Code: `retrieve` (line 987) — both filters together

```python
def retrieve(query, token_budget=None):
    budget = token_budget or TARGET_RETRIEVED_TOKENS
    k = max(2, round(base_k * budget / TARGET_RETRIEVED_TOKENS))

    scores = embs @ q_emb   # vectorised cosine similarity

    # Stage 1: absolute floor + top-k cap
    top_pairs = [(ids[i], float(scores[i]))
                 for i in np.argsort(-scores)
                 if scores[i] >= MIN_SCORE][:k]

    # Stage 2: relative floor — keeps tight clusters, cuts stragglers
    if top_pairs:
        threshold = top_pairs[0][1] * RELATIVE_SCORE_FLOOR
        top_pairs = [(cid, s) for cid, s in top_pairs if s >= threshold]

    ctx = _dedup_retrieved_context(top_pairs)
    return ctx, top_pairs
```

The two stages are independent and commutative — Stage 1 is applied first only because it is cheaper to compute (it directly uses the sorted scores array). Stage 2 never increases the chunk count; it can only decrease it.

### Estimated Token Savings
- **Budget scaling alone:** a `low/haiku` query retrieves at most 1,500 tokens vs 6,000 — a 75% reduction in retrieved context tokens for that turn.
- **Relative floor alone:** for a focused question with a clear best match (score 0.90+), low-relevance stragglers (scores 0.47–0.53) are eliminated, saving typically 500–1,500 tokens per query.
- **Combined:** on a multi-turn debugging session routed mostly at `high` effort, the relative floor trims the tail consistently, while the budget scaling caps cost on every incidental lookup question in the same session.
