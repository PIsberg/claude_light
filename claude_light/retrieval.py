import re

import numpy as np
from pathlib import Path

from claude_light.config import (
    TARGET_RETRIEVED_TOKENS, MIN_SCORE, RELATIVE_SCORE_FLOOR, _QUERY_PREFIX,
    FAST_EMBED_MODEL, HYBRID_MIN_SCORE, HYBRID_SCORE_GAP,
)
import claude_light.state as state
from claude_light.parsing import _strip_comments

_SEP = "\n    // ...\n"


def _chunk_id_to_file(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def _chunk_id_to_symbol(chunk_id: str) -> str:
    if "::" in chunk_id:
        return chunk_id.rsplit("::", 1)[1]
    return Path(chunk_id).name


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _compact_code_snippet(text: str, ext: str) -> str:
    lines = _strip_comments(text, ext).splitlines()
    kept = []
    blank_pending = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_pending = bool(kept)
            continue
        if stripped.startswith(("import ", "from ", "package ", "using ", "namespace ")):
            if len(kept) < 3:
                kept.append(line)
            continue
        if blank_pending and kept:
            kept.append("")
            blank_pending = False
        kept.append(line)
        if len(kept) >= 12:
            break
    return "\n".join(kept).strip()


def _render_chunk_summary(chunk_id: str, text: str) -> str:
    filepath = _chunk_id_to_file(chunk_id)
    symbol = _chunk_id_to_symbol(chunk_id)
    ext = Path(filepath).suffix.lower()
    if "::" in chunk_id and _SEP in text:
        _header, body = text.split(_SEP, 1)
    else:
        body = text
    snippet = _compact_code_snippet(body, ext)
    summary_line = snippet.splitlines()[0].strip() if snippet else symbol
    return f"- {filepath} :: {symbol} — {summary_line[:120]}"


def _render_file_summaries(top_pairs, max_files=8, max_symbols_per_file=3):
    if not top_pairs:
        return ""
    grouped = {}
    order = []
    with state.lock:
        for chunk_id, _score in top_pairs:
            filepath = _chunk_id_to_file(chunk_id)
            if filepath not in grouped:
                order.append(filepath)
                grouped[filepath] = []
            if len(grouped[filepath]) >= max_symbols_per_file:
                continue
            text = state.chunk_store.get(chunk_id, {}).get("text", "")
            grouped[filepath].append(_render_chunk_summary(chunk_id, text))
            if len(order) >= max_files and all(len(grouped[p]) >= 1 for p in order):
                break
    lines = ["Relevant Files:"]
    for filepath in order[:max_files]:
        lines.append(f"* {filepath}")
        lines.extend(grouped[filepath])
    return "\n".join(lines)


def _dedup_retrieved_context(top_pairs):
    _SEP = "\n    // ...\n"
    file_order  = []
    file_data   = {}
    whole_files = []

    with state.lock:
        for chunk_id, _score in top_pairs:
            if chunk_id not in state.chunk_store:
                continue
            text = state.chunk_store[chunk_id]["text"]
            if "::" not in chunk_id:
                ext = Path(chunk_id).suffix.lower()
                whole_files.append(_strip_comments(text, ext))
                continue
            filepath, method_name = chunk_id.rsplit("::", 1)
            ext = Path(filepath).suffix.lower()
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


def _adaptive_select_pairs(ids, scores, budget, k):
    ranked = [(ids[i], float(scores[i])) for i in np.argsort(-scores) if scores[i] >= MIN_SCORE]
    if not ranked:
        return []

    threshold = ranked[0][1] * RELATIVE_SCORE_FLOOR
    candidates = [(cid, score) for cid, score in ranked if score >= threshold]
    return _select_within_budget(candidates, budget, k)


def _zscore(scores):
    sd = float(scores.std())
    return (scores - scores.mean()) / sd if sd > 0 else np.zeros_like(scores)


def hybrid_scores(fast_scores, dense_scores=None, fast_min=None, dense_min=None):
    """Fused per-chunk scores for the hybrid index, or None when no chunk
    relates to the query.

    Each model's cosine similarities are z-scored over all chunks and
    averaged (HYBRID_FAST_WEIGHT), so neither model's scale dominates. The
    query counts as unrelated when neither model's best chunk clears its
    floor in HYBRID_MIN_SCORE. Without dense scores (the dense pass is still
    running) the static model's z-scores are used alone.
    """
    from claude_light.config import HYBRID_FAST_WEIGHT
    fast_min = HYBRID_MIN_SCORE[FAST_EMBED_MODEL] if fast_min is None else fast_min
    related = fast_scores.max() >= fast_min
    if dense_scores is not None:
        dense_min = HYBRID_MIN_SCORE.get(state.EMBED_MODEL, MIN_SCORE) if dense_min is None else dense_min
        related = related or dense_scores.max() >= dense_min
    if not related:
        return None
    fused = _zscore(fast_scores)
    if dense_scores is not None:
        fused = HYBRID_FAST_WEIGHT * fused + (1 - HYBRID_FAST_WEIGHT) * _zscore(dense_scores)
    return fused


# Directories that hold tests. A chunk under one of them is a distractor for a
# question about the package: its code reuses the question's words (model,
# effort, budget) and crowds the answer out. Measured on this repository:
# excluding them raised answering-file recall and lowered tokens sent (see
# docs/BENCHMARKS.md, "Test-file exclusion").
_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs"})
_TEST_QUESTION = re.compile(r"\btests?\b|\btesting\b|\bspec\b|\bpytest\b", re.IGNORECASE)


def _is_test_chunk(chunk_id: str) -> bool:
    """True when any directory on the chunk's file path is a test directory."""
    dirs = _chunk_id_to_file(chunk_id).replace("\\", "/").split("/")[:-1]
    return any(d in _TEST_DIRS for d in dirs)


def _exclude_test_chunks(fused, ids, query):
    """Fused scores with test chunks set to -inf, so they are never selected.
    Questions that mention tests keep every chunk. Returns a new array."""
    if _TEST_QUESTION.search(query):
        return fused
    mask = np.array([_is_test_chunk(cid) for cid in ids], dtype=bool)
    return np.where(mask, -np.inf, fused)


def _hybrid_select_pairs(ids, fused, budget, k, gap=None):
    """Chunks within HYBRID_SCORE_GAP of the top fused score, then the usual
    budget and file-diversity selection."""
    gap = HYBRID_SCORE_GAP if gap is None else gap
    order = np.argsort(-fused)
    top = float(fused[order[0]])
    candidates = [(ids[i], float(fused[i])) for i in order if fused[i] >= top - gap]
    return _select_within_budget(candidates, budget, k, strong=lambda s, t: s >= t - gap / 4)


def _select_within_budget(candidates, budget, k, strong=None):
    """Pick chunks from `candidates` (best first): one per file first, then
    extra strong chunks while the token budget allows."""
    if not candidates:
        return []
    strong = strong or (lambda score, top: score >= top * 0.92)

    selected = []
    selected_ids = set()
    selected_files = set()
    spent_tokens = 0
    top_score = candidates[0][1]

    with state.lock:
        # First pass: maximize file diversity.
        for cid, score in candidates:
            file_id = _chunk_id_to_file(cid)
            if file_id in selected_files:
                continue
            text = state.chunk_store[cid]["text"]
            chunk_tokens = _estimate_tokens(text)
            if spent_tokens + chunk_tokens > budget and selected:
                continue
            selected.append((cid, score))
            selected_ids.add(cid)
            selected_files.add(file_id)
            spent_tokens += chunk_tokens
            if len(selected) >= k or spent_tokens >= budget:
                break

        # Second pass: add only very strong extra chunks if there is room.
        for cid, score in candidates:
            if cid in selected_ids or len(selected) >= k:
                continue
            if not strong(score, top_score) or spent_tokens >= budget * 0.9:
                break
            text = state.chunk_store[cid]["text"]
            chunk_tokens = _estimate_tokens(text)
            if spent_tokens + chunk_tokens > budget and selected:
                continue
            selected.append((cid, score))
            spent_tokens += chunk_tokens

    return selected


def _related_pairs(top_pairs, budget, effort):
    """Callers and callees of the best hit that still fit in the token budget.

    Returned with score 0.0: they were not found by similarity. Skipped at low
    effort, whose budget leaves no room, and when CALL_GRAPH is off.
    """
    from claude_light import config, callgraph
    if not top_pairs or effort == "low" or not config.CALL_GRAPH:
        return []
    chosen = {cid for cid, _ in top_pairs}
    with state.lock:
        room = budget - sum(_estimate_tokens(state.chunk_store[c]["text"])
                            for c in chosen if c in state.chunk_store)
    related = []
    for cid in callgraph.interleaved(top_pairs[0][0]):
        if cid in chosen:
            continue
        with state.lock:
            entry = state.chunk_store.get(cid)
        if entry is None:
            continue
        tokens = _estimate_tokens(entry["text"])
        if tokens > room:
            continue
        related.append((cid, 0.0))
        room -= tokens
        if len(related) >= config.CALL_GRAPH_MAX:
            break
    return related


def _finish(top_pairs, budget, effort):
    """Render the context for `top_pairs` plus their related chunks."""
    related = _related_pairs(top_pairs, budget, effort)
    return _render_retrieved_context(top_pairs, effort, related), top_pairs + related


def _render_retrieved_context(top_pairs, effort: str, related=()) -> str:
    if not top_pairs:
        return ""
    if effort == "low":
        return _render_file_summaries(top_pairs, max_files=6, max_symbols_per_file=2)

    detail_limit = 1 if effort == "medium" else 2
    # Related chunks are shown in full: a caller is only useful with its body.
    detail_pairs = top_pairs[:detail_limit] + list(related)
    summary = _render_file_summaries(list(top_pairs) + list(related))
    details = _dedup_retrieved_context(detail_pairs)
    if not details:
        return summary
    return summary + "\n\nDetailed Code Context:\n" + details


def explain_last_retrieval() -> str:
    """Render a human-readable explanation of the most recent query's routing
    and retrieval, for the /why command. Returns a message if nothing is recorded.
    """
    info = state.last_retrieval
    if not info:
        return "  No query has been run yet — ask something first, then /why."

    lines = []
    lines.append(f"  Query   : {info['query']}")
    lines.append(f"  Routed  : {info['model']}  ·  effort={info['effort']}  "
                 f"·  budget={info['token_budget']} tok")

    if info.get("skipped_reason"):
        lines.append(f"  Retrieval skipped: {info['skipped_reason']}")
    else:
        hits = info.get("hits") or []
        if not hits:
            lines.append("  Retrieved: (no chunks cleared the score floor)")
        else:
            top = hits[0][1]
            mode = getattr(state, "last_retrieval_mode", "")
            if mode:
                lines.append(f"  Retrieved {len(hits)} chunk(s) with the {mode} index "
                             f"(fused z-scores, within {HYBRID_SCORE_GAP} of the top):")
            else:
                lines.append(f"  Retrieved {len(hits)} chunk(s) "
                             f"(MIN_SCORE={MIN_SCORE}, floor={RELATIVE_SCORE_FLOOR:.0%} of top):")
            for cid, score in hits:
                rel = (score / top) if top else 0.0
                file = _chunk_id_to_file(cid)
                sym = _chunk_id_to_symbol(cid)
                label = f"{Path(file).name}::{sym}" if "::" in cid else Path(file).name
                lines.append(f"    {score:5.3f}  ({rel:4.0%})  {label}")

    ci = info.get("compression")
    if ci and not ci.get("skipped"):
        pre, post = ci.get("tokens_before"), ci.get("tokens_after")
        lines.append(f"  Compressed: {pre} → {post} tok "
                     f"({ci.get('ratio', 0) * 100:.0f}% kept)")

    r = info.get("routing")
    if r:
        sem_flag = "semantic on" if r.get("semantic_used") else "semantic off"
        lines.append(
            f"  Routing   : arch={r['arch']:.1f}  logic={r['logic']:.1f}  "
            f"infra={r['infra']:.1f}  ({sem_flag})  est≈${r['est_cost']:.4f}"
        )

    return "\n".join(lines)


def record_retrieval(query, model, effort, token_budget, hits,
                     compression=None, skipped_reason=None, routing=None):
    """Store diagnostics for the last query so /why can explain it."""
    state.last_retrieval = {
        "query": query,
        "model": model,
        "effort": effort,
        "token_budget": token_budget,
        "hits": list(hits) if hits else [],
        "compression": compression,
        "skipped_reason": skipped_reason,
        "routing": routing,
    }


def _retrieve_hybrid(query, budget, k, effort):
    with state.lock:
        if not state.chunk_store:
            return "", []
        ids = list(state.chunk_store.keys())
        entries = [state.chunk_store[cid] for cid in ids]
        fast_embs = np.stack([e["emb_fast"] for e in entries])
        dense_embs = np.stack([e["emb"] for e in entries]) if all("emb" in e for e in entries) else None

    fast_scores = fast_embs @ state.fast_embedder.encode(query, normalize_embeddings=True)
    dense_scores = None
    if dense_embs is not None and not state.embedder_ready.is_set():
        # Every dense vector is cached and only the model is still loading
        # (warm start): worth a short wait for the better answer.
        state.embedder_ready.wait(30)
    # Fuse only when every chunk has its dense vector and the model is free
    # within 2 s (a saved file being re-embedded); otherwise static alone.
    if dense_embs is not None and state.embedder_ready.is_set() and state.embedder is not None \
            and state.embedder_lock.acquire(timeout=2):
        try:
            prefix = _QUERY_PREFIX.get(state.EMBED_MODEL, "")
            dense_scores = dense_embs @ state.embedder.encode(prefix + query, normalize_embeddings=True)
        finally:
            state.embedder_lock.release()
    state.last_retrieval_mode = (f"hybrid: static + {state.EMBED_MODEL}" if dense_scores is not None
                                 else "static model only (dense vectors not ready)")

    fused = hybrid_scores(fast_scores, dense_scores)
    if fused is None:
        return "", []
    fused = _exclude_test_chunks(fused, ids, query)
    top_pairs = _hybrid_select_pairs(ids, fused, budget, k)
    return _finish(top_pairs, budget, effort)


def retrieve(query, token_budget=None, effort="medium"):
    budget = token_budget or TARGET_RETRIEVED_TOKENS
    base_k = state.TOP_K or 4
    k = max(2, round(base_k * budget / TARGET_RETRIEVED_TOKENS))

    from claude_light import config
    if config.HYBRID_RETRIEVAL and state.hybrid_index:
        state.fast_ready.wait()  # loading in the background after a warm start
        if state.fast_embedder is None:
            raise RuntimeError("the static embedding model failed to load")
        return _retrieve_hybrid(query, budget, k, effort)

    with state.lock:
        if not state.chunk_store:
            return "", []
        ids  = list(state.chunk_store.keys())
        embs = np.stack([state.chunk_store[cid]["emb"] for cid in ids])

    # On full-cache-hit startup the embedder loads in a background thread —
    # block here (typically already done) before the first query encode.
    state.embedder_ready.wait()
    if state.embedder_load_error is not None:
        raise state.embedder_load_error

    query_prefix = _QUERY_PREFIX.get(state.EMBED_MODEL, "")
    # Serialize with concurrent reindex encodes (watchdog timers) — see
    # state.embedder_lock.
    with state.embedder_lock:
        q_emb    = state.embedder.encode(query_prefix + query, normalize_embeddings=True)
    scores       = embs @ q_emb

    top_pairs = _adaptive_select_pairs(ids, scores, budget, k)
    return _finish(top_pairs, budget, effort)
