import numpy as np
from pathlib import Path

from claude_light.config import (
    TARGET_RETRIEVED_TOKENS, MIN_SCORE, RELATIVE_SCORE_FLOOR, _QUERY_PREFIX
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
    if not candidates:
        return []

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
            if score < top_score * 0.92 or spent_tokens >= budget * 0.9:
                break
            text = state.chunk_store[cid]["text"]
            chunk_tokens = _estimate_tokens(text)
            if spent_tokens + chunk_tokens > budget and selected:
                continue
            selected.append((cid, score))
            spent_tokens += chunk_tokens

    return selected


def _render_retrieved_context(top_pairs, effort: str) -> str:
    if not top_pairs:
        return ""
    if effort == "low":
        return _render_file_summaries(top_pairs, max_files=6, max_symbols_per_file=2)

    detail_limit = 1 if effort == "medium" else 2
    detail_pairs = top_pairs[:detail_limit]
    summary = _render_file_summaries(top_pairs)
    details = _dedup_retrieved_context(detail_pairs)
    if not details:
        return summary
    return summary + "\n\nDetailed Code Context:\n" + details


def retrieve(query, token_budget=None, effort="medium"):
    budget = token_budget or TARGET_RETRIEVED_TOKENS
    base_k = state.TOP_K or 4
    k = max(2, round(base_k * budget / TARGET_RETRIEVED_TOKENS))

    with state.lock:
        if not state.chunk_store:
            return "", []
        ids  = list(state.chunk_store.keys())
        embs = np.stack([state.chunk_store[cid]["emb"] for cid in ids])

    query_prefix = _QUERY_PREFIX.get(state.EMBED_MODEL, "")
    q_emb        = state.embedder.encode(query_prefix + query, normalize_embeddings=True)
    scores       = embs @ q_emb

    top_pairs = _adaptive_select_pairs(ids, scores, budget, k)
    ctx = _render_retrieved_context(top_pairs, effort)
    return ctx, top_pairs
