import numpy as np
from pathlib import Path

from claude_light.config import (
    TARGET_RETRIEVED_TOKENS, MIN_SCORE, RELATIVE_SCORE_FLOOR, _QUERY_PREFIX
)
import claude_light.state as state
from claude_light.parsing import _strip_comments

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


def retrieve(query, token_budget=None):
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

    top_pairs = [(ids[i], float(scores[i]))
                 for i in np.argsort(-scores)
                 if scores[i] >= MIN_SCORE][:k]

    if top_pairs:
        threshold = top_pairs[0][1] * RELATIVE_SCORE_FLOOR
        top_pairs = [(cid, s) for cid, s in top_pairs if s >= threshold]

    ctx = _dedup_retrieved_context(top_pairs)
    return ctx, top_pairs
