import sys
import os
import time
import json
import subprocess
import re
import shutil
import threading
from pathlib import Path

import claude_light.config as config
from claude_light.config import (
    API_KEY, AUTH_MODE, API_KEY_SOURCE, ECONOMY_MODE, MODEL, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS, MODEL_FABLE, SUMMARY_MODEL,
    MAX_HISTORY_TURNS, SUMMARIZE_BATCH, _RETRIEVAL_BUDGET, SYSTEM_PROMPT,
    ENABLE_STREAMING,
    ROUTER_SEMANTIC_WEIGHT, ROUTER_SEMANTIC_FLOOR, ROUTER_COST_CEILING,
    ROUTER_ESCALATE_MARGIN, ROUTER_FIXED_INPUT_TOKENS, ROUTER_PROTOTYPES,
)
from claude_light.ui import (
    _T_ROUTE, _T_SYS, _T_ERR, _T_RAG, _T_CACHE, _T_EDIT,
    _ANSI_GREEN, _ANSI_CYAN, _ANSI_YELLOW, _ANSI_MAGENTA, _ANSI_BOLD, _ANSI_DIM, _ANSI_RESET, _ANSI_RED,
    _SYM_RESP, _SYM_TOOL, print_stats, _Spinner, calculate_cost, _print_reply, usage_unavailable, ResponseView, NullView, notify, choose,
)
import claude_light.state as state
from claude_light import providers
from claude_light.providers.base import NotLoggedIn, AGENT, EFFORT, TOP_TIER
from claude_light.skeleton import build_skeleton, _refresh_tree_only, _assemble_skeleton, _refresh_single_md
from claude_light.indexer import index_files
from claude_light.retrieval import retrieve, record_retrieval
from claude_light.editor import parse_edit_blocks, apply_edits
from claude_light.retry import retry_with_backoff
from claude_light.compressor import compress_context

from claude_light.streaming import stream_chat_response, accumulate_usage_from_dict, calculate_usage_cost

# ---------------------------------------------------------------------------
# Router — module-level keyword sets (not re-allocated per call)
# ---------------------------------------------------------------------------

_META_SIGNALS = frozenset({
    "hi", "hello", "help", "who are you", "what are you", "clear history", "compact"
})
_ARCH_SIGNALS = frozenset({
    "architect", "architecture", "design system", "deeply", "deep analysis",
    "performance", "optimize", "security", "scalability", "trade-off",
    "trade off", "tradeoff", "cross-cutting", "evaluate", "compare",
    "strategy", "reasoning", "step by step", "comprehensive", "deadlock", "race condition"
})
_LOGIC_SIGNALS = frozenset({
    "implement", "write", "create", "add", "refactor", "fix", "debug",
    "build", "develop", "generate", "update", "change", "modify",
    "migrate", "convert", "extend", "integrate", "logic", "algorithm"
})
# Edit verbs matched as whole words: "make" must not fire on "makefile" or
# "make_report". Without them, "Make divide raise ValueError..." and
# "Rename x to y" went to Haiku at low effort.
_EDIT_VERB_RE = re.compile(
    r"\b(?:make|rename|remove|delete|replace|raise|handle|rewrite|move|split|validate|wrap|drop|inline|extract)\b"
)
_INFRA_SIGNALS = frozenset({
    "list", "show", "where", "what is", "what are",
    "how many", "print", "display", "tell me", "which file",
    "which files", "find", "locate", "count", "search", "grep"
})

# Prototype embedding cache — keyed by EMBED_MODEL, built once per model load.
_PROTO_CACHE = {"model": None, "mat": None}

# Side-channel: last routing diagnostic written by route_query, read by chat().
_last_route_info: dict = {}


def _encode_if_free(texts):
    """Encode with the shared embedder unless it is busy; None if it is.

    The index build holds embedder_lock for its whole embedding pass, which
    is minutes on a cold start. Routing must not wait for that: it falls back
    to lexical scores instead.
    """
    import numpy as np
    if not state.embedder_lock.acquire(timeout=0.5):
        return None
    try:
        return np.asarray(state.embedder.encode(texts, normalize_embeddings=True))
    finally:
        state.embedder_lock.release()


def _get_prototype_embeddings():
    """Return {dim: ndarray (P,D)} for the current embedder, or None.

    Lazy, cached per EMBED_MODEL. Never blocks, never raises — returns None on
    any problem (test mode, embedder mid-load, encode error).
    """
    if state.embedder is None or not state.embedder_ready.is_set():
        return None
    if _PROTO_CACHE["model"] == state.EMBED_MODEL and _PROTO_CACHE["mat"] is not None:
        return _PROTO_CACHE["mat"]
    try:
        import numpy as np
        from claude_light.config import _QUERY_PREFIX
        prefix = _QUERY_PREFIX.get(state.EMBED_MODEL, "")
        dims = list(ROUTER_PROTOTYPES.keys())
        all_phrases = [prefix + p for d in dims for p in ROUTER_PROTOTYPES[d]]
        all_vecs = _encode_if_free(all_phrases)
        if all_vecs is None:
            return None
        mat = {}
        idx = 0
        for d in dims:
            n = len(ROUTER_PROTOTYPES[d])
            mat[d] = all_vecs[idx:idx + n]
            idx += n
        _PROTO_CACHE.update(model=state.EMBED_MODEL, mat=mat)
        return mat
    except Exception:
        return None


def _lexical_dimension_scores(q: str) -> dict:
    """Weighted substring match scores for each intent dimension."""
    return {
        "arch":  sum(4.0 for s in _ARCH_SIGNALS  if s in q),
        "logic": sum(5.0 for s in _LOGIC_SIGNALS if s in q)
                 + (5.0 if _EDIT_VERB_RE.search(q) else 0.0),
        "infra": sum(0.5 for s in _INFRA_SIGNALS if s in q),
    }


def _semantic_dimension_scores(query: str):
    """Embedding-based per-dimension scores in the same units as lexical.

    Returns (scores: dict, used: bool). When embedder is unavailable, all
    scores are 0 and used=False — the caller's fused score then equals the
    pure lexical score exactly, keeping unit-test outcomes unchanged.
    """
    zeros = {"arch": 0.0, "logic": 0.0, "infra": 0.0}
    protos = _get_prototype_embeddings()
    if protos is None:
        return zeros, False
    try:
        from claude_light.config import _QUERY_PREFIX
        import numpy as np
        prefix = _QUERY_PREFIX.get(state.EMBED_MODEL, "")
        q_emb = _encode_if_free(prefix + query)
        if q_emb is None:
            return zeros, False
        scores = {}
        for dim, proto_mat in protos.items():
            sim = float(np.max(proto_mat @ q_emb))
            floor = ROUTER_SEMANTIC_FLOOR
            scores[dim] = (
                0.0 if sim < floor
                else ROUTER_SEMANTIC_WEIGHT * (sim - floor) / (1.0 - floor)
            )
        return scores, True
    except Exception:
        return zeros, False


def _estimate_query_cost(max_tokens: int, retrieval_budget: int, model=None) -> float:
    """Rough per-query cost estimate used only for tier gating."""
    from claude_light.config import model_prices
    p = model_prices(model)
    expected_input  = retrieval_budget + ROUTER_FIXED_INPUT_TOKENS
    expected_output = max_tokens * 0.5
    return expected_input / 1e6 * p["input"] + expected_output / 1e6 * p["output"]


def route_query(query: str) -> tuple[str, str, int]:
    import re
    q = query.lower()
    words = q.split()
    word_count = len(words)

    # 1. Meta quick-route — short greetings/commands → Haiku immediately
    if word_count < 5 and any(s in q for s in _META_SIGNALS):
        return MODEL_HAIKU, "low", 2_048

    # 2. Fused intent scores (keyword + semantic, additive)
    lex = _lexical_dimension_scores(q)
    sem, sem_used = _semantic_dimension_scores(query)
    arch  = lex["arch"]  + sem["arch"]
    logic = lex["logic"] + sem["logic"]
    infra = lex["infra"] + sem["infra"]
    score = arch + logic + infra

    # 3. Contextual multipliers (identical math to previous router)
    if re.search(r'[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]{2,4}', q):
        score += 3.0

    # Word-count complexity bonus, gated on non-lookup intent
    if arch + logic > 0:
        if word_count > 30:
            score += 5.0
        elif word_count > 10:
            score += 2.0
        elif word_count > 4:
            score += 1.5
    elif word_count > 30:
        score += 5.0

    # Deeper conversations need better instruction following
    history_depth = len(state.conversation_history) // 2
    score += history_depth * 0.5

    # 4. Intent shape
    intent_is_lookup = infra > (arch + logic)

    # 5. Raw tier (same thresholds as before)
    if score >= 12.0:
        effort, model, max_tokens = "max",    MODEL_OPUS,    16_000
    elif score >= 5.0:
        effort, model, max_tokens = "high",   MODEL_SONNET,   8_192
    elif score >= 1.5 or word_count > 15:
        effort, model, max_tokens = "medium", MODEL_SONNET,   4_096
    else:
        effort, model, max_tokens = "low",    MODEL_HAIKU,    2_048

    # 6. Cost-aware adjustments

    def _record_and_return(effort, model, max_tokens):
        _last_route_info.clear()
        _last_route_info.update(
            arch=arch, logic=logic, infra=infra,
            semantic_used=sem_used,
            est_cost=_estimate_query_cost(max_tokens, _RETRIEVAL_BUDGET[effort], model),
        )
        return model, effort, max_tokens

    # Explicit downgrade: lookup-dominant queries (no real arch/logic intent)
    # should never be promoted by verbosity or file-path bonuses.
    if intent_is_lookup and (arch + logic) < 1.0:
        return _record_and_return("low", MODEL_HAIKU, 2_048)

    # Escalation gate: only allow an expensive tier when the score justifiably clears
    # a margin above its threshold. Uses raw score (not conf) so the gate stays
    # reachable — conf caps at 1.0, which would block max-tier entirely.
    _tier_threshold = {"max": 12.0, "high": 5.0, "medium": 1.5, "low": 0.0}
    _tier_order     = ["max", "high", "medium", "low"]
    _tier_params    = {
        "max":    (MODEL_OPUS,    16_000),
        "high":   (MODEL_SONNET,   8_192),
        "medium": (MODEL_SONNET,   4_096),
        "low":    (MODEL_HAIKU,    2_048),
    }
    while True:
        cand_cost = _estimate_query_cost(max_tokens, _RETRIEVAL_BUDGET[effort], model)
        required_score = _tier_threshold[effort] * (1.0 + ROUTER_ESCALATE_MARGIN)
        if cand_cost <= ROUTER_COST_CEILING or score >= required_score:
            break
        next_idx = _tier_order.index(effort) + 1
        if next_idx >= len(_tier_order):
            break
        effort = _tier_order[next_idx]
        model, max_tokens = _tier_params[effort]

    return _record_and_return(effort, model, max_tokens)

_TOP_TIER_MODELS = (MODEL_OPUS, MODEL_FABLE)


def _top_tier_model(effort, max_tokens) -> str:
    """The model a question routed to Opus runs on: asked, or Sonnet 5.5."""
    with state.lock:
        current = state.session_model if config.ROUTE_ONCE else None
        declined = state.top_tier_declined
    if current in _TOP_TIER_MODELS:
        return current
    if declined:
        return MODEL_SONNET
    if not sys.stdin.isatty():
        print(f"  {_ANSI_DIM}◆  top tier needs a terminal to confirm: using sonnet-5-5{_ANSI_RESET}",
              file=sys.stderr)
        return MODEL_SONNET

    def label(name, model):
        est = _estimate_query_cost(max_tokens, _RETRIEVAL_BUDGET[effort], model)
        return f"{name}  ~${est:.2f}"

    choice = choose("This question routed to the top tier. Run it on", [
        ("opus", label("Opus 5.5", MODEL_OPUS)),
        ("fable", label("Fable 5.1", MODEL_FABLE)),
        ("sonnet", label("Sonnet 5.5", MODEL_SONNET)),
        ("never", "Sonnet 5.5, and don't ask again this session"),
    ])
    if choice == "never":
        with state.lock:
            state.top_tier_declined = True
    return {"opus": MODEL_OPUS, "fable": MODEL_FABLE}.get(choice, MODEL_SONNET)


def _finalize_route(model, effort, max_tokens) -> tuple[str, str, int]:
    """Apply the session to route_query()'s pick: budget cap, top-tier
    confirmation and ROUTE_ONCE. route_query() itself stays a pure router, so
    benchmarks and /why never pin or prompt."""
    _effort_color = {
        "low":    _ANSI_GREEN,
        "medium": _ANSI_CYAN,
        "high":   _ANSI_YELLOW,
        "max":    _ANSI_MAGENTA,
    }
    provider = providers.active()
    if not provider.has(TOP_TIER):
        # The provider's tool picks the model: nothing to pin, prompt for or price.
        label = provider.model_label(model)
        note = "" if provider.has(EFFORT) else f"  ·  effort unavailable for {provider.name}"
        if model == MODEL_OPUS:
            note += f"  ·  top tier unavailable for {provider.name}"
        print(f"  {_ANSI_DIM}◆  {provider.name}  ·  {label}{note}{_ANSI_RESET}", file=sys.stderr)
        return label, effort, max_tokens
    budget = state.session_budget
    over_budget = budget is not None and state.session_cost >= budget
    if over_budget:
        effort, model, max_tokens = "low", MODEL_HAIKU, 2_048
    elif model == MODEL_OPUS:
        # The only route that may move a ROUTE_ONCE session to another model.
        model = _top_tier_model(effort, max_tokens)
        if config.ROUTE_ONCE:
            with state.lock:
                state.session_model = model
    elif config.ROUTE_ONCE:
        with state.lock:
            if state.session_model is None:
                state.session_model = model
            model = state.session_model
    color = _effort_color[effort]
    model_short = model.replace("claude-", "").replace("-20251001", "")
    extra = ""
    if budget is not None:
        est = _estimate_query_cost(max_tokens, _RETRIEVAL_BUDGET[effort], model)
        extra = f"  ·  ~${est:.3f}" + ("  ·  over budget" if over_budget else "")
    print(
        f"  {_ANSI_DIM}◆  {model_short}  ·  {color}{effort}{_ANSI_RESET}{_ANSI_DIM}{extra}{_ANSI_RESET}",
        file=sys.stderr,
    )
    return model, effort, max_tokens

def _extract_text(content_blocks) -> str:
    return "".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )

def _accumulate_usage(usage, model=None):
    from claude_light.config import model_prices
    from claude_light.ui import cost_breakdown
    p = model_prices(model)
    input_tokens = usage.input_tokens
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    cache_write = getattr(usage, "cache_creation_input_tokens", 0)
    output_tokens = usage.output_tokens
    costs = cost_breakdown(usage, model)

    with state.lock:
        state.session_tokens["input"]       += input_tokens
        state.session_tokens["cache_write"] += cache_write
        state.session_tokens["cache_read"]  += cache_read
        state.session_tokens["output"]      += output_tokens
        for category, dollars in costs.items():
            state.session_costs[category] += dollars
        state.session_costs["without_cache"] += (
            (input_tokens + cache_write + cache_read) / 1e6 * p["input"]
            + output_tokens / 1e6 * p["output"]
        )

        # Update Global Stats
        # Saved tokens = everything that hit the cache (read)
        # Full tokens = everything that didn't hit the cache (input + write)
        state.global_stats["total_tokens_full"]  += (input_tokens + cache_write)
        state.global_stats["total_tokens_saved"] += cache_read
        
        # Dollars saved = tokens * (Full Price - Cache Read Price)
        dollars_saved = (cache_read / 1_000_000.0) * (p["input"] - p["read"])
        state.global_stats["total_dollars_saved"] += dollars_saved
    
    # Persist after every interaction
    state.save_global_stats()

def _record_usage(usage, model) -> bool:
    """Add one call's figures to the session totals. A provider that reports
    none (usage is None) adds nothing; returns False so the caller can say so."""
    if usage is None:
        return False
    cost = calculate_cost(usage, model)
    _accumulate_usage(usage, model)
    with state.lock:
        state.session_cost += cost
    return True


def _accumulate_compression_stats(info: dict, model=None) -> None:
    """Record tokens saved by LLMLingua-2 in global_stats.

    Priced at the model's 5-minute cache-write rate: the retrieved-context
    block is new on nearly every query (unique top-K chunks), so its first
    trip through the API is a cache write. The input rate would overstate
    savings; the read rate would understate them.
    """
    if not info or info.get("skipped"):
        return
    from claude_light.config import model_prices
    delta = max(0, info["tokens_before"] - info["tokens_after"])
    if delta == 0:
        return
    dollars = (delta / 1_000_000.0) * model_prices(model)["write"]
    with state.lock:
        state.global_stats["total_tokens_pre_compress"]    += info["tokens_before"]
        state.global_stats["total_tokens_post_compress"]   += info["tokens_after"]
        state.global_stats["total_dollars_saved_llmlingua"] += dollars
        state.session_tokens["compress_pre"]  += info["tokens_before"]
        state.session_tokens["compress_post"] += info["tokens_after"]
    state.save_global_stats()

def _summarize_turns(messages: list) -> tuple:
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        lines.append(f"{role}: {content}")
    dialogue = "\n\n".join(lines)
    instructions = (
        "Summarize the following conversation turns concisely (under 300 words).\n"
        "Capture: what was asked, decisions reached, files edited and what changed, "
        "key facts established about the codebase.\n"
        "Output only the summary — no preamble.\n\n"
        f"<turns>\n{dialogue}\n</turns>"
    )

    return providers.active().summarize(instructions)

def compact_history() -> int:
    """Replace the whole conversation with a summary of it (/compact).

    Returns how many turns were summarised; 0 means there was nothing to do.
    """
    history = state.conversation_history
    if not history:
        return 0
    turns = (len(history) + 1) // 2
    summary_text, usage = _summarize_turns(list(history))
    _record_usage(usage, SUMMARY_MODEL)
    history[:] = [
        {"role": "user",      "content": f"[Summary of earlier conversation]\n{summary_text}"},
        {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
    ]
    return turns


def _maybe_compress_history():
    if len(state.conversation_history) <= MAX_HISTORY_TURNS * 2:
        return
    batch_msgs   = SUMMARIZE_BATCH * 2
    to_summarize = state.conversation_history[:batch_msgs]
    remaining    = state.conversation_history[batch_msgs:]
    try:
        print(f"  {_T_SYS}  Compressing history…", end="", flush=True)
        summary_text, usage = _summarize_turns(to_summarize)
        _record_usage(usage, SUMMARY_MODEL)
        summary_pair = [
            {"role": "user",      "content": f"[Summary of earlier conversation]\n{summary_text}"},
            {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
        ]
        state.conversation_history = summary_pair + remaining
        print(f" done")
    except Exception as e:
        print(f"\n  {_T_ERR}  History compression failed ({e}) — truncating.")
        state.conversation_history = state.conversation_history[-(MAX_HISTORY_TURNS * 2):]

# Changing the top-level effort between requests invalidates the cached
# conversation, and on some models the cached tools and system prompt too. So
# every request sends BASE_EFFORT, and each turn's routed effort travels as an
# effort-only system message (EFFORT_BETA), which keeps the cache. Lowering
# effort that way is reliable, so the base is the highest level used often.
BASE_EFFORT = "high"
# Most tool rounds one API turn may run (tests/unit/test_llm.py reads it here).
_API_TOOL_STEPS = 8
EFFORT_BETA = "mid-conversation-output-config-2026-07-01"

# The 1h cache TTL used in _build_system_blocks() requires the first beta;
# without it the API silently drops the cache_control block (no caching at all).
# The second allows the effort messages above.
_EXTENDED_CACHE_HEADERS = {"anthropic-beta": f"extended-cache-ttl-2025-04-11,{EFFORT_BETA}"}

# The 5.5 models always think adaptively; depth is set by output_config.effort
# and a thinking budget is rejected with a 400. Thinking tokens count against
# max_tokens, so the router's caps (sized for the answer alone) would truncate
# replies. Requests stream, so a larger cap costs nothing unless it is used.
_MIN_MAX_TOKENS = {"low": 8_000, "medium": 16_000, "high": 32_000, "max": 64_000}


def _agent_oauth() -> bool:
    """True when a CLI agent (not the SDK) answers and can edit files itself."""
    return AUTH_MODE == "OAUTH" and providers.active().has(AGENT)


def _apply_effort(create_kwargs: dict, effort: str) -> None:
    """Run the request's last user turn at `effort` without changing the top-level effort."""
    create_kwargs["output_config"] = {"effort": BASE_EFFORT}
    create_kwargs["max_tokens"] = max(create_kwargs["max_tokens"], _MIN_MAX_TOKENS[effort])
    # The effort message goes right before the question and is not stored in
    # the history: the history's cache breakpoint sits ahead of it, so the next
    # turn's prefix still matches. The CLI path flattens messages to text.
    if effort != BASE_EFFORT and providers.active().has(EFFORT) and not _agent_oauth():
        messages = create_kwargs["messages"]
        create_kwargs["messages"] = messages[:-1] + [
            {"role": "system", "content": [], "output_config": {"effort": effort}},
            messages[-1],
        ]


def _build_system_blocks(skeleton):
    # Skeleton uses the 1-hour extended TTL: it changes only when source files
    # or .md docs are edited (rare within a session), so the 5-minute default
    # forces unnecessary cache writes on idle conversations. The 1h variant
    # costs 2x on write but reads stay at $0.30/M for an hour, which dominates
    # for any session with idle gaps over 5 min.
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    return blocks


def _git_modified_snapshot():
    """Return the set of files currently dirty in the working tree.

    Used in OAUTH mode to detect edits the Claude CLI agent made directly
    via its own Read/Edit/Bash tools (`--tools ""` doesn't actually disable
    those in CLI 2.1.118). We snapshot before the API call and diff after
    so we only commit files the agent touched during this turn, not
    whatever the user had dirty beforehand.
    """
    from claude_light import git_manager
    if not git_manager.is_git_repo():
        return set()
    return set(git_manager.get_modified_files())


def _commit_agent_edits(new_files, explanation, auto_apply=False):
    """Commit edits the OAUTH CLI agent made directly (outside our
    SEARCH/REPLACE pipeline). Shows a git diff per file first; if stdin is
    interactive and auto_apply is False, asks before committing.
    """
    from claude_light import git_manager
    files = sorted(new_files)
    if not files:
        return

    print()
    # Render the diff straight from git so we don't have to track old content
    from claude_light.ui import show_diff, show_unified_diff
    import subprocess as _sub
    for f in files:
        try:
            diff_out = _sub.run(
                ["git", "diff", "--", f],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                stdin=_sub.DEVNULL,
            ).stdout
        except Exception:
            diff_out = ""
        if diff_out.strip():
            show_unified_diff(f"{f}  (agent edit)", diff_out)
        elif os.path.isfile(f):
            # git diff is silent for an untracked file: a new one the agent made.
            try:
                show_diff(f, Path(f).read_text(encoding="utf-8", errors="replace"), old_content="")
            except OSError:
                print(f"  {_T_EDIT}  {f}  {_ANSI_DIM}(agent edit){_ANSI_RESET}")
        else:
            print(f"  {_T_EDIT}  {f}  {_ANSI_DIM}(deleted by the agent){_ANSI_RESET}")

    if auto_apply or state.auto_commit_agent_edits:
        commit = True
    elif not sys.stdin.isatty():
        commit = True
    else:
        n = len(files)
        choice = choose(
            f"Commit the agent's change{'s' if n != 1 else ''} to {n} file{'s' if n != 1 else ''}?",
            [("yes", "Yes"),
             ("always", "Yes, and don't ask again this session"),
             ("no", "No, leave them uncommitted")],
        )
        if choice == "always":
            state.auto_commit_agent_edits = True
        commit = choice in ("yes", "always")

    if not commit:
        print(f"  {_T_EDIT}  Kept in working tree; not committed.\n")
        return

    git_manager.auto_commit(files, explanation)



def _apply_skeleton(new_skeleton):
    with state.lock:
        state.skeleton_context = new_skeleton
        state.last_interaction = time.time()

def _update_skeleton():
    _apply_skeleton(build_skeleton())

def warm_cache(quiet=False):
    providers.active().warm(quiet=quiet)

def full_refresh(quiet=False):
    if quiet:
        _update_skeleton()
        index_files(quiet=True)
        if AUTH_MODE == "API_KEY":
            warm_cache(quiet=True)
    else:
        with _Spinner("Building skeleton") as spinner:
            _update_skeleton()
            spinner.update("Indexing source files")
            index_files(quiet=True)
            if AUTH_MODE == "API_KEY":
                spinner.update("Warming cache")
                warm_cache(quiet=True)

# Both run on the file watcher's thread: quiet, so nothing prints into a
# streaming reply or over the prompt.
def refresh_skeleton_only():
    _apply_skeleton(_refresh_tree_only())
    warm_cache(quiet=True)

def refresh_md_file(path_str: str):
    if _refresh_single_md(path_str):
        _apply_skeleton(_assemble_skeleton())
        warm_cache(quiet=True)
        notify(f"{path_str} changed; system prompt rebuilt")


_CLARIFY_PHRASES = frozenset({
    "why", "how", "how come", "explain", "explain that", "elaborate",
    "more", "tell me more", "go on", "continue", "and", "and then",
    "what do you mean", "huh", "wait", "really", "ok", "okay",
    "that's it", "is that all", "anything else", "go ahead", "please",
})

_PRONOUN_REFS = frozenset({"that", "this", "it", "those", "these", "them"})


def _is_followup_clarification(query: str) -> bool:
    """Detect short pronoun-reference follow-ups that don't need fresh RAG.

    Returns True only when there's prior conversation history AND the query
    is a short clarification ("why?", "explain that", "more"). Skipping
    retrieval on these saves the full retrieval token budget (1.5K-9K) per
    turn — they would have pulled chunks based on noise anyway, since
    pronouns don't embed meaningfully.
    """
    if not state.conversation_history:
        return False
    q = query.lower().strip().rstrip("?.! ")
    if not q:
        return False
    words = q.split()
    if len(words) > 6:
        return False
    if q in _CLARIFY_PHRASES:
        return True
    # Short queries dominated by pronoun refs: "explain that", "more on it"
    if len(words) <= 4 and any(w in _PRONOUN_REFS for w in words):
        return True
    return False


def chat(query, auto_apply=False, use_index=True):
    routed_model, effort, max_tok = _finalize_route(*route_query(query))
    token_budget = _RETRIEVAL_BUDGET[effort]

    from claude_light.indexer import _chunk_label
    skipped_reason = None
    if not use_index:
        retrieved_ctx, hits = "", []
        skipped_reason = "index still building"
        print(
            f"  {_T_RAG}  {_ANSI_DIM}index still building — Claude will search the code itself{_ANSI_RESET}"
        )
    elif _is_followup_clarification(query):
        retrieved_ctx, hits = "", []
        skipped_reason = "clarification follow-up"
        print(
            f"  {_T_RAG}  {_ANSI_DIM}skipping retrieval — clarification follow-up{_ANSI_RESET}"
        )
    else:
        retrieved_ctx, hits = retrieve(query, token_budget=token_budget, effort=effort)

    compression_info = None
    if retrieved_ctx and config.LLMLINGUA_ENABLED:
        retrieved_ctx, compression_info = compress_context(retrieved_ctx)
        _accumulate_compression_stats(compression_info, routed_model)
        if compression_info and not compression_info.get("skipped"):
            pct = compression_info["ratio"] * 100
            tag = "cached" if compression_info.get("cached") else f"{compression_info['elapsed_ms']:.0f} ms"
            print(
                f"  {_T_RAG}  {_ANSI_DIM}LLMLingua "
                f"{compression_info['tokens_before']:,}→{compression_info['tokens_after']:,} "
                f"({pct:.0f}%, {tag}){_ANSI_RESET}"
            )

    if hits:
        names = "  ".join(_chunk_label(p) for p, _ in hits)
        print(f"  {_T_RAG}  {_ANSI_DIM}{names}{_ANSI_RESET}")

    record_retrieval(query, routed_model, effort, token_budget, hits,
                     compression=compression_info, skipped_reason=skipped_reason,
                     routing=_last_route_info.copy() if _last_route_info else None)

    with state.lock:
        skeleton = state.skeleton_context

    _maybe_compress_history()
    trimmed = state.conversation_history[-(MAX_HISTORY_TURNS * 2):]

    if trimmed:
        last_msg = trimmed[-1]
        if isinstance(last_msg["content"], str):
            trimmed[-1] = {
                "role": last_msg["role"],
                "content": [{"type": "text", "text": last_msg["content"], "cache_control": {"type": "ephemeral"}}]
            }

    current_blocks = []
    if retrieved_ctx:
        current_blocks.append({
            "type": "text",
            "text": f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n",
            "cache_control": {"type": "ephemeral"},
        })
    import re
    from claude_light.editor import _ANY_BLOCK
    current_blocks.append({"type": "text", "text": f"Question:\n{query}"})
    messages = trimmed + [{"role": "user", "content": current_blocks}]

    system = _build_system_blocks(skeleton)

    create_kwargs = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=system,
        messages=messages,
        extra_headers=_EXTENDED_CACHE_HEADERS,
    )
    _apply_effort(create_kwargs, effort)

    # Snapshot dirty files BEFORE the API call so we can distinguish agent
    # edits (OAUTH mode: the CLI's own Read/Edit/Bash tools aren't actually
    # blocked by --tools "") from changes the user had pending.
    pre_modified = _git_modified_snapshot() if _agent_oauth() else set()

    try:
        for attempt in range(3):
            reply, usage, was_streamed = providers.active().complete(create_kwargs)

            edits = parse_edit_blocks(reply)

            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    _record_usage(usage, routed_model)

                    print(f"\n  {_T_ERR}  {_ANSI_RED}Syntax errors detected — requesting correction (attempt {attempt+1}/3)…{_ANSI_RESET}")
                    for err in lint_errs:
                        print(f"    {_ANSI_DIM}{err.strip().replace(chr(10), ' ')}{_ANSI_RESET}")

                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."

                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})

                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]:
                                b.pop("cache_control", None)
                    continue

            state.conversation_history.append({"role": "user", "content": query})

            if edits:
                labels = ", ".join(e["path"] for e in edits)
                clean_reply = _ANY_BLOCK.sub("", reply).strip()
                clean_reply = (clean_reply + f"\n[Files edited: {labels}]").strip()
            else:
                clean_reply = reply
            state.conversation_history.append({"role": "assistant", "content": clean_reply})

            has_usage = _record_usage(usage, routed_model)
            with state.lock:
                state.last_interaction = time.time()

            state.turn_count += 1
            turns = state.turn_count

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation and not was_streamed:
                print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)
                _print_reply(explanation)
                sys.stdout.flush()
            if edits:
                apply_edits(edits, explanation=explanation, auto_apply=auto_apply)
            elif _agent_oauth():
                # No SEARCH/REPLACE block but the CLI agent may have edited
                # files itself. Diff against the pre-call snapshot; commit
                # anything new so /undo can revert it. Use the user's
                # original query as the commit message — the agent's reply
                # is narration ("Let me first check…") and makes for a
                # noisy commit log.
                post_modified = _git_modified_snapshot()
                agent_edits = post_modified - pre_modified
                if agent_edits:
                    _commit_agent_edits(agent_edits, query, auto_apply=auto_apply)

            print()
            if has_usage:
                print_stats(usage, label=f"Turn {turns}", model=routed_model)
            else:
                print(f"  {_ANSI_DIM}{usage_unavailable(providers.active().name)}{_ANSI_RESET}")
            break

    except KeyboardInterrupt:
        note = (" The agent was stopped; files it already changed are left"
                " uncommitted in the working tree.") if _agent_oauth() else ""
        print(f"\n  {_T_ERR}  Interrupted.{note}")
    except NotLoggedIn:
        # Re-raise so main.py can handle the interactive login trigger
        raise
    except Exception as e:
        print(f"\n[Error] API call failed: {e}")

# How long a one-shot OAuth run waits for the index before asking without it.
# A warm cache loads in a few seconds; a cold build takes minutes.
_ONE_SHOT_INDEX_WAIT_SECS = 20


def one_shot(prompt, auto_apply=False):
    print(f"{_T_SYS} Building context...", file=sys.stderr)
    _update_skeleton()

    if not _agent_oauth():
        # Retrieval is the model's starting point here, so wait for the index.
        # dense="skip": one-shot exits before a background pass could finish;
        # dense vectors already cached are still used.
        index_files(dense="skip")
        _one_shot_turn(prompt, auto_apply, use_index=True)
        return

    # The CLI agent can Read/Grep the repo itself, so a cold build (minutes
    # of CPU embedding) must not hold up the answer. It keeps running and is
    # joined before exit, so the next run starts from a full cache.
    import threading as _threading
    index_error = []

    def _build_index():
        try:
            index_files(quiet=True, dense="skip")
        except Exception as e:  # reported below; the turn goes on without it
            index_error.append(e)

    index_thread = _threading.Thread(target=_build_index, daemon=True, name="claude-light-index")
    index_thread.start()
    index_thread.join(_ONE_SHOT_INDEX_WAIT_SECS)
    use_index = not index_thread.is_alive() and not index_error
    if index_thread.is_alive():
        print(f"  {_T_RAG}  {_ANSI_DIM}index still building — Claude will search the code itself{_ANSI_RESET}",
              file=sys.stderr)
    elif index_error:
        print(f"  {_T_ERR}  Indexing failed ({index_error[0]}); continuing without retrieval", file=sys.stderr)
    try:
        _one_shot_turn(prompt, auto_apply, use_index=use_index)
    finally:
        if index_thread.is_alive():
            print(f"  {_T_RAG}  {_ANSI_DIM}finishing the index so the next run can use it…{_ANSI_RESET}",
                  file=sys.stderr, flush=True)
            index_thread.join()


def _one_shot_turn(prompt, auto_apply, use_index):
    routed_model, effort, max_tok = _finalize_route(*route_query(prompt))
    if use_index:
        retrieved_ctx, hits = retrieve(prompt, token_budget=_RETRIEVAL_BUDGET[effort], effort=effort)
    else:
        retrieved_ctx, hits = "", []

    if retrieved_ctx and config.LLMLINGUA_ENABLED:
        retrieved_ctx, info = compress_context(retrieved_ctx)
        _accumulate_compression_stats(info, routed_model)

    from claude_light.indexer import _chunk_label
    import re
    from claude_light.editor import _ANY_BLOCK

    if hits:
        names = "  ".join(_chunk_label(p) for p, _ in hits)
        print(f"  {_T_RAG}  {_ANSI_DIM}{names}{_ANSI_RESET}", file=sys.stderr)

    with state.lock:
        skeleton = state.skeleton_context

    ctx_prefix = f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n" if retrieved_ctx else ""
    create_kwargs = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=_build_system_blocks(skeleton),
        messages=[{"role": "user", "content": f"{ctx_prefix}Question:\n{prompt}"}],
        extra_headers=_EXTENDED_CACHE_HEADERS,
    )
    _apply_effort(create_kwargs, effort)

    # See chat() for rationale — snapshot to detect agent edits in OAUTH mode.
    pre_modified = _git_modified_snapshot() if _agent_oauth() else set()

    try:
        for attempt in range(3):
            reply, usage, was_streamed = providers.active().complete(create_kwargs)

            edits = parse_edit_blocks(reply)

            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    _record_usage(usage, routed_model)

                    print(f"  {_T_ERR}  Syntax errors — correcting (attempt {attempt+1}/3)…", file=sys.stderr)
                    for err in lint_errs:
                        print(f"    {err.strip().replace(chr(10), ' ')}", file=sys.stderr)

                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."
                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})

                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]:
                                b.pop("cache_control", None)
                    continue

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation and not was_streamed:
                print(f"\n{_ANSI_CYAN}{_ANSI_BOLD}{_SYM_RESP}{_ANSI_RESET} ", end="", flush=True)
                print(explanation)
                sys.stdout.flush()
            if edits:
                apply_edits(edits, explanation=explanation, auto_apply=auto_apply)
            elif _agent_oauth():
                # See chat() — use the prompt, not the agent's narration.
                post_modified = _git_modified_snapshot()
                agent_edits = post_modified - pre_modified
                if agent_edits:
                    _commit_agent_edits(agent_edits, prompt, auto_apply=auto_apply)

            if _record_usage(usage, routed_model):
                print_stats(usage, label="Stats", file=sys.stderr, model=routed_model)
            else:
                print(f"  {_ANSI_DIM}{usage_unavailable(providers.active().name)}{_ANSI_RESET}",
                      file=sys.stderr)
            break

    except Exception as e:
        raise SystemExit(f"{_T_ERR} API call failed: {e}")
