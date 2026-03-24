import sys
import time
import anthropic

from claude_light.config import (
    API_KEY, MODEL, MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS, SUMMARY_MODEL,
    MAX_HISTORY_TURNS, SUMMARIZE_BATCH, _RETRIEVAL_BUDGET, SYSTEM_PROMPT
)
from claude_light.ui import (
    _T_ROUTE, _T_SYS, _T_ERR, _T_RAG, _T_CACHE,
    _ANSI_GREEN, _ANSI_CYAN, _ANSI_YELLOW, _ANSI_MAGENTA, _ANSI_BOLD, _ANSI_DIM, _ANSI_RESET, _ANSI_RED,
    print_stats, _Spinner, calculate_cost, _print_reply
)
import claude_light.state as state
from claude_light.skeleton import build_skeleton, _refresh_tree_only, _assemble_skeleton, _refresh_single_md
from claude_light.indexer import index_files
from claude_light.retrieval import retrieve
from claude_light.editor import parse_edit_blocks, apply_edits

client = anthropic.Anthropic(api_key=API_KEY) if API_KEY != "sk-ant-test-mock-key" else None
if not client:
    class MockClient:
        pass
    client = MockClient()
    client.messages = MockClient()
    client.messages.create = None

def route_query(query: str) -> tuple[str, str, int]:
    q = query.lower()
    words = q.split()
    word_count = len(words)

    _LOW_SIGNALS = {
        "list", "show", "where", "what is", "what are", "how many",
        "print", "display", "tell me", "which file", "which files",
        "find", "locate", "count",
    }
    _HIGH_SIGNALS = {
        "implement", "write", "create", "add", "refactor", "fix", "debug",
        "build", "develop", "generate", "update", "change", "modify",
        "migrate", "convert", "extend", "integrate",
    }
    _MAX_SIGNALS = {
        "architect", "architecture", "design system", "deeply", "deep analysis",
        "performance", "optimize", "security", "scalability", "trade-off",
        "trade off", "tradeoff", "cross-cutting", "evaluate", "compare",
        "strategy", "reasoning", "step by step", "comprehensive",
    }

    low_hits  = sum(1 for s in _LOW_SIGNALS  if s in q)
    high_hits = sum(1 for s in _HIGH_SIGNALS if s in q)
    max_hits  = sum(1 for s in _MAX_SIGNALS  if s in q)

    if max_hits >= 2 or (max_hits >= 1 and word_count > 35):
        effort, model, max_tokens = "max",    MODEL_OPUS,   16_000
    elif high_hits >= 1 or word_count > 30:
        effort, model, max_tokens = "high",   MODEL_SONNET,  8_192
    elif low_hits >= 1 and high_hits == 0 and word_count <= 20:
        effort, model, max_tokens = "low",    MODEL_HAIKU,   2_048
    else:
        effort, model, max_tokens = "medium", MODEL_SONNET,  4_096

    _effort_color = {
        "low":    _ANSI_GREEN,
        "medium": _ANSI_CYAN,
        "high":   _ANSI_YELLOW,
        "max":    _ANSI_MAGENTA,
    }
    color = _effort_color[effort]
    model_short = model.split("-")[1]
    print(
        f"{_T_ROUTE} effort={color}{effort}{_ANSI_RESET}  "
        f"model={_ANSI_BOLD}{model_short}{_ANSI_RESET}",
        file=sys.stderr,
    )
    return model, effort, max_tokens

def _extract_text(content_blocks) -> str:
    return "".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )

def _accumulate_usage(usage):
    state.session_tokens["input"]       += usage.input_tokens
    state.session_tokens["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0)
    state.session_tokens["cache_read"]  += getattr(usage, "cache_read_input_tokens", 0)
    state.session_tokens["output"]      += usage.output_tokens

def _summarize_turns(messages: list) -> tuple:
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        lines.append(f"{role}: {content}")
    dialogue = "\n\n".join(lines)
    response = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                "Summarize the following conversation turns concisely (under 300 words).\n"
                "Capture: what was asked, decisions reached, files edited and what changed, "
                "key facts established about the codebase.\n"
                "Output only the summary — no preamble.\n\n"
                f"<turns>\n{dialogue}\n</turns>"
            ),
        }],
    )
    return response.content[0].text.strip(), response.usage

def _maybe_compress_history():
    if len(state.conversation_history) <= MAX_HISTORY_TURNS * 2:
        return
    batch_msgs   = SUMMARIZE_BATCH * 2
    to_summarize = state.conversation_history[:batch_msgs]
    remaining    = state.conversation_history[batch_msgs:]
    try:
        print(f"{_T_SYS} Compressing {SUMMARIZE_BATCH} old turns...", end="", flush=True)
        summary_text, usage = _summarize_turns(to_summarize)
        _accumulate_usage(usage)
        with state.lock:
            state.session_cost += calculate_cost(usage)
        summary_pair = [
            {"role": "user",      "content": f"[Summary of earlier conversation]\n{summary_text}"},
            {"role": "assistant", "content": "Understood, I have context from our earlier discussion."},
        ]
        state.conversation_history = summary_pair + remaining
        print(f" done ({usage.input_tokens}→{usage.output_tokens} tok)")
    except Exception as e:
        print(f"\n{_T_ERR} History compression failed ({e}) — truncating instead.")
        state.conversation_history = state.conversation_history[-(MAX_HISTORY_TURNS * 2):]

def _build_system_blocks(skeleton):
    blocks = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": skeleton, "cache_control": {"type": "ephemeral"}},
    ]
    return blocks

def _apply_skeleton(new_skeleton: str):
    with state.lock:
        state.skeleton_context = new_skeleton
        state.last_interaction = time.time()

def _update_skeleton():
    _apply_skeleton(build_skeleton())

def warm_cache(quiet=False):
    with state.lock:
        skeleton = state.skeleton_context
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1,
            system=_build_system_blocks(skeleton),
            messages=[{"role": "user", "content": "ok"}],
        )
        cost = calculate_cost(response.usage)
        _accumulate_usage(response.usage)
        with state.lock:
            state.session_cost += cost

        if not quiet:
            print(f"\n{_ANSI_DIM}{'═'*56}{_ANSI_RESET}")
            print_stats(response.usage, label="Cache")
            print(f"{_ANSI_DIM}{'═'*56}{_ANSI_RESET}\n")
    except Exception as e:
        print(f"{_T_ERR} Cache warm failed: {e}")

def full_refresh():
    with _Spinner("Building skeleton") as spinner:
        _update_skeleton()
        spinner.update("Indexing source files")
        index_files(quiet=True)
        spinner.update("Warming cache")
        warm_cache(quiet=True)

def refresh_skeleton_only():
    _apply_skeleton(_refresh_tree_only())
    warm_cache()

def refresh_md_file(path_str: str):
    if _refresh_single_md(path_str):
        _apply_skeleton(_assemble_skeleton())
        warm_cache()


def chat(query):
    routed_model, effort, max_tok = route_query(query)
    token_budget = _RETRIEVAL_BUDGET[effort]

    from claude_light.indexer import _chunk_label
    retrieved_ctx, hits = retrieve(query, token_budget=token_budget)

    if hits:
        names      = ", ".join(_chunk_label(p) for p, _ in hits)
        scores_str = "  ".join(f"{_chunk_label(p)} {s:.2f}" for p, s in hits)
        print(f"{_T_RAG} {_ANSI_CYAN}{names}{_ANSI_RESET}")
        print(f"      {_ANSI_DIM}{scores_str}{_ANSI_RESET}")

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
    )
    if effort == "max":
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}

    try:
        for attempt in range(3):
            response = client.messages.create(**create_kwargs)
            reply = _extract_text(response.content)
            edits = parse_edit_blocks(reply)
            
            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    cost = calculate_cost(response.usage)
                    _accumulate_usage(response.usage)
                    with state.lock:
                        state.session_cost += cost

                    print(f"\n{_T_ERR} {_ANSI_RED}Auto-detected errors in AI's code:{_ANSI_RESET}")
                    for err in lint_errs:
                        err_clean = err.strip().replace('\n', ' ')
                        print(f"  {err_clean}")
                    print(f"[{_ANSI_CYAN}Auto-correction{_ANSI_RESET}] Requesting fix from Claude (attempt {attempt+1}/3)...")
                    
                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."
                    
                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})
                    
                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]: b.pop("cache_control", None)
                    continue

            state.conversation_history.append({"role": "user", "content": query})

            if edits:
                labels = ", ".join(e["path"] for e in edits)
                clean_reply = _ANY_BLOCK.sub("", reply).strip()
                clean_reply = (clean_reply + f"\n[Files edited: {labels}]").strip()
            else:
                clean_reply = reply
            state.conversation_history.append({"role": "assistant", "content": clean_reply})

            cost = calculate_cost(response.usage)
            _accumulate_usage(response.usage)
            with state.lock:
                state.last_interaction = time.time()
                state.session_cost += cost

            turns = len(state.conversation_history) // 2

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation:
                _print_reply(explanation)
            if edits:
                apply_edits(edits)

            print_stats(response.usage, label=f"Turn {turns}")
            break

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[Error] API call failed: {e}")

def one_shot(prompt):
    print(f"{_T_SYS} Building context...", file=sys.stderr)
    _update_skeleton()
    index_files()

    routed_model, effort, max_tok = route_query(prompt)
    retrieved_ctx, hits = retrieve(prompt, token_budget=_RETRIEVAL_BUDGET[effort])
    
    from claude_light.indexer import _chunk_label
    import re
    from claude_light.editor import _ANY_BLOCK
    
    if hits:
        names = ", ".join(_chunk_label(p) for p, _ in hits)
        print(f"{_T_RAG} {_ANSI_CYAN}{names}{_ANSI_RESET}", file=sys.stderr)

    with state.lock:
        skeleton = state.skeleton_context

    ctx_prefix = f"Retrieved Codebase Context:\n{retrieved_ctx}\n\n" if retrieved_ctx else ""
    create_kwargs = dict(
        model=routed_model,
        max_tokens=max_tok,
        system=_build_system_blocks(skeleton),
        messages=[{"role": "user", "content": f"{ctx_prefix}Question:\n{prompt}"}],
    )
    if effort == "max":
        create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10_000}
    try:
        for attempt in range(3):
            response = client.messages.create(**create_kwargs)
            reply = _extract_text(response.content)
            edits = parse_edit_blocks(reply)

            if edits:
                lint_errs = apply_edits(edits, check_only=True)
                if lint_errs:
                    cost = calculate_cost(response.usage)
                    _accumulate_usage(response.usage)
                    with state.lock:
                        state.session_cost += cost

                    print(f"\n{_T_ERR} Auto-detected errors:", file=sys.stderr)
                    for err in lint_errs:
                        err_clean = err.strip().replace('\n', ' ')
                        print(f"  {err_clean}", file=sys.stderr)
                    print(f"[Auto-correction] Requesting fix from Claude (attempt {attempt+1}/3)...", file=sys.stderr)
                    
                    err_msg = "[Error] The code you provided failed with the following errors:\n" + "\n".join(lint_errs) + "\nPlease provide corrected SEARCH/REPLACE blocks."
                    create_kwargs["messages"].append({"role": "assistant", "content": reply})
                    create_kwargs["messages"].append({"role": "user", "content": [{"type": "text", "text": err_msg}]})
                    
                    for m in create_kwargs["messages"][:-1]:
                        if isinstance(m["content"], list):
                            for b in m["content"]: b.pop("cache_control", None)
                    continue

            explanation = _ANY_BLOCK.sub("", reply).strip() if edits else reply
            if explanation:
                print(explanation)
            if edits:
                apply_edits(edits)

            cost = calculate_cost(response.usage)
            _accumulate_usage(response.usage)
            with state.lock:
                state.session_cost += cost
            print_stats(response.usage, label="Stats", file=sys.stderr)
            break

    except Exception as e:
        raise SystemExit(f"{_T_ERR} API call failed: {e}")
