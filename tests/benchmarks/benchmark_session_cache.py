#!/usr/bin/env python3
"""
Session cache cost model: what three routing policies cost over one session.

Routes the questions in tests/fixtures/repo_questions.json with the real
route_query() and static embedder, plays them as one session (the fixture
order, then shuffled orders) and prices every turn with MODEL_PRICES and a
prompt-cache model:

  * the skeleton, tools and system prompt (measured from this repo) are cached
    per model for 1 hour
  * the conversation history is cached per model for 5 minutes; a cached
    prefix stays readable across turns on other models, so a switch back pays
    only for the turns in between
  * history compression (MAX_HISTORY_TURNS) rewrites the history

Policies:

  old      per-turn model and a per-turn top-level effort, which invalidates the
           history cache whenever the effort changes on the same model
  effort   per-turn model, one top-level effort, the routed effort in an
           effort message (what llm._apply_effort does)
  pinned   effort, plus the model only moves up a tier within a session
           (measured and rejected, see TOKEN-SAVINGS.md)

Whether an effort change also invalidates the cached skeleton is model-specific
and not documented per model; --effort-in-prefix models the case where it does.
Output tokens per effort and the answer size kept in history are assumptions,
scaled by --output-scale. This is a model, not a live measurement: it shows
which policy is cheaper and by roughly how much, not what a session costs.

    python tests/benchmarks/benchmark_session_cache.py
    python tests/benchmarks/benchmark_session_cache.py --gap 3 --output-scale 0.3 --effort-in-prefix

Not collected by pytest. Run from the repo root.
"""
import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import claude_light.llm as llm  # noqa: E402
import claude_light.state as state  # noqa: E402
from claude_light import config  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "repo_questions.json"
OUTPUT = {"low": 600, "medium": 1_500, "high": 3_000, "max": 6_000}  # thinking + answer, assumed
ANSWER = 400      # answer tokens kept in history, assumed
SUMMARY = 420     # the summary pair compression leaves, assumed
RANK = {config.MODEL_HAIKU: 0, config.MODEL_SONNET: 1, config.MODEL_OPUS: 2}
POLICIES = ("old", "effort", "pinned")


def prefix_tokens():
    """Skeleton + tools + system prompt, at 4 characters per token."""
    from claude_light import tools
    from claude_light.skeleton import build_skeleton
    chars = len(build_skeleton()) + len(json.dumps(tools.TOOLS)) + len(config.SYSTEM_PROMPT)
    return chars // 4


def route_all():
    from sentence_transformers import SentenceTransformer
    state.EMBED_MODEL = config.FAST_EMBED_MODEL
    state.embedder = SentenceTransformer(config.FAST_EMBED_MODEL, device="cpu")
    state.embedder_ready.set()
    questions = [q["question"] for q in json.loads(FIXTURE.read_text(encoding="utf-8"))["questions"]]
    routes = []
    with contextlib.redirect_stderr(io.StringIO()):   # route_query prints the route line
        for q in questions:
            model, effort, _ = llm.route_query(q)
            routes.append((model, effort, len(q) // 4 + 10))
    return routes


def simulate(routes, policy, prefix, gap, output_scale, effort_in_prefix):
    total, history = 0.0, []
    prefix_cache, history_cache = {}, {}   # key -> minute written / (minute used, tokens)
    session = None
    for i, (model, effort, q_tokens) in enumerate(routes):
        now = i * gap
        if policy == "pinned" and session and RANK[model] < RANK[session]:
            model = session
        session = model
        p = config.MODEL_PRICES[model]
        varies = policy == "old"           # only 'old' renders the routed effort at the top level
        pkey = (model, effort) if varies and effort_in_prefix else model
        if pkey in prefix_cache and now - prefix_cache[pkey] < 60:
            total += prefix * p["read"] / 1e6
        else:
            total += prefix * p["write_1h"] / 1e6
            prefix_cache[pkey] = now
        hkey = (model, effort) if varies else model
        h = sum(history)
        hit = history_cache.get(hkey)
        cached = min(hit[1], h) if hit and now - hit[0] < 5 else 0
        total += cached * p["read"] / 1e6 + (h - cached) * p["write"] / 1e6
        history_cache[hkey] = (now, h)
        # retrieved context + question carry a breakpoint, so they are written
        total += (config._RETRIEVAL_BUDGET[effort] + q_tokens) * p["write"] / 1e6
        total += OUTPUT[effort] * output_scale * p["output"] / 1e6
        history += [q_tokens, ANSWER]
        if len(history) > config.MAX_HISTORY_TURNS * 2:
            history = [SUMMARY] + history[config.SUMMARIZE_BATCH * 2:]
            history_cache.clear()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=float, default=2.0, help="minutes between turns (default 2)")
    ap.add_argument("--output-scale", type=float, default=1.0, help="multiplies the output assumption")
    ap.add_argument("--effort-in-prefix", action="store_true",
                    help="an effort change also invalidates the cached skeleton")
    ap.add_argument("--shuffles", type=int, default=20, help="shuffled session orders (default 20)")
    args = ap.parse_args()

    prefix = prefix_tokens()
    routes = route_all()
    mix = {}
    for model, effort, _ in routes:
        mix[f"{model.split('-')[1]} {effort}"] = mix.get(f"{model.split('-')[1]} {effort}", 0) + 1
    print(f"{len(routes)} questions {mix}; prefix {prefix:,} tokens; gap {args.gap} min; "
          f"output x{args.output_scale}; effort in prefix: {args.effort_in_prefix}")
    orders = [routes] + [random.Random(s).sample(routes, len(routes)) for s in range(args.shuffles)]
    base = None
    for policy in POLICIES:
        costs = [simulate(o, policy, prefix, args.gap, args.output_scale, args.effort_in_prefix)
                 for o in orders]
        mean = sum(costs[1:]) / max(len(costs) - 1, 1)
        base = base or mean
        print(f"  {policy:7} fixture order ${costs[0]:.4f}   shuffled mean ${mean:.4f} "
              f"({100 * (mean / base - 1):+.1f}% vs old)")


if __name__ == "__main__":
    main()
