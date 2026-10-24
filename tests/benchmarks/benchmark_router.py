#!/usr/bin/env python3
"""
Router replay: which effort tier and model each question gets (lever 4 in TOKEN-SAVINGS.md).

Runs the real route_query() with the real static embedder on the questions in
tests/fixtures/repo_questions.json, plus any extra prompts given as a file
(one per line), and reports:

  * how many questions land on each effort tier and model
  * how many scores fall in the band ROUTER_ESCALATE_MARGIN can change: the
    Opus (max) tier is only kept when score >= 12.0 * (1 + margin), so a score
    between 12.0 and 12.0 * (1 + margin) is the only place a larger margin
    moves a question off Opus

    python tests/benchmarks/benchmark_router.py
    python tests/benchmarks/benchmark_router.py extra_prompts.txt --margin 0.30

Quality is not measured here: this only shows where the router would send a
question, and what that costs. Not collected by pytest. Run from the repo root.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import claude_light.llm as llm  # noqa: E402
import claude_light.state as state  # noqa: E402
from claude_light import config  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "repo_questions.json"


def load_queries(extra_path):
    questions = [q["question"] for q in json.loads(FIXTURE.read_text(encoding="utf-8"))["questions"]]
    if extra_path:
        lines = Path(extra_path).read_text(encoding="utf-8").splitlines()
        questions += [line.strip() for line in lines if line.strip()]
    return questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extra", nargs="?", help="file of extra prompts, one per line")
    ap.add_argument("--margin", type=float, default=None,
                    help="ROUTER_ESCALATE_MARGIN to evaluate (default: the configured value)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    state.EMBED_MODEL = config.FAST_EMBED_MODEL
    state.embedder = SentenceTransformer(config.FAST_EMBED_MODEL, device="cpu")
    state.embedder_ready.set()
    if args.margin is not None:
        llm.ROUTER_ESCALATE_MARGIN = args.margin
    margin = llm.ROUTER_ESCALATE_MARGIN

    tiers, models = Counter(), Counter()
    in_band = []
    for q in load_queries(args.extra):
        model, effort, _ = llm.route_query(q)
        info = llm._last_route_info
        score = info["arch"] + info["logic"] + info["infra"]
        tiers[effort] += 1
        models[model] += 1
        if 12.0 <= score < 12.0 * (1.0 + margin):
            in_band.append((score, q))

    total = sum(tiers.values())
    print(f"{total} queries, ROUTER_ESCALATE_MARGIN {margin}")
    for effort in ("max", "high", "medium", "low"):
        print(f"  {effort:7} {tiers[effort]:4}  ({100 * tiers[effort] / total:4.0f}%)")
    print(f"  models: {dict(models)}")
    print(f"  scores in the margin band (12.0 to {12.0 * (1.0 + margin):.1f}): {len(in_band)}")
    for score, q in in_band:
        print(f"    {score:5.2f}  {q[:80]}")


if __name__ == "__main__":
    main()
