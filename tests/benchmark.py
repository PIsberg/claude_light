#!/usr/bin/env python3
"""
benchmark.py -- Token-savings benchmark for claude_light.py

Measures, analytically, how many tokens and dollars each optimization layer
saves vs naively sending the full codebase to Claude on every query.

Three cost scenarios per query:
  Naive -- full codebase at $3.00/M (no RAG, no cache)
  Cold  -- skeleton + RAG chunks at cache-write price $3.75/M (first cache hit)
  Warm  -- skeleton + RAG chunks at cache-read  price $0.30/M (subsequent hits)

Uses the same synthetic data as --test-mode and replicates all formulas
exactly from claude_light.py. No external dependencies required.

Usage:
  python benchmark.py                    # all presets, table output
  python benchmark.py --preset small     # one preset
  python benchmark.py --json             # machine-readable JSON to stdout
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Pricing -- matches claude_light.py exactly (lines 208-211)
# ---------------------------------------------------------------------------
PRICE_INPUT  = 3.00    # $/M uncached input tokens
PRICE_WRITE  = 3.75    # $/M cache-creation tokens (1.25x input)
PRICE_READ   = 0.30    # $/M cache-read tokens     (0.10x input)
PRICE_OUTPUT = 15.00   # $/M output tokens

TARGET_RETRIEVED_TOKENS = 6_000   # denominator in retrieve() k formula

# Per-effort retrieval token budgets -- matches _RETRIEVAL_BUDGET (line 99)
RETRIEVAL_BUDGET = {"low": 1_500, "medium": 3_000, "high": 6_000, "max": 9_000}

# Approximations for query text and expected answer length
QUERY_TOKENS = 20
OUTPUT_TOKENS = {"low": 150, "medium": 300, "high": 500, "max": 1_000}

# ---------------------------------------------------------------------------
# Preset definitions -- matches MockManager configs (lines 1643-1645)
# ---------------------------------------------------------------------------
PRESETS = {
    "small":       (5,    10),
    "medium":      (50,   15),
    "large":       (200,  20),
    "extra-large": (1000, 20),
}

# 10-turn session model: realistic effort distribution
# Turn 1 = cold (cache write); turns 2-10 = warm (cache read)
SESSION_TURNS = [
    "low", "low", "low",
    "medium", "medium", "medium",
    "high", "high", "high",
    "max",
]

# ---------------------------------------------------------------------------
# 16 benchmark queries -- 4 per effort tier
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    # Low effort -- simple lookups
    ("low",    "List all service classes in the codebase"),
    ("low",    "Where is doTask5 defined?"),
    ("low",    "How many methods does Service0 have?"),
    ("low",    "Which files are in the com.synthetic package?"),
    # Medium effort -- explanations and summaries
    ("medium", "Explain what Service3 does and how it initializes"),
    ("medium", "Summarize the doTask methods across all service classes"),
    ("medium", "Describe the relationship between services and their task methods"),
    ("medium", "Describe the overall structure of the synthetic codebase"),
    # High effort -- code generation and refactoring
    ("high",   "Implement a new method addTask that takes a string and logs it"),
    ("high",   "Refactor Service0 to implement a Runnable interface"),
    ("high",   "Write a factory method that creates service instances by name"),
    ("high",   "Create an abstract base class for all service classes"),
    # Max effort -- architecture and deep analysis
    ("max",    "Evaluate the overall architecture trade-offs and scalability of this service design"),
    ("max",    "Provide a comprehensive deep analysis to optimize performance across all service classes"),
    ("max",    "Design a strategy for migrating this architecture to a microservices pattern step by step"),
    ("max",    "Analyze cross-cutting concerns and compare approaches to improve the architecture"),
]

# ---------------------------------------------------------------------------
# Synthetic data templates -- exact replicas of MockManager (lines 1650-1665)
# ---------------------------------------------------------------------------

def _make_method(i: int, m: int) -> str:
    return (
        f"\n    public void doTask{m}() {{\n"
        f"        System.out.println(\"Task {m} in Service {i}\");\n"
        f"        for(int j=0; j<10; j++) {{\n"
        f"            // realistic logic simulated here\n"
        f"        }}\n"
        f"    }}"
    )


def _make_file(i: int, num_methods: int) -> str:
    methods = "".join(_make_method(i, m) for m in range(num_methods))
    return (
        f"package com.synthetic;\n\n"
        f"import java.util.*;\n\n"
        f"public class Service{i} {{\n"
        f"    private String name = \"Service{i}\";\n"
        f"    {methods}\n"
        f"}}\n"
    )


def _make_chunk_text(i: int, m: int) -> str:
    """
    Approximates the chunk format produced by _chunk_with_treesitter
    (claude_light.py:654-662):

        "// {filepath}\\n" + preamble + "\\n" + "    // ...\\n" + node_text.strip() + "\\n"

    preamble  = source lines before the first method (package + imports + class header)
    node_text = the method body as extracted by tree-sitter, then .strip()-ped
    """
    filepath = f"src/main/java/com/synthetic/Service{i}.java"
    preamble = (
        f"package com.synthetic;\n\n"
        f"import java.util.*;\n\n"
        f"public class Service{i} {{\n"
        f"    private String name = \"Service{i}\";"
    )
    # Tree-sitter extracts the method declaration without its leading indent;
    # .strip() removes any remaining leading/trailing whitespace.
    method_stripped = (
        f"public void doTask{m}() {{\n"
        f"        System.out.println(\"Task {m} in Service {i}\");\n"
        f"        for(int j=0; j<10; j++) {{\n"
        f"            // realistic logic simulated here\n"
        f"        }}\n"
        f"    }}"
    )
    return f"// {filepath}\n{preamble}\n    // ...\n{method_stripped}\n"


# ---------------------------------------------------------------------------
# Token computation -- all formulas match claude_light.py exactly
# ---------------------------------------------------------------------------

def compute_naive_tokens(num_files: int, num_methods: int) -> int:
    """Total tokens across all synthetic files (the naive no-RAG baseline)."""
    return sum(len(_make_file(i, num_methods)) // 4 for i in range(num_files))


def compute_skeleton_tokens(num_files: int) -> int:
    """
    Tokens in the skeleton directory tree for the synthetic dataset.

    _build_compressed_tree (claude_light.py:453-522) collapses the
    single-child chain src/main/java/com/synthetic/ into one line and
    brace-groups all .java siblings into one line:

        Directory structure:
        src/main/java/com/synthetic/
          {Service0,Service1,...,ServiceN-1}.java
    """
    stems = sorted(f"Service{i}" for i in range(num_files))
    brace_line = "{" + ",".join(stems) + "}.java"
    skeleton = (
        f"Directory structure:\n"
        f"src/main/java/com/synthetic/\n"
        f"  {brace_line}"
    )
    return len(skeleton) // 4


def compute_avg_chunk_tokens(num_files: int, num_methods: int) -> int:
    """Average tokens per method chunk, used by auto_tune to select TOP_K."""
    total_chars = sum(
        len(_make_chunk_text(i, m))
        for i in range(num_files)
        for m in range(num_methods)
    )
    total_chunks = num_files * num_methods
    return max(1, (total_chars // total_chunks) // 4)


def compute_top_k(avg_chunk_tokens: int) -> int:
    """Exact replica of auto_tune TOP_K formula (claude_light.py:715)."""
    return max(2, min(15, round(TARGET_RETRIEVED_TOKENS / avg_chunk_tokens)))


def compute_retrieved_tokens(
    top_k: int,
    avg_chunk_tokens: int,
    budget: int,
    total_chunks: int,
) -> tuple:
    """
    Exact replica of retrieve() k formula (claude_light.py:1002).

    Assumes all top-k chunks pass the MIN_SCORE threshold -- a realistic
    assumption for a real codebase where queries match relevant code.
    Returns (k, retrieved_tokens).
    """
    k = max(2, round(top_k * budget / TARGET_RETRIEVED_TOKENS))
    k = min(k, total_chunks)
    return k, k * avg_chunk_tokens


def compute_query_costs(
    naive_tok: int,
    skeleton_tok: int,
    retrieved_tok: int,
    effort: str,
) -> dict:
    """
    USD cost for one query under three scenarios.

    Naive  -- full codebase at PRICE_INPUT, no caching
    Cold   -- (skeleton + retrieved) at PRICE_WRITE + query text at PRICE_INPUT
    Warm   -- (skeleton + retrieved) at PRICE_READ  + query text at PRICE_INPUT

    Query text and output tokens are identical across all scenarios.
    Note: actual warm costs are even lower when _dedup_retrieved_context
    eliminates repeated preambles from same-file chunks (5–20% extra saving,
    not modelled here -- these numbers are slightly conservative).
    """
    out_cost = (OUTPUT_TOKENS[effort] / 1_000_000) * PRICE_OUTPUT

    naive_cost = (naive_tok + QUERY_TOKENS) / 1_000_000 * PRICE_INPUT + out_cost
    cold_cost = (
        (skeleton_tok + retrieved_tok) / 1_000_000 * PRICE_WRITE
        + QUERY_TOKENS / 1_000_000 * PRICE_INPUT
        + out_cost
    )
    warm_cost = (
        (skeleton_tok + retrieved_tok) / 1_000_000 * PRICE_READ
        + QUERY_TOKENS / 1_000_000 * PRICE_INPUT
        + out_cost
    )

    def save_pct(actual: float) -> float:
        return (naive_cost - actual) / naive_cost * 100 if naive_cost > 0 else 0.0

    return {
        "naive_cost":     naive_cost,
        "cold_cost":      cold_cost,
        "warm_cost":      warm_cost,
        "cold_save_pct":  save_pct(cold_cost),
        "warm_save_pct":  save_pct(warm_cost),
        "naive_tokens":   naive_tok + QUERY_TOKENS,
        "rag_tokens":     skeleton_tok + retrieved_tok,
    }


# ---------------------------------------------------------------------------
# Stats aggregation
# ---------------------------------------------------------------------------

def compute_preset_stats(preset_name: str) -> dict:
    """Compute all token and cost stats for one preset."""
    num_files, num_methods = PRESETS[preset_name]
    total_chunks = num_files * num_methods

    naive_tokens     = compute_naive_tokens(num_files, num_methods)
    skeleton_tokens  = compute_skeleton_tokens(num_files)
    avg_chunk_tokens = compute_avg_chunk_tokens(num_files, num_methods)
    top_k            = compute_top_k(avg_chunk_tokens)

    per_effort: dict = {}
    for effort, budget in RETRIEVAL_BUDGET.items():
        k, retrieved_tokens = compute_retrieved_tokens(
            top_k, avg_chunk_tokens, budget, total_chunks
        )
        costs = compute_query_costs(naive_tokens, skeleton_tokens, retrieved_tokens, effort)
        per_effort[effort] = {"budget": budget, "k": k, "retrieved_tokens": retrieved_tokens, **costs}

    per_query = []
    for effort, query in BENCHMARK_QUERIES:
        pe = per_effort[effort]
        per_query.append({
            "query":         query,
            "effort":        effort,
            "naive_tokens":  pe["naive_tokens"],
            "rag_tokens":    pe["rag_tokens"],
            "naive_cost":    pe["naive_cost"],
            "cold_cost":     pe["cold_cost"],
            "warm_cost":     pe["warm_cost"],
            "cold_save_pct": pe["cold_save_pct"],
            "warm_save_pct": pe["warm_save_pct"],
        })

    return {
        "preset":            preset_name,
        "files":             num_files,
        "methods_per_file":  num_methods,
        "total_chunks":      total_chunks,
        "naive_tokens":      naive_tokens,
        "skeleton_tokens":   skeleton_tokens,
        "avg_chunk_tokens":  avg_chunk_tokens,
        "top_k":             top_k,
        "per_effort":        per_effort,
        "per_query":         per_query,
    }


def simulate_session(stats: dict) -> dict:
    """
    Model a 10-turn session: [lowx3, mediumx3, highx3, maxx1].

    Turn 1  → cold  (skeleton + chunks written to cache at PRICE_WRITE)
    Turn 2+ → warm  (skeleton + chunks read from cache at PRICE_READ)

    Returns naive / cold / warm total costs and savings percentages.
    """
    naive_total = cold_total = warm_total = 0.0

    for turn_idx, effort in enumerate(SESSION_TURNS):
        pe = stats["per_effort"][effort]
        naive_total += pe["naive_cost"]
        cold_total  += pe["cold_cost"]
        warm_total  += pe["cold_cost"] if turn_idx == 0 else pe["warm_cost"]

    def save_pct(actual: float) -> float:
        return (naive_total - actual) / naive_total * 100 if naive_total > 0 else 0.0

    return {
        "turns":           len(SESSION_TURNS),
        "naive_total":     naive_total,
        "cold_total":      cold_total,
        "warm_total":      warm_total,
        "cold_save_pct":   save_pct(cold_total),
        "warm_save_pct":   save_pct(warm_total),
        "naive_vs_warm_x": naive_total / warm_total if warm_total > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Table formatting -- stdlib only
# ---------------------------------------------------------------------------

def _table(headers: list, rows: list, aligns: list) -> str:
    """Render a plain-text box-drawn table with column alignment."""
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells: list) -> str:
        parts = []
        for i, c in enumerate(cells):
            text = str(c)
            padded = text.rjust(widths[i]) if aligns[i] == "r" else text.ljust(widths[i])
            parts.append(f" {padded} ")
        return "|" + "|".join(parts) + "|"

    lines = [sep, fmt_row(headers), sep, *[fmt_row(r) for r in rows], sep]
    return "\n".join(lines)


def _d(v: float) -> str: return f"${v:.4f}"
def _p(v: float) -> str: return f"{v:.1f}%"
def _n(v: int)   -> str: return f"{v:,}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _embed_model(num_files: int) -> str:
    if num_files < 50:  return "all-MiniLM-L6-v2"
    if num_files < 200: return "all-mpnet-base-v2"
    return "nomic-embed-text-v1.5"


def print_preset_overview(all_stats: list) -> None:
    print("\n" + "=" * 72)
    print("  PRESET OVERVIEW")
    print("=" * 72)
    headers = ["Preset", "Files", "Chunks", "Avg chunk tok", "TOP_K", "Skeleton tok", "Naive tok"]
    aligns  = ["l",     "r",     "r",      "r",             "r",     "r",            "r"]
    rows = [
        [
            s["preset"],
            _n(s["files"]),
            _n(s["total_chunks"]),
            _n(s["avg_chunk_tokens"]),
            s["top_k"],
            _n(s["skeleton_tokens"]),
            _n(s["naive_tokens"]),
        ]
        for s in all_stats
    ]
    print(_table(headers, rows, aligns))
    print(
        "\n  Naive = full codebase every turn at $3.00/M\n"
        "  Cold  = skeleton+RAG at cache-write price $3.75/M (first access)\n"
        "  Warm  = skeleton+RAG at cache-read  price $0.30/M (subsequent hits)\n"
        "  Note: warm costs are slightly conservative -- preamble deduplication\n"
        "        (_dedup_retrieved_context) saves an additional 5-20% on top.\n"
    )


def print_per_query_table(stats: dict) -> None:
    embed = _embed_model(stats["files"])
    print(f"\n{'=' * 104}")
    print(
        f"  PRESET: {stats['preset'].upper()}  |  "
        f"{stats['files']} files x {stats['methods_per_file']} methods/file  |  "
        f"Embed: {embed}"
    )
    print(
        f"  Naive baseline: {_n(stats['naive_tokens'])} tok  |  "
        f"Skeleton: {_n(stats['skeleton_tokens'])} tok  |  "
        f"Avg chunk: {_n(stats['avg_chunk_tokens'])} tok  |  "
        f"TOP_K: {stats['top_k']}"
    )
    print("=" * 104)

    headers = ["Query", "Effort", "Naive tok", "RAG tok", "Naive $", "Cold $", "Warm $", "Cold save", "Warm save"]
    aligns  = ["l",     "l",      "r",         "r",       "r",       "r",      "r",      "r",         "r"]
    rows = [
        [
            q["query"][:44] + ("..." if len(q["query"]) > 44 else ""),
            q["effort"],
            _n(q["naive_tokens"]),
            _n(q["rag_tokens"]),
            _d(q["naive_cost"]),
            _d(q["cold_cost"]),
            _d(q["warm_cost"]),
            _p(q["cold_save_pct"]),
            _p(q["warm_save_pct"]),
        ]
        for q in stats["per_query"]
    ]
    print(_table(headers, rows, aligns))


def print_effort_aggregate(all_stats: list) -> None:
    print(f"\n{'=' * 80}")
    print("  SAVINGS BY EFFORT TIER  (averaged across all presets)")
    print("=" * 80)
    headers = ["Effort", "Budget (tok)", "Avg k", "Avg RAG tok", "Cold save%", "Warm save%"]
    aligns  = ["l",     "r",           "r",    "r",            "r",          "r"]
    rows = []
    for effort in ("low", "medium", "high", "max"):
        cold_saves, warm_saves, ks, rag_toks = [], [], [], []
        for s in all_stats:
            pe = s["per_effort"][effort]
            cold_saves.append(pe["cold_save_pct"])
            warm_saves.append(pe["warm_save_pct"])
            ks.append(pe["k"])
            rag_toks.append(pe["rag_tokens"])
        rows.append([
            effort,
            _n(RETRIEVAL_BUDGET[effort]),
            f"{sum(ks) / len(ks):.1f}",
            _n(int(sum(rag_toks) / len(rag_toks))),
            _p(sum(cold_saves) / len(cold_saves)),
            _p(sum(warm_saves) / len(warm_saves)),
        ])
    print(_table(headers, rows, aligns))


def print_session_table(all_stats: list) -> None:
    dist = {e: SESSION_TURNS.count(e) for e in ("low", "medium", "high", "max")}
    dist_str = "  +  ".join(f"{v}x{k}" for k, v in dist.items() if v)
    print(f"\n{'=' * 80}")
    print(f"  10-TURN SESSION PROJECTION  ({dist_str})")
    print(f"  Turn 1 = cold (cache write), Turns 2-10 = warm (cache read)")
    print("=" * 80)
    headers = ["Preset", "Naive 10Q $", "Cold 10Q $", "Warm 10Q $", "Cold save%", "Warm save%", "Naive/Warm x"]
    aligns  = ["l",     "r",           "r",          "r",          "r",          "r",          "r"]
    rows = [
        [
            s["preset"],
            _d(s["session"]["naive_total"]),
            _d(s["session"]["cold_total"]),
            _d(s["session"]["warm_total"]),
            _p(s["session"]["cold_save_pct"]),
            _p(s["session"]["warm_save_pct"]),
            f"{s['session']['naive_vs_warm_x']:.1f}x",
        ]
        for s in all_stats
    ]
    print(_table(headers, rows, aligns))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-savings benchmark for claude_light.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS) + ["all"],
        default="all",
        help="Which preset(s) to benchmark (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout",
    )
    args = parser.parse_args()

    preset_names = list(PRESETS) if args.preset == "all" else [args.preset]

    if not args.json:
        print(
            f"\nClaude Light -- Token Savings Benchmark\n"
            f"Pricing: input ${PRICE_INPUT:.2f}/M  |  "
            f"cache-write ${PRICE_WRITE:.2f}/M  |  "
            f"cache-read ${PRICE_READ:.2f}/M  |  "
            f"output ${PRICE_OUTPUT:.2f}/M"
        )

    all_stats = []
    for name in preset_names:
        if not args.json:
            print(f"  Computing {name}...", end="", flush=True)
        stats = compute_preset_stats(name)
        stats["session"] = simulate_session(stats)
        all_stats.append(stats)
        if not args.json:
            print(" done.")

    if args.json:
        print(json.dumps(all_stats, indent=2))
        return

    print_preset_overview(all_stats)
    for stats in all_stats:
        print_per_query_table(stats)
    print_effort_aggregate(all_stats)
    print_session_table(all_stats)


if __name__ == "__main__":
    main()
