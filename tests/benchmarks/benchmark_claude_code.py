"""benchmark_claude_code.py — Claude Code vs claude_light vs naive baseline

PURPOSE
-------
Measures the real API cost of using Claude Code (the Anthropic CLI) to answer
the same queries used in benchmark_cost.py, then compares three approaches:

  1. Naive baseline  — sending the entire repo as context on every call ($3/M)
  2. Claude Code     — the Anthropic CLI, using its built-in tool-calling RAG
  3. claude_light    — hybrid RAG + prompt caching (from a saved benchmark_cost
                       --json run, via --claude-light-results)

WHAT CLAUDE CODE DOES
---------------------
Unlike the naive baseline, Claude Code uses tool calls (Read, Bash, Glob, Grep)
to selectively fetch relevant files per query. It also caches its large system
prompt (~33-38K tokens of instructions + tool definitions), so warm queries pay
cache-read rates ($0.30/M) rather than full input rate ($3/M) for that overhead.

Each query in this benchmark starts a completely fresh Claude Code session
(--no-session-persistence), isolating each query's cost.

HOW IT WORKS
------------
For each (repo, query) pair the script runs:

    claude --print --output-format json --no-session-persistence \
           --dangerously-skip-permissions "<query>"

from inside the repo directory, with ANTHROPIC_API_KEY unset so Claude Code
authenticates via OAuth (your claude.ai subscription). The JSON response
contains total_cost_usd and a full usage breakdown.

ASSUMPTIONS / LIMITATIONS
--------------------------
- Claude Code must be authenticated: run `claude auth login` first.
- Costs are charged to your claude.ai subscription (not an API key).
- Each query starts a cold Claude Code session, so system-prompt cache-writes
  are counted on the first turn of every query. In real interactive use the
  system prompt stays cached across turns.
- Claude Code's tool-selection is non-deterministic; costs may vary run-to-run.
- The naive baseline is the same analytical formula used in benchmark_cost.py
  (entire repo at $3.00/M per query).

USAGE
-----
  # Run Claude Code benchmark (OAuth — no API key needed)
  python benchmark_claude_code.py

  # Only one repo
  python benchmark_claude_code.py --repo pallets/flask

  # Three-way comparison: also show claude_light numbers
  python benchmark_cost.py --json > cl_results.json
  python benchmark_claude_code.py --claude-light-results cl_results.json

  # Machine-readable output
  python benchmark_claude_code.py --json

  # Dry-run: list queries without making API calls
  python benchmark_claude_code.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared config — keep in sync with benchmark_cost.py
# ---------------------------------------------------------------------------
PRICE_INPUT  = 3.00   # $/M uncached input tokens
PRICE_WRITE  = 3.75   # $/M cache-write tokens
PRICE_READ   = 0.30   # $/M cache-read tokens
PRICE_OUTPUT = 15.00  # $/M output tokens

INDEXABLE_EXTS = {".java", ".py", ".js", ".go", ".rs", ".ts", ".tsx", ".md"}
SKIP_DIRS = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

REPOS = {
    "psf/requests": {
        "url":    "https://github.com/psf/requests.git",
        "ref":    "v2.31.0",
        "desc":   "HTTP library for Python",
    },
    "pallets/flask": {
        "url":    "https://github.com/pallets/flask.git",
        "ref":    "3.0.0",
        "desc":   "Lightweight WSGI web framework",
    },
    "encode/httpx": {
        "url":    "https://github.com/encode/httpx.git",
        "ref":    "0.25.2",
        "desc":   "Next-generation HTTP client",
    },
    "bottlepy/bottle": {
        "url":    "https://github.com/bottlepy/bottle.git",
        "ref":    "0.13.2",
        "desc":   "Fast, simple micro web-framework",
    },
}

STANDARD_QUERIES = [
    ("low",    "List all public classes or modules in this codebase"),
    ("low",    "Where is the main entry point defined?"),
    ("low",    "What Python version is required by this project?"),
    ("medium", "Explain how HTTP sessions or connections are managed"),
    ("medium", "Summarize the error handling and exception hierarchy"),
    ("medium", "Describe how routing or request dispatching works"),
    ("high",   "Show me an example of adding a custom middleware or hook"),
    ("high",   "Implement a helper function that retries a request on 5xx errors with exponential backoff"),
    ("high",   "Identify potential thread-safety issues and suggest fixes"),
    ("max",    "Evaluate the overall architecture trade-offs and scalability of this codebase"),
]


# ---------------------------------------------------------------------------
# Naive baseline (same formula as benchmark_cost.py)
# ---------------------------------------------------------------------------

def is_skipped(path: Path) -> bool:
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)


def count_naive_tokens(repo_dir: Path) -> int:
    total = 0
    for f in repo_dir.rglob("*"):
        if f.is_file() and f.suffix in INDEXABLE_EXTS:
            rel = f.relative_to(repo_dir)
            if not is_skipped(rel):
                try:
                    total += len(f.read_bytes()) // 4
                except OSError:
                    pass
    return total


def naive_cost_for_query(naive_tokens: int, output_cost: float = 0.0) -> float:
    return (naive_tokens / 1_000_000) * PRICE_INPUT + output_cost


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

def clone_repo(repo_key: str, repos_dir: Path) -> Path:
    info = REPOS[repo_key]
    safe_name = repo_key.replace("/", "__")
    dest = repos_dir / safe_name
    if dest.exists():
        print(f"  [clone] {repo_key} already exists at {dest}", flush=True)
        return dest
    print(f"  [clone] Cloning {repo_key}@{info['ref']} ...", flush=True)
    subprocess.run(
        ["git", "clone", "--branch", info["ref"], "--depth", "1",
         "--quiet", info["url"], str(dest)],
        check=True,
    )
    return dest


# ---------------------------------------------------------------------------
# Running Claude Code
# ---------------------------------------------------------------------------

def run_claude_code(repo_dir: Path, query: str, timeout: int = 300) -> dict:
    """Run `claude --print --output-format json` and parse the result."""
    env = os.environ.copy()
    # Use OAuth auth (claude.ai subscription), not API key
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--output-format", "json",
                "--no-session-persistence",
                "--dangerously-skip-permissions",
                query,
            ],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except FileNotFoundError:
        return {"ok": False, "error": "claude CLI not found — install Claude Code first"}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "JSON parse failed",
            "stdout_tail": result.stdout[-500:],
            "stderr_tail": result.stderr[-500:],
        }

    if data.get("is_error"):
        return {"ok": False, "error": str(data.get("result", "unknown"))[:300]}

    usage = data.get("usage", {})
    model_usage = data.get("modelUsage", {})

    # Derive which model(s) were used
    models_used = list(model_usage.keys())
    primary_model = models_used[0] if models_used else "unknown"

    total_input  = usage.get("input_tokens", 0)
    cache_write  = usage.get("cache_creation_input_tokens", 0)
    cache_read   = usage.get("cache_read_input_tokens", 0)
    output_tok   = usage.get("output_tokens", 0)
    all_input    = total_input + cache_write + cache_read

    return {
        "ok":            True,
        "cost":          data["total_cost_usd"],
        "input_tokens":  total_input,
        "cache_write":   cache_write,
        "cache_read":    cache_read,
        "all_input":     all_input,
        "output_tokens": output_tok,
        "num_turns":     data.get("num_turns", 1),
        "model":         primary_model,
        "duration_ms":   data.get("duration_ms", 0),
        "model_usage":   model_usage,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _col(s, w):
    s = str(s)
    return s[:w].ljust(w)


def _print_table(headers, rows, widths):
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(_col(h, w) for h, w in zip(headers, widths)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(_col(c, w) for c, w in zip(row, widths)) + " |")
    print(sep)


# ---------------------------------------------------------------------------
# Per-repo benchmark
# ---------------------------------------------------------------------------

def run_repo_benchmark(repo_key: str, repo_dir: Path, queries: list,
                       dry_run: bool = False, timeout: int = 300) -> dict:
    print(f"\n{'='*72}")
    print(f"  Repo: {repo_key}  ({REPOS[repo_key]['desc']})")
    print(f"  Path: {repo_dir}")
    print(f"{'='*72}")

    naive_tokens = count_naive_tokens(repo_dir)
    print(f"  Naive repo size: {naive_tokens:,} tokens  ({naive_tokens/1_000_000:.3f}M)")

    query_results = []

    for i, (effort_hint, query) in enumerate(queries, 1):
        short_q = query[:55] + "..." if len(query) > 55 else query
        print(f"\n  Q{i:02d} [{effort_hint:6s}] {short_q}", flush=True)

        if dry_run:
            print("          [dry-run — skipping API call]")
            query_results.append({"query": query, "effort_hint": effort_hint, "dry_run": True})
            continue

        stats = run_claude_code(repo_dir, query, timeout=timeout)

        if not stats["ok"]:
            print(f"          [FAILED: {stats.get('error', '?')}]")
            query_results.append({"query": query, "effort_hint": effort_hint, **stats})
            continue

        # Estimate output cost from actual output tokens
        output_cost = (stats["output_tokens"] / 1_000_000) * PRICE_OUTPUT
        naive_cost  = naive_cost_for_query(naive_tokens, output_cost)
        save_pct    = (naive_cost - stats["cost"]) / naive_cost * 100 if naive_cost > 0 else 0.0

        stats["naive_cost"]  = naive_cost
        stats["output_cost"] = output_cost
        stats["savings_pct"] = save_pct
        stats["query"]       = query
        stats["effort_hint"] = effort_hint
        query_results.append(stats)

        cache_total = stats["cache_write"] + stats["cache_read"]
        cache_pct   = cache_total / stats["all_input"] * 100 if stats["all_input"] > 0 else 0.0

        print(
            f"          model={stats['model'][:20]}  turns={stats['num_turns']}  "
            f"all_input={stats['all_input']:>7,}  cache={cache_pct:5.1f}%  "
            f"actual=${stats['cost']:.4f}  naive=${naive_cost:.4f}  "
            f"save={save_pct:.1f}%"
        )

    return {
        "repo":         repo_key,
        "naive_tokens": naive_tokens,
        "queries":      query_results,
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_repo_table(repo_result: dict):
    repo_key = repo_result["repo"]
    print(f"\n  Results for {repo_key}:")
    headers = ["#", "Query (truncated)", "Turns", "Model (short)",
               "All input", "Cache%", "Actual $", "Naive $", "Save%"]
    widths  = [2, 44, 5, 14, 9, 6, 9, 9, 7]

    rows = []
    for i, r in enumerate(repo_result["queries"], 1):
        q_short = (r["query"][:42] + "..") if len(r["query"]) > 42 else r["query"]
        if r.get("dry_run") or not r.get("ok"):
            rows.append([str(i), q_short, "-", "-", "-", "-", "-", "-", "-"])
            continue
        cache_total = r["cache_write"] + r["cache_read"]
        cache_pct   = cache_total / r["all_input"] * 100 if r["all_input"] > 0 else 0.0
        # Shorten model name: "claude-sonnet-4-6" -> "sonnet-4-6"
        model_short = r["model"].replace("claude-", "") if r["model"] != "unknown" else "?"
        rows.append([
            str(i), q_short,
            str(r["num_turns"]),
            model_short,
            f"{r['all_input']:,}",
            f"{cache_pct:.0f}%",
            f"${r['cost']:.4f}",
            f"${r['naive_cost']:.4f}",
            f"{r['savings_pct']:.1f}%",
        ])
    _print_table(headers, rows, widths)


def print_aggregate_table(all_results: list, cl_results: dict | None = None):
    """Print aggregate table. Optionally include claude_light column."""
    print("\n\n  AGGREGATE RESULTS")
    print("  " + "="*80)

    has_cl = cl_results is not None
    if has_cl:
        headers = ["Repo", "Naive $", "Claude Code $", "CC save%", "claude_light $", "CL save%", "CC vs CL"]
        widths  = [20, 10, 14, 9, 15, 9, 10]
    else:
        headers = ["Repo", "Desc", "Queries", "Naive $", "Claude Code $", "Save%"]
        widths  = [20, 28, 7, 10, 14, 7]

    rows = []
    total_naive = 0.0
    total_cc    = 0.0
    total_cl    = 0.0
    total_ok    = 0

    for r in all_results:
        ok_qs = [q for q in r["queries"] if q.get("ok")]
        if not ok_qs:
            if has_cl:
                rows.append([r["repo"], "-", "-", "-", "-", "-", "-"])
            else:
                rows.append([r["repo"], REPOS[r["repo"]]["desc"][:26], "0", "-", "-", "-"])
            continue

        naive = sum(q["naive_cost"] for q in ok_qs)
        cc    = sum(q["cost"]       for q in ok_qs)
        cc_save = (naive - cc) / naive * 100 if naive > 0 else 0.0
        total_naive += naive
        total_cc    += cc
        total_ok    += len(ok_qs)

        if has_cl:
            # Look up claude_light result for this repo
            cl_repo = next((x for x in cl_results if x["repo"] == r["repo"]), None)
            if cl_repo:
                cl_ok = [q for q in cl_repo["queries"] if q.get("ok")]
                cl_cost = sum(q["actual_cost"] for q in cl_ok)
                cl_save = (naive - cl_cost) / naive * 100 if naive > 0 else 0.0
                ratio = cc / cl_cost if cl_cost > 0 else float("inf")
                total_cl += cl_cost
                rows.append([
                    r["repo"],
                    f"${naive:.2f}",
                    f"${cc:.4f}",
                    f"{cc_save:.1f}%",
                    f"${cl_cost:.4f}",
                    f"{cl_save:.1f}%",
                    f"{ratio:.1f}× more",
                ])
            else:
                rows.append([r["repo"], f"${naive:.2f}", f"${cc:.4f}", f"{cc_save:.1f}%", "-", "-", "-"])
        else:
            rows.append([
                r["repo"],
                REPOS[r["repo"]]["desc"][:26],
                str(len(ok_qs)),
                f"${naive:.4f}",
                f"${cc:.4f}",
                f"{cc_save:.1f}%",
            ])

    overall_cc_save = (total_naive - total_cc) / total_naive * 100 if total_naive > 0 else 0.0

    if has_cl and total_cl > 0:
        overall_cl_save = (total_naive - total_cl) / total_naive * 100 if total_naive > 0 else 0.0
        ratio = total_cc / total_cl
        rows.append([
            "TOTAL",
            f"${total_naive:.2f}",
            f"${total_cc:.4f}",
            f"{overall_cc_save:.1f}%",
            f"${total_cl:.4f}",
            f"{overall_cl_save:.1f}%",
            f"{ratio:.1f}× more",
        ])
    else:
        rows.append([
            "TOTAL", "" if has_cl else "", str(total_ok),
            f"${total_naive:.4f}", f"${total_cc:.4f}", f"{overall_cc_save:.1f}%",
        ] if not has_cl else [
            "TOTAL", f"${total_naive:.2f}", f"${total_cc:.4f}", f"{overall_cc_save:.1f}%",
            "-", "-", "-",
        ])

    _print_table(headers, rows, widths)

    if total_cc > 0:
        cc_ratio = total_naive / total_cc
        print(f"\n  Naive/Claude Code ratio: {cc_ratio:.1f}×")
    if has_cl and total_cl > 0:
        cl_ratio = total_naive / total_cl
        cc_cl_ratio = total_cc / total_cl
        print(f"  Naive/claude_light ratio: {cl_ratio:.1f}×")
        print(f"  Claude Code / claude_light ratio: {cc_cl_ratio:.1f}× (Claude Code costs this much more than claude_light)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code vs naive baseline cost benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--repo", metavar="OWNER/NAME",
                        help="Run only one repo (e.g. pallets/flask)")
    parser.add_argument("--repos-dir", default=".benchmark_repos", metavar="DIR",
                        help="Directory where repos are cloned (default: .benchmark_repos)")
    parser.add_argument("--claude-light-results", metavar="JSON_FILE",
                        help="JSON output from `benchmark_cost.py --json` for three-way comparison")
    parser.add_argument("--json", action="store_true",
                        help="Emit full results as JSON to stdout")
    parser.add_argument("--dry-run", action="store_true",
                        help="Clone repos and count tokens but skip API calls")
    parser.add_argument("--timeout", type=int, default=300, metavar="SECS",
                        help="Per-query subprocess timeout in seconds (default: 300)")
    args = parser.parse_args()

    # Check claude CLI is available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[error] `claude` CLI not found. Install Claude Code first.", file=sys.stderr)
        sys.exit(1)

    # Load optional claude_light results
    cl_results = None
    if args.claude_light_results:
        try:
            cl_results = json.loads(Path(args.claude_light_results).read_text())
            print(f"  Loaded claude_light results from {args.claude_light_results}")
        except Exception as e:
            print(f"[warning] Could not load --claude-light-results: {e}", file=sys.stderr)

    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    if args.repo:
        if args.repo not in REPOS:
            print(f"[error] Unknown repo '{args.repo}'. Choose from: {', '.join(REPOS)}", file=sys.stderr)
            sys.exit(1)
        repo_keys = [args.repo]
    else:
        repo_keys = list(REPOS.keys())

    print(f"benchmark_claude_code.py")
    print(f"  Repos:        {', '.join(repo_keys)}")
    print(f"  Queries/repo: {len(STANDARD_QUERIES)}")
    print(f"  Mode:         {'dry-run' if args.dry_run else 'live (charged to claude.ai subscription)'}")
    print(f"  Auth:         OAuth (ANTHROPIC_API_KEY not used)")
    print()

    all_results = []
    for repo_key in repo_keys:
        try:
            repo_dir = clone_repo(repo_key, repos_dir)
        except subprocess.CalledProcessError as e:
            print(f"[error] Failed to clone {repo_key}: {e}", file=sys.stderr)
            continue

        result = run_repo_benchmark(
            repo_key, repo_dir, STANDARD_QUERIES,
            dry_run=args.dry_run, timeout=args.timeout,
        )
        all_results.append(result)
        print_repo_table(result)

    if len(all_results) >= 1:
        print_aggregate_table(all_results, cl_results)

    if args.json:
        json.dump(all_results, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
