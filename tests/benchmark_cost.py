"""benchmark_cost.py — Real-World Token Cost Benchmark for claude_light.py

PURPOSE
-------
Measures actual API token usage and cost savings when claude_light.py is used
against real open-source Python repositories. Unlike benchmark.py (which uses
analytical formulas on synthetic data), this script runs claude_light.py as a
subprocess and parses the real stats it emits to stderr.

WHAT IT MEASURES
----------------
For each (repo, query) pair the script records:
  - total_input    : all input tokens (regular + cache_write + cache_read)
  - cache_read     : tokens served from cache (at $0.30/M)
  - cache_write    : tokens written to cache (at $3.75/M)
  - actual_cost    : what was actually charged (mixed pricing)
  - nocache_cost   : what the same tokens would cost at full $3.00/M + output
  - naive_cost     : cost of sending the ENTIRE repo on every query at $3.00/M
  - true_savings   : (naive_cost - actual_cost) / naive_cost

METHODOLOGY
-----------
1.  Clone each repo at a fixed tagged commit (shallow, --depth 1).
2.  For each of 10 standard queries, run:
      python claude_light.py "query"
    in one-shot mode (stdin=/dev/null so no interactive prompts).
3.  Strip ANSI codes from stderr and extract the [Stats], [Cost], and [Router]
    lines with regex.
4.  Compute naive_cost by counting all source tokens in the repo (same
    extensions + SKIP_DIRS as claude_light.py) and pricing them at PRICE_INPUT.

ASSUMPTIONS / LIMITATIONS
--------------------------
- Requires ANTHROPIC_API_KEY to be set — incurs real API costs (~$0.05–$0.50
  per full run depending on cache hit rate and repo size).
- Query 1 for each repo is always a cold-cache run (cache_write dominant).
  Queries 2–10 benefit from warm-cache reads.
- "Naive cost" assumes the entire repo is sent on every single query at full
  input price — this is the worst-case baseline against which the tool competes.
- The [Stats] line is written to stderr by claude_light.py's print_stats().
  If the line format changes, update the STATS_RE / COST_RE / ROUTER_RE patterns.
- tree-sitter is recommended for chunking; without it claude_light.py falls back
  to whole-file mode which may affect retrieval quality but not cost measurement.

DEPENDENCIES
------------
  pip install anthropic    # required by claude_light.py
  git                      # for cloning repos

USAGE
-----
  # Run all repos, all queries (real API calls — incurs costs)
  python benchmark_cost.py

  # Only one repo
  python benchmark_cost.py --repo pallets/flask

  # Use pre-cloned repos
  python benchmark_cost.py --repos-dir /path/to/repos

  # Machine-readable output
  python benchmark_cost.py --json

  # Dry-run: print queries without making API calls
  python benchmark_cost.py --dry-run

OUTPUT
------
Per-repo table:  Query | Effort | Model | Input tok | Cache% | Actual $ | Naive $ | True save%
Aggregate table: Repo  | Queries | Total actual $ | Total naive $ | True save%
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pricing constants — must match claude_light.py exactly
# ---------------------------------------------------------------------------
PRICE_INPUT  = 3.00   # $/M uncached input tokens
PRICE_WRITE  = 3.75   # $/M cache-write tokens
PRICE_READ   = 0.30   # $/M cache-read tokens
PRICE_OUTPUT = 15.00  # $/M output tokens

# Extensions indexed by claude_light.py (from _WANTED_LANGS + .md)
INDEXABLE_EXTS = {".java", ".py", ".js", ".go", ".rs", ".ts", ".tsx", ".md"}

# Directories skipped by claude_light.py (from SKIP_DIRS)
SKIP_DIRS = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

# ---------------------------------------------------------------------------
# Repos — 4 popular Python projects at fixed stable tags
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Standard queries — 10 queries spanning all 4 effort tiers
# Generic enough to apply to any Python web library
# ---------------------------------------------------------------------------
STANDARD_QUERIES = [
    # low effort (simple lookups)
    ("low",    "List all public classes or modules in this codebase"),
    ("low",    "Where is the main entry point defined?"),
    ("low",    "What Python version is required by this project?"),
    # medium effort (explanations)
    ("medium", "Explain how HTTP sessions or connections are managed"),
    ("medium", "Summarize the error handling and exception hierarchy"),
    ("medium", "Describe how routing or request dispatching works"),
    # high effort (code generation / analysis)
    ("high",   "Show me an example of adding a custom middleware or hook"),
    ("high",   "Implement a helper function that retries a request on 5xx errors with exponential backoff"),
    ("high",   "Identify potential thread-safety issues and suggest fixes"),
    # max effort (architecture)
    ("max",    "Evaluate the overall architecture trade-offs and scalability of this codebase"),
]

# ---------------------------------------------------------------------------
# Regex patterns — match claude_light.py stderr output after ANSI stripping
#
# [Stats]  12,345 tokens  |  cached 10,000 (81.0%)  |  new 2,345
# [Cost]   $0.0023  |  saved $0.0187 (89.0% vs no-cache)  |  session $0.0023
# [Router] effort=high  model=sonnet
# ---------------------------------------------------------------------------
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

STATS_RE  = re.compile(
    r"\[Stats\]\s+([\d,]+)\s+tokens\s+\|\s+cached\s+([\d,]+)\s+\(([\d.]+)%\)\s+\|\s+new\s+([\d,]+)"
)
COST_RE   = re.compile(
    r"\[Cost\]\s+\$([\d.]+)\s+\|\s+saved\s+\$(-?[\d.]+)\s+\((-?[\d.]+)%\s+vs\s+no-cache\)\s+\|\s+session\s+\$([\d.]+)"
)
ROUTER_RE = re.compile(
    r"\[Router\]\s+effort=(\w+)\s+model=(\w+)"
)


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def parse_stderr(stderr_text: str) -> dict | None:
    """Extract stats from one claude_light.py stderr dump. Returns None on parse failure."""
    clean = strip_ansi(stderr_text)
    sm = STATS_RE.search(clean)
    cm = COST_RE.search(clean)
    rm = ROUTER_RE.search(clean)
    if not (sm and cm):
        return None
    return {
        "total_input":    int(sm.group(1).replace(",", "")),
        "cache_read":     int(sm.group(2).replace(",", "")),
        "hit_pct":        float(sm.group(3)),
        "cache_write":    int(sm.group(4).replace(",", "")),
        "actual_cost":    float(cm.group(1)),
        "saved_nocache":  float(cm.group(2)),
        "savings_pct":    float(cm.group(3)),
        "session_cost":   float(cm.group(4)),
        "effort":         rm.group(1) if rm else "unknown",
        "model":          rm.group(2) if rm else "unknown",
    }


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

def is_skipped(path: Path) -> bool:
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)


def count_naive_tokens(repo_dir: Path) -> int:
    """Count total tokens in all indexed source files (same logic as claude_light.py)."""
    total = 0
    for f in repo_dir.rglob("*"):
        if f.is_file() and f.suffix in INDEXABLE_EXTS:
            # Check parts relative to repo_dir so the repo_dir prefix itself
            # (which may start with ".") doesn't trigger the skip
            rel = f.relative_to(repo_dir)
            if not is_skipped(rel):
                try:
                    total += len(f.read_bytes()) // 4
                except OSError:
                    pass
    return total


def clone_repo(repo_key: str, repos_dir: Path) -> Path:
    """Clone a repo at a fixed ref into repos_dir/owner__name/. Returns the path."""
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
# Running claude_light.py
# ---------------------------------------------------------------------------

def run_query(claude_light_path: Path, repo_dir: Path, query: str,
              timeout: int = 120) -> dict:
    """Run claude_light.py in one-shot mode and parse its output."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, str(claude_light_path), query],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        env=env,
    )
    parsed = parse_stderr(result.stderr)
    if parsed is None:
        # Return a failure record so the rest of the run continues
        return {
            "ok": False,
            "stderr": result.stderr[-2000:],  # last 2 KB for debugging
            "returncode": result.returncode,
        }
    parsed["ok"] = True
    parsed["returncode"] = result.returncode
    return parsed


# ---------------------------------------------------------------------------
# True savings computation
# ---------------------------------------------------------------------------

def compute_true_savings(stats: dict, naive_tokens: int) -> dict:
    """
    Compute cost vs naive baseline (sending whole repo every query).

    claude_light's [Cost] line reports savings vs "no-cache" (same tokens at
    full price). True savings compares against sending the ENTIRE repo.

    nocache_cost = actual_cost + saved_vs_nocache   (from [Cost] line)
    output_cost  = nocache_cost - (total_input/1M) * PRICE_INPUT
    naive_cost   = (naive_tokens/1M) * PRICE_INPUT + output_cost
    true_savings = (naive_cost - actual_cost) / naive_cost
    """
    actual     = stats["actual_cost"]
    nocache    = actual + stats["saved_nocache"]
    output_cost = nocache - (stats["total_input"] / 1_000_000) * PRICE_INPUT
    output_cost = max(output_cost, 0.0)
    naive_cost  = (naive_tokens / 1_000_000) * PRICE_INPUT + output_cost
    true_pct    = (naive_cost - actual) / naive_cost * 100 if naive_cost > 0 else 0.0
    return {
        "nocache_cost": nocache,
        "output_cost":  output_cost,
        "naive_cost":   naive_cost,
        "true_savings_pct": true_pct,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _col(s, w):
    s = str(s)
    return s[:w].ljust(w)


def _print_table(headers, rows, widths):
    sep   = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr   = "| " + " | ".join(_col(h, w) for h, w in zip(headers, widths)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(_col(c, w) for c, w in zip(row, widths)) + " |")
    print(sep)


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

def run_repo_benchmark(repo_key: str, repo_dir: Path, claude_light_path: Path,
                       queries: list, dry_run: bool = False) -> dict:
    """Run all queries against one repo. Returns structured results."""
    print(f"\n{'='*70}")
    print(f"  Repo: {repo_key}  ({REPOS[repo_key]['desc']})")
    print(f"  Path: {repo_dir}")
    print(f"{'='*70}")

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

        try:
            stats = run_query(claude_light_path, repo_dir, query)
        except subprocess.TimeoutExpired:
            print("          [TIMEOUT]")
            query_results.append({"query": query, "effort_hint": effort_hint, "ok": False,
                                  "error": "timeout"})
            continue
        except Exception as exc:
            print(f"          [ERROR: {exc}]")
            query_results.append({"query": query, "effort_hint": effort_hint, "ok": False,
                                  "error": str(exc)})
            continue

        if not stats["ok"]:
            print(f"          [PARSE FAILED — rc={stats['returncode']}]")
            print(f"          stderr tail: {stats.get('stderr','')[-300:]}")
            query_results.append({"query": query, "effort_hint": effort_hint, **stats})
            continue

        savings = compute_true_savings(stats, naive_tokens)
        stats.update(savings)
        stats["query"] = query
        stats["effort_hint"] = effort_hint
        query_results.append(stats)

        print(
            f"          effort={stats['effort']:6s}  model={stats['model']:8s}  "
            f"input={stats['total_input']:>7,}  cache={stats['hit_pct']:5.1f}%  "
            f"actual=${stats['actual_cost']:.4f}  naive=${stats['naive_cost']:.4f}  "
            f"true_save={stats['true_savings_pct']:.1f}%"
        )

    return {
        "repo":         repo_key,
        "naive_tokens": naive_tokens,
        "queries":      query_results,
    }


def print_repo_table(repo_result: dict):
    """Print per-repo query table."""
    repo_key = repo_result["repo"]
    rows     = repo_result["queries"]

    print(f"\n  Results for {repo_key}:")
    headers = ["#", "Query (truncated)", "Effort", "Model",
               "Input tok", "Cache%", "Actual $", "Naive $", "True save%"]
    widths  = [2, 45, 6, 7, 9, 6, 9, 9, 10]

    table_rows = []
    for i, r in enumerate(rows, 1):
        if r.get("dry_run") or not r.get("ok"):
            table_rows.append([
                str(i),
                (r["query"][:43] + "...") if len(r["query"]) > 43 else r["query"],
                r.get("effort_hint", "?"),
                "-", "-", "-", "-", "-", "-",
            ])
        else:
            q_short = (r["query"][:43] + "...") if len(r["query"]) > 43 else r["query"]
            table_rows.append([
                str(i),
                q_short,
                r.get("effort", r.get("effort_hint", "?")),
                r.get("model", "?"),
                f"{r['total_input']:,}",
                f"{r['hit_pct']:.1f}%",
                f"${r['actual_cost']:.4f}",
                f"${r['naive_cost']:.4f}",
                f"{r['true_savings_pct']:.1f}%",
            ])
    _print_table(headers, table_rows, widths)


def print_aggregate_table(all_results: list):
    """Print aggregate table across all repos."""
    print("\n\n  AGGREGATE RESULTS ACROSS ALL REPOS")
    print("  " + "="*68)
    headers = ["Repo", "Desc", "Queries", "Actual $", "Naive $", "True save%"]
    widths  = [22, 30, 7, 9, 9, 10]
    rows    = []
    total_actual = 0.0
    total_naive  = 0.0
    total_ok     = 0

    for r in all_results:
        ok_qs = [q for q in r["queries"] if q.get("ok")]
        if not ok_qs:
            rows.append([r["repo"], REPOS[r["repo"]]["desc"][:28], "0", "-", "-", "-"])
            continue
        actual = sum(q["actual_cost"]  for q in ok_qs)
        naive  = sum(q["naive_cost"]   for q in ok_qs)
        save   = (naive - actual) / naive * 100 if naive > 0 else 0.0
        total_actual += actual
        total_naive  += naive
        total_ok     += len(ok_qs)
        rows.append([
            r["repo"],
            REPOS[r["repo"]]["desc"][:28],
            str(len(ok_qs)),
            f"${actual:.4f}",
            f"${naive:.4f}",
            f"{save:.1f}%",
        ])

    overall_save = (total_naive - total_actual) / total_naive * 100 if total_naive > 0 else 0.0
    rows.append([
        "TOTAL", "", str(total_ok),
        f"${total_actual:.4f}", f"${total_naive:.4f}", f"{overall_save:.1f}%",
    ])
    _print_table(headers, rows, widths)
    print(f"\n  Naive/Actual ratio: {(total_naive/total_actual):.1f}x" if total_actual > 0 else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-world token cost benchmark for claude_light.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        help="Run only one repo (e.g. pallets/flask). Default: all four.",
    )
    parser.add_argument(
        "--repos-dir",
        default=".benchmark_repos",
        metavar="DIR",
        help="Directory where repos are cloned (default: .benchmark_repos)",
    )
    parser.add_argument(
        "--claude-light",
        default=None,
        metavar="PATH",
        help="Path to claude_light.py (default: auto-detect next to this script)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full results as JSON to stdout (tables still go to stderr)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clone repos and count tokens but skip API calls",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECS",
        help="Per-query subprocess timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    # Locate claude_light.py
    if args.claude_light:
        cl_path = Path(args.claude_light).resolve()
    else:
        cl_path = Path(__file__).parent / "claude_light.py"
    if not cl_path.exists():
        print(f"[error] claude_light.py not found at {cl_path}", file=sys.stderr)
        sys.exit(1)

    # API key check (skip for dry-run)
    if not args.dry_run:
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY")
            or _read_key_file()
        )
        if not api_key:
            print(
                "[error] ANTHROPIC_API_KEY is not set.\n"
                "Set it as an environment variable or use --dry-run to skip API calls.",
                file=sys.stderr,
            )
            sys.exit(1)

    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Select repos to run
    if args.repo:
        if args.repo not in REPOS:
            print(f"[error] Unknown repo '{args.repo}'. Choose from: {', '.join(REPOS)}", file=sys.stderr)
            sys.exit(1)
        repo_keys = [args.repo]
    else:
        repo_keys = list(REPOS.keys())

    print(f"claude_light benchmark_cost.py")
    print(f"  Repos:        {', '.join(repo_keys)}")
    print(f"  Queries/repo: {len(STANDARD_QUERIES)}")
    print(f"  Mode:         {'dry-run' if args.dry_run else 'live (incurs API costs)'}")
    print(f"  claude_light: {cl_path}")
    print()

    all_results = []
    for repo_key in repo_keys:
        try:
            repo_dir = clone_repo(repo_key, repos_dir)
        except subprocess.CalledProcessError as e:
            print(f"[error] Failed to clone {repo_key}: {e}", file=sys.stderr)
            continue

        result = run_repo_benchmark(
            repo_key, repo_dir, cl_path,
            STANDARD_QUERIES, dry_run=args.dry_run,
        )
        all_results.append(result)
        print_repo_table(result)

    if len(all_results) > 1:
        print_aggregate_table(all_results)

    if args.json:
        json.dump(all_results, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")


def _read_key_file() -> str | None:
    """Try reading API key from ~/.anthropic or ./.env (mirrors claude_light.py)."""
    home_file = Path.home() / ".anthropic"
    if home_file.exists():
        txt = home_file.read_text().strip()
        if txt.startswith("sk-"):
            return txt
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


if __name__ == "__main__":
    main()
