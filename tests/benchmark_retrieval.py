#!/usr/bin/env python3
"""
benchmark_retrieval.py -- RAG Retrieval Quality Benchmark
==========================================================

PURPOSE
-------
Tests whether claude_light's embedding + retrieval pipeline correctly surfaces
the *files* that need to change when a developer describes a real bug.

Uses the SWE-bench Lite dataset (princeton-nlp/SWE-bench_Lite): each instance
is a real GitHub issue paired with the actual patch that fixed it. The patch
tells us which files were touched -- the ground truth. We embed the problem
statement, run retrieval, and check whether those files appear in the top-K
results.

This is purely an embedding benchmark. No API key or Anthropic calls are made.

WHAT IT MEASURES
----------------
For K values [5, 10, 15] (matching claude_light's retrieval range):

  Hit@K         -- fraction of instances where >= 1 gold file is in top-K chunks
                   (binary: did we surface ANY relevant file?)
  Recall@K      -- fraction of gold files found in top-K chunks, averaged across
                   instances. e.g. patch touches 3 files; 2 are retrieved → 66.7%
  Precision@K   -- fraction of retrieved *files* that are gold files, averaged.
                   (low by design: we retrieve chunks, not just gold files)
  MRR           -- Mean Reciprocal Rank of the first gold-file chunk. Higher is
                   better; MRR=1.0 means a gold chunk is always rank-1.

METHODOLOGY
-----------
1. Load SWE-bench Lite from HuggingFace (dev split by default: 23 instances).
2. For each instance:
   a. Clone the repo at base_commit (cached in --repos-dir between runs).
   b. Scan all source files using the same SKIP_DIRS + extension filter as
      claude_light.py (lines 106-162 of that script).
   c. Chunk files using the exact same tree-sitter logic as claude_light.py
      (lines 600-678), with whole-file fallback if tree-sitter is missing.
   d. Embed chunks with the same SentenceTransformer model that claude_light
      would auto_tune to (lines 692-700): MiniLM for <50 files, mpnet for
      50-199, nomic for 200+.
   e. Embed the problem_statement (using the same query prefix convention).
   f. Rank chunks by cosine similarity. Extract unique file paths from top-K.
   g. Compare against gold files parsed from the instance's patch field.
3. Cache embeddings per (repo, commit, model) so re-runs are instant.
4. Report per-instance, per-repo, and aggregate metrics.

ASSUMPTIONS AND LIMITATIONS
----------------------------
- File paths: gold files come from `+++ b/path` lines in the unified diff.
  Chunk IDs use `filepath::method` where filepath is relative to the repo root.
  These match when indexing is run from the repo root, which we enforce.
- Truncation: problem statements up to 6 700 chars; we pass them as-is to the
  embedder, which truncates at its own max token limit (typically 256-512 tok).
- MIN_SCORE filter: claude_light drops chunks below cosine similarity 0.45 in
  production. We skip this filter here to get unbiased Hit@K and Recall@K
  curves. Real production recall may be lower for noisy queries.
- Language coverage: SWE-bench Lite is Python-only, so the .py grammar carries
  the full weight. Other grammars are loaded but rarely exercised here.
- New/deleted files: patches that add new files or delete files are excluded
  from gold_files since those files don't exist in the base repo to retrieve.

DEPENDENCIES
------------
Required:
  pip install datasets sentence-transformers numpy

Optional (strongly recommended for accurate method-level chunking):
  pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-go \\
              tree-sitter-rust tree-sitter-javascript tree-sitter-typescript

System:
  git  (for cloning repos)

USAGE
-----
  python benchmark_retrieval.py                         # dev split, all instances
  python benchmark_retrieval.py --split test            # full 300-instance test split
  python benchmark_retrieval.py --n 10                  # first N instances only
  python benchmark_retrieval.py --repo sympy/sympy      # filter to one repo
  python benchmark_retrieval.py --k 5 10 15             # K values to evaluate
  python benchmark_retrieval.py --repos-dir /data/repos # custom clone cache dir
  python benchmark_retrieval.py --json                  # JSON output to stdout

DISK REQUIREMENTS
-----------------
Each unique (repo, commit) pair requires a shallow treeless clone:
  - Small repos (requests, marshmallow): ~30-100 MB per commit
  - Large repos (sympy, django):         ~200-600 MB per commit
  - Dev split (23 instances, ~10 unique repos): ~2-4 GB total
  - Full test split (300 instances):           ~10-30 GB total
Embedding cache adds ~10-50 MB per (repo, commit) depending on repo size.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies -- fail gracefully with clear instructions
# ---------------------------------------------------------------------------

try:
    from datasets import load_dataset
except ImportError:
    sys.exit(
        "[Error] 'datasets' is required:\n"
        "  pip install datasets\n"
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit(
        "[Error] 'sentence-transformers' is required:\n"
        "  pip install sentence-transformers\n"
    )

# ---------------------------------------------------------------------------
# Constants -- mirrored from claude_light.py to ensure identical behaviour
# ---------------------------------------------------------------------------

# K values to evaluate retrieval at (matches claude_light's retrieval range)
DEFAULT_K_VALUES = [5, 10, 15]

# From claude_light.py line 106-109 -- directories to skip during indexing
SKIP_DIRS = {
    ".git", "target", "build", "node_modules",
    ".idea", "__pycache__", ".mvn", ".gradle",
}

# From claude_light.py lines 125-136 -- extensions we index
INDEXABLE_EXTENSIONS = {".java", ".py", ".js", ".go", ".rs", ".ts", ".tsx"}

# Embedding model selection thresholds -- from claude_light.py auto_tune() lines 692-700
# < 50 files -> MiniLM, 50-199 -> mpnet, 200+ -> nomic
EMBED_THRESHOLDS = [
    (50,  "all-MiniLM-L6-v2"),
    (200, "all-mpnet-base-v2"),
    (None, "nomic-ai/nomic-embed-text-v1.5"),
]

# Nomic model needs explicit prefixes -- from claude_light.py lines 204-205
DOC_PREFIX   = {"nomic-ai/nomic-embed-text-v1.5": "search_document: "}
QUERY_PREFIX = {"nomic-ai/nomic-embed-text-v1.5": "search_query: "}

# AST node types per language -- from claude_light.py _WANTED_LANGS lines 126-136
_WANTED_LANGS = {
    ".java": (lambda: _ts_lang("tree_sitter_java"),
              ["method_declaration", "constructor_declaration"]),
    ".py":   (lambda: _ts_lang("tree_sitter_python"),
              ["function_definition", "async_function_definition", "decorated_definition"]),
    ".js":   (lambda: _ts_lang("tree_sitter_javascript"),
              ["function_declaration", "method_definition"]),
    ".go":   (lambda: _ts_lang("tree_sitter_go"),
              ["function_declaration", "method_declaration"]),
    ".rs":   (lambda: _ts_lang("tree_sitter_rust"),
              ["function_item"]),
}

# Child node type names that hold a symbol's identifier
_NAME_CHILD_TYPES = {"identifier", "name", "field_identifier", "type_identifier"}

# ---------------------------------------------------------------------------
# Tree-sitter helpers -- replicated from claude_light.py lines 600-678
# (with references to original line numbers for auditability)
# ---------------------------------------------------------------------------

def _ts_lang(pkg_name: str):
    """Import a tree-sitter language package and return its Language object."""
    from tree_sitter import Language
    mod = __import__(pkg_name)
    fn = getattr(mod, "language", None) or getattr(mod, "LANGUAGE", None)
    return Language(fn())


def load_language_config() -> dict:
    """
    Load tree-sitter grammars.
    Returns {ext: {"lang": Language, "node_types": list} | None}.
    Returns {ext: None} for all extensions if tree-sitter is not installed.
    Replicates claude_light.py _load_languages() logic (lines 139-165).
    """
    try:
        from tree_sitter import Language, Parser  # noqa: F401
        ts_available = True
    except ImportError:
        ts_available = False

    config: dict = {}
    if not ts_available:
        print("[retrieval] tree-sitter not installed; using whole-file fallback.", file=sys.stderr)
        return {ext: None for ext in INDEXABLE_EXTENSIONS}

    for ext, (get_lang, node_types) in _WANTED_LANGS.items():
        try:
            config[ext] = {"lang": get_lang(), "node_types": node_types}
        except Exception:
            config[ext] = None

    try:
        import tree_sitter_typescript as _tspy
        from tree_sitter import Language
        ts_nodes = ["function_declaration", "method_definition", "arrow_function"]
        config[".ts"]  = {"lang": Language(_tspy.language_typescript()), "node_types": ts_nodes}
        config[".tsx"] = {"lang": Language(_tspy.language_tsx()),         "node_types": ts_nodes}
    except Exception:
        config[".ts"] = config[".tsx"] = None

    loaded = [ext for ext, v in config.items() if v is not None]
    print(f"[retrieval] Tree-sitter grammars loaded: {', '.join(loaded) or 'none'}", file=sys.stderr)
    return config


def _walk(node, node_types: list, results: list) -> None:
    """DFS: collect AST nodes whose type is in node_types.
    Replicated from claude_light.py lines 600-605."""
    if node.type in node_types:
        results.append(node)
        return
    for child in node.children:
        _walk(child, node_types, results)


def _extract_symbol_name(node) -> str:
    """Return the identifier string for an AST symbol node.
    Replicated from claude_light.py lines 609-619."""
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in {"function_definition", "async_function_definition"}:
                return _extract_symbol_name(child)
    for child in node.children:
        if child.type in _NAME_CHILD_TYPES:
            return child.text.decode("utf-8", errors="replace")
    return f"{node.type}_{node.start_point[0]}"


def _chunk_with_treesitter(filepath: str, source: str, lang_cfg: dict) -> list:
    """
    Parse source with tree-sitter and return one chunk per symbol.
    Chunk format: "// filepath\\n{preamble}\\n    // ...\\n{method_body}\\n"
    Replicated from claude_light.py lines 622-664.
    """
    from tree_sitter import Parser
    language = lang_cfg["lang"]
    node_types = lang_cfg["node_types"]

    parser = Parser(language)
    tree = parser.parse(bytes(source, "utf-8"))

    symbols: list = []
    _walk(tree.root_node, node_types, symbols)

    if not symbols:
        return [{"id": filepath, "text": source}]

    lines = source.splitlines(keepends=True)
    first_sym_line = symbols[0].start_point[0]
    preamble = "".join(lines[:first_sym_line]).rstrip()

    chunks, seen = [], {}
    for node in symbols:
        name = _extract_symbol_name(node)
        if name in seen:
            seen[name] += 1
            uid = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
            uid = name

        body = source[node.start_byte:node.end_byte]
        text = f"// {filepath}\n{preamble}\n    // ...\n{body.strip()}\n"
        chunks.append({"id": f"{filepath}::{uid}", "text": text})

    return chunks


def chunk_file(filepath: str, source: str, lang_config: dict) -> list:
    """
    Chunk a source file into symbol-level pieces.
    Falls back to whole-file if tree-sitter is unavailable or no symbols found.
    Replicates claude_light.py chunk_file() lines 667-678.
    """
    ext = Path(filepath).suffix.lower()
    cfg = lang_config.get(ext)
    if cfg is None:
        return [{"id": filepath, "text": source}]
    return _chunk_with_treesitter(filepath, source, cfg)


def is_skipped(path: Path) -> bool:
    """Return True if this path should be excluded from indexing.
    Replicates claude_light.py _is_skipped() line 449-450."""
    return any(p in SKIP_DIRS or p.startswith(".") for p in path.parts)


# ---------------------------------------------------------------------------
# Embedding model selection -- replicates auto_tune() logic
# ---------------------------------------------------------------------------

def select_embed_model(n_files: int) -> str:
    """Select embedding model based on file count, matching claude_light auto_tune()."""
    for threshold, model in EMBED_THRESHOLDS:
        if threshold is None or n_files < threshold:
            return model
    return EMBED_THRESHOLDS[-1][1]  # nomic fallback


# ---------------------------------------------------------------------------
# Embedding cache -- persist per (repo, commit, model) to avoid re-embedding
# ---------------------------------------------------------------------------

def _cache_dir(cache_root: Path, repo: str, commit: str, model: str) -> Path:
    slug = f"{repo.replace('/', '__')}_{commit[:8]}_{model.split('/')[-1]}"
    return cache_root / slug


def load_embedding_cache(cache_root: Path, repo: str, commit: str, model: str):
    """Load cached embeddings. Returns (chunk_ids, embeddings) or None."""
    d = _cache_dir(cache_root, repo, commit, model)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != model:
            return None
        ids = json.loads((d / "chunk_ids.json").read_text())
        embs = np.load(d / "embeddings.npy")
        return ids, embs
    except Exception:
        return None


def save_embedding_cache(cache_root: Path, repo: str, commit: str, model: str,
                         chunk_ids: list, embeddings: np.ndarray) -> None:
    d = _cache_dir(cache_root, repo, commit, model)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "n_chunks": len(chunk_ids), "created_at": time.time()}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "chunk_ids.json").write_text(json.dumps(chunk_ids))
    np.save(d / "embeddings.npy", embeddings.astype(np.float32))


# ---------------------------------------------------------------------------
# Repo management -- clone at specific commit, cached by (repo, commit)
# ---------------------------------------------------------------------------

def get_repo_path(repo: str, commit: str, repos_dir: Path) -> Path | None:
    """
    Return path to a local clone of `repo` checked out at `commit`.
    Clones on first access; subsequent calls return the cached path immediately.
    Uses --filter=blob:none (treeless clone) for speed on large repos.
    """
    dest = repos_dir / f"{repo.replace('/', '__')}_{commit[:8]}"
    if dest.exists():
        return dest

    url = f"https://github.com/{repo}.git"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Cloning {repo}@{commit[:8]}...", end="", flush=True)
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)],
            check=True, capture_output=True, timeout=300,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", commit],
            check=True, capture_output=True, timeout=180,
        )
        print(" done")
        return dest
    except Exception as exc:
        print(f" FAILED ({exc.__class__.__name__})")
        shutil.rmtree(dest, ignore_errors=True)
        return None


# ---------------------------------------------------------------------------
# Gold file extraction from unified diff
# ---------------------------------------------------------------------------

def extract_gold_files(patch: str) -> set:
    """
    Parse a unified diff and return the set of files that were modified.
    Format: lines starting with '+++ b/filepath' give the destination file.
    New files (--- /dev/null) are excluded because they don't exist in the
    base repo and therefore cannot be retrieved.
    Deleted files (+++ /dev/null) are also excluded.
    """
    gold = set()
    deleted = set()
    lines = patch.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("--- /dev/null"):
            # The +++ line follows immediately; the file is NEW, skip
            if i + 1 < len(lines) and lines[i + 1].startswith("+++ b/"):
                pass  # Will be skipped when we see '+++ /dev/null' on delete
        if line.startswith("+++ /dev/null"):
            # File was deleted -- can't retrieve what doesn't exist
            if i > 0 and lines[i - 1].startswith("--- b/"):
                deleted.add(lines[i - 1][6:])
        elif line.startswith("+++ b/"):
            path = line[6:]
            # Check if the previous --- line was /dev/null (new file)
            if i > 0 and lines[i - 1].startswith("--- /dev/null"):
                pass  # New file -- skip
            else:
                gold.add(path)
    return gold - deleted


# ---------------------------------------------------------------------------
# Index a repo -- chunk + embed all source files
# ---------------------------------------------------------------------------

def index_repo(repo_path: Path, lang_config: dict, model_name: str,
               embedder: SentenceTransformer) -> tuple:
    """
    Scan, chunk, and embed all source files in repo_path.
    Returns (chunk_ids, embeddings) where chunk_ids[i] corresponds to embeddings[i].
    """
    source_files = [
        p for p in repo_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in INDEXABLE_EXTENSIONS
        and not is_skipped(p.relative_to(repo_path))
    ]

    all_chunks: list = []
    for f in source_files:
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
            rel = str(f.relative_to(repo_path))
            all_chunks.extend(chunk_file(rel, source, lang_config))
        except OSError:
            continue

    if not all_chunks:
        return [], np.empty((0,))

    doc_prefix = DOC_PREFIX.get(model_name, "")
    texts = [doc_prefix + c["text"] for c in all_chunks]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
        batch_size=64,
    )
    ids = [c["id"] for c in all_chunks]
    return ids, np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Retrieval and metrics
# ---------------------------------------------------------------------------

def retrieve_top_k(query_emb: np.ndarray, chunk_embs: np.ndarray,
                   chunk_ids: list, k: int) -> list:
    """
    Return top-K (chunk_id, score) pairs by cosine similarity.
    Assumes embeddings are already L2-normalised (so dot product = cosine similarity).
    Replicates claude_light.py retrieve() scoring (line 1012), without MIN_SCORE
    filtering (we want unfiltered curves for the benchmark).
    """
    scores = chunk_embs @ query_emb
    top_idx = np.argsort(-scores)[:k]
    return [(chunk_ids[i], float(scores[i])) for i in top_idx]


def chunk_id_to_file(chunk_id: str) -> str:
    """'path/to/file.py::method' -> 'path/to/file.py'. Whole-file IDs pass through."""
    return chunk_id.split("::")[0]


def compute_instance_metrics(top_k_pairs: list, gold_files: set, k_values: list) -> dict:
    """
    Compute retrieval metrics for a single instance.

    Returns a dict with keys:
      hit_at_k    {k: bool}    -- True if any retrieved chunk is from a gold file
      recall_at_k {k: float}   -- fraction of gold files retrieved
      precision_at_k {k: float}-- fraction of retrieved files that are gold
      mrr         float        -- reciprocal rank of first gold chunk (0.0 if none)
    """
    # Rank of first gold chunk (1-indexed), or infinity if not found in full list
    first_hit_rank = float("inf")
    for rank, (cid, _score) in enumerate(top_k_pairs, start=1):
        if chunk_id_to_file(cid) in gold_files:
            first_hit_rank = rank
            break

    mrr = 1.0 / first_hit_rank if first_hit_rank != float("inf") else 0.0

    hit_at_k:       dict = {}
    recall_at_k:    dict = {}
    precision_at_k: dict = {}

    for k in k_values:
        top_k = top_k_pairs[:k]
        retrieved_files = {chunk_id_to_file(cid) for cid, _ in top_k}
        overlap = retrieved_files & gold_files

        hit_at_k[k]       = len(overlap) > 0
        recall_at_k[k]    = len(overlap) / len(gold_files) if gold_files else 0.0
        precision_at_k[k] = len(overlap) / len(retrieved_files) if retrieved_files else 0.0

    return {
        "hit_at_k":       hit_at_k,
        "recall_at_k":    recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr":            mrr,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _table(headers: list, rows: list, aligns: list) -> str:
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells):
        parts = []
        for i, c in enumerate(cells):
            t = str(c)
            parts.append(f" {t.rjust(widths[i]) if aligns[i]=='r' else t.ljust(widths[i])} ")
        return "|" + "|".join(parts) + "|"

    return "\n".join([sep, fmt_row(headers), sep, *[fmt_row(r) for r in rows], sep])


def _pct(v: float) -> str: return f"{v * 100:.1f}%"
def _f3(v: float) -> str:  return f"{v:.3f}"


def print_aggregate(results: list, k_values: list) -> None:
    n = len(results)
    print(f"\n{'=' * 72}")
    print(f"  AGGREGATE  ({n} instances)")
    print("=" * 72)

    headers = ["Metric"] + [f"K={k}" for k in k_values]
    aligns  = ["l"] + ["r"] * len(k_values)

    def avg(key, k):
        return sum(r["metrics"][key][k] for r in results) / n

    rows = [
        ["Hit@K"]       + [_pct(avg("hit_at_k", k))       for k in k_values],
        ["Recall@K"]    + [_pct(avg("recall_at_k", k))     for k in k_values],
        ["Precision@K"] + [_pct(avg("precision_at_k", k))  for k in k_values],
        ["MRR"]         + [_f3(sum(r["metrics"]["mrr"] for r in results) / n)] + [""] * (len(k_values) - 1),
    ]
    print(_table(headers, rows, aligns))

    # Per-repo breakdown
    repos = sorted({r["repo"] for r in results})
    if len(repos) > 1:
        k_mid = k_values[len(k_values) // 2]
        print(f"\n  Per-repo breakdown (K={k_mid}):")
        headers2 = ["Repo", "N", f"Hit@{k_mid}", f"Recall@{k_mid}", "MRR"]
        aligns2  = ["l", "r", "r", "r", "r"]
        rows2 = []
        for repo in repos:
            rs = [r for r in results if r["repo"] == repo]
            rows2.append([
                repo,
                str(len(rs)),
                _pct(sum(r["metrics"]["hit_at_k"][k_mid] for r in rs) / len(rs)),
                _pct(sum(r["metrics"]["recall_at_k"][k_mid] for r in rs) / len(rs)),
                _f3(sum(r["metrics"]["mrr"] for r in rs) / len(rs)),
            ])
        print(_table(headers2, rows2, aligns2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG retrieval quality benchmark using SWE-bench Lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="SWE-bench Lite split to use (default: dev, 23 instances)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N instances (default: all)")
    parser.add_argument("--repo", default=None,
                        help="Filter to a specific repo, e.g. sympy/sympy")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES,
                        help="K values to evaluate (default: 5 10 15)")
    parser.add_argument("--repos-dir", type=Path, default=Path(".benchmark_repos"),
                        help="Directory for caching cloned repos")
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark_retrieval_cache"),
                        help="Directory for caching embeddings")
    parser.add_argument("--json", action="store_true",
                        help="Emit full results as JSON to stdout")
    args = parser.parse_args()

    k_values = sorted(args.k)

    print(f"\nRAG Retrieval Quality Benchmark")
    print(f"Dataset: SWE-bench Lite ({args.split} split)")
    print(f"K values: {k_values}")
    print(f"Repos cache: {args.repos_dir}")
    print(f"Embed cache: {args.cache_dir}\n")

    # Load dataset
    print("Loading SWE-bench Lite from HuggingFace...", end="", flush=True)
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.split)
    instances = list(dataset)
    print(f" {len(instances)} instances loaded.")

    if args.repo:
        instances = [i for i in instances if i["repo"] == args.repo]
        print(f"Filtered to repo '{args.repo}': {len(instances)} instances.")

    if args.n:
        instances = instances[: args.n]
        print(f"Limited to first {args.n} instances.")

    # Load tree-sitter grammars once
    lang_config = load_language_config()

    results = []
    embedder = None
    current_model = None

    for idx, instance in enumerate(instances, 1):
        iid    = instance["instance_id"]
        repo   = instance["repo"]
        commit = instance["base_commit"]
        stmt   = instance["problem_statement"]
        patch  = instance["patch"]

        gold_files = extract_gold_files(patch)
        if not gold_files:
            print(f"\n[{idx}/{len(instances)}] {iid}: no retrievable gold files, skipping.")
            continue

        print(f"\n[{idx}/{len(instances)}] {iid}")
        print(f"  Repo: {repo}@{commit[:8]}  |  Gold files: {len(gold_files)}")

        # Clone / load repo
        repo_path = get_repo_path(repo, commit, args.repos_dir)
        if repo_path is None:
            print(f"  Skipping (clone failed).")
            continue

        # Count files to select embedding model
        n_files = sum(
            1 for p in repo_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in INDEXABLE_EXTENSIONS
            and not is_skipped(p.relative_to(repo_path))
        )
        model_name = select_embed_model(n_files)

        # Load or create embedder (reuse if same model)
        if model_name != current_model or embedder is None:
            print(f"  Loading embedder: {model_name}...", end="", flush=True)
            embedder = SentenceTransformer(model_name, trust_remote_code=True)
            current_model = model_name
            print(" done")

        # Load embedding cache or index repo
        cached = load_embedding_cache(args.cache_dir, repo, commit, model_name)
        if cached is not None:
            chunk_ids, chunk_embs = cached
            print(f"  Embedding cache hit: {len(chunk_ids)} chunks ({n_files} files)")
        else:
            print(f"  Indexing {n_files} files...", end="", flush=True)
            chunk_ids, chunk_embs = index_repo(repo_path, lang_config, model_name, embedder)
            if len(chunk_ids) == 0:
                print(" no indexable content, skipping.")
                continue
            save_embedding_cache(args.cache_dir, repo, commit, model_name, chunk_ids, chunk_embs)
            print(f" {len(chunk_ids)} chunks embedded and cached.")

        # Embed problem statement
        query_prefix = QUERY_PREFIX.get(model_name, "")
        query_emb = embedder.encode(
            query_prefix + stmt, normalize_embeddings=True, show_progress_bar=False
        )

        # Retrieve top-K (use max k to get all needed ranks at once)
        max_k = max(k_values)
        top_pairs = retrieve_top_k(query_emb, chunk_embs, chunk_ids, max_k)

        # Compute metrics
        metrics = compute_instance_metrics(top_pairs, gold_files, k_values)

        # Show per-instance summary
        top_files = list(dict.fromkeys(chunk_id_to_file(cid) for cid, _ in top_pairs))
        gold_list = sorted(gold_files)
        print(f"  Gold ({len(gold_files)}): {', '.join(gold_list[:3])}"
              + (" ..." if len(gold_list) > 3 else ""))
        hit_strs = " | ".join(
            f"Hit@{k}={'YES' if metrics['hit_at_k'][k] else 'NO'}  "
            f"Recall@{k}={metrics['recall_at_k'][k]*100:.0f}%"
            for k in k_values
        )
        print(f"  {hit_strs}  |  MRR={metrics['mrr']:.3f}")

        results.append({
            "instance_id":  iid,
            "repo":         repo,
            "commit":       commit,
            "n_gold_files": len(gold_files),
            "n_chunks":     len(chunk_ids),
            "n_files":      n_files,
            "embed_model":  model_name,
            "metrics":      metrics,
        })

    if not results:
        print("\nNo instances processed.")
        return

    if args.json:
        # Serialize metrics (convert bool/int keys for JSON)
        out = []
        for r in results:
            m = r["metrics"]
            out.append({**r, "metrics": {
                "hit_at_k":       {str(k): v for k, v in m["hit_at_k"].items()},
                "recall_at_k":    {str(k): round(v, 4) for k, v in m["recall_at_k"].items()},
                "precision_at_k": {str(k): round(v, 4) for k, v in m["precision_at_k"].items()},
                "mrr":            round(m["mrr"], 4),
            }})
        print(json.dumps(out, indent=2))
        return

    print_aggregate(results, k_values)
    print()


if __name__ == "__main__":
    main()
