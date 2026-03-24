# claude_light — Benchmark Suite

Three standalone benchmarks ship with `claude_light.py`. Each tests a different
aspect of what the tool optimises and can be shared independently to demonstrate
its value.

## Published Headline Numbers

The current headline benchmark figures published in [`README.md`](../README.md)
come from `benchmark_claude_code.py`, which compares Claude Code against
`claude_light` on the same 4 Python repositories and 10-query workload.

Measured on 2026-03-23:

| Repository | Repo size | Claude Code | claude_light | Savings |
|---|---:|---:|---:|---:|
| `psf/requests` v2.31.0 | 108K tokens | $0.91 | $0.23 | **75%** |
| `pallets/flask` 3.0.0 | 144K tokens | $1.62 | $0.17 | **89%** |
| `encode/httpx` 0.25.2 | 191K tokens | $1.75 | $0.45 | **74%** |
| `bottlepy/bottle` 0.13.2 | 92K tokens | $1.52 | $0.30 | **80%** |
| **Total (≈40 queries)** | | **$5.81** | **$1.16** | **80% / 5× cheaper** |

`benchmark_cost.py` is still useful, but it answers a different question:
Claude Light vs a naive "send the whole repo" baseline. The README headline
stats should therefore be treated as the source of truth for tool-vs-tool cost
comparison, and this document now reflects that.

---

## Quick Reference

| Script | What it tests | API key needed | Est. cost |
|---|---|---|---|
| `benchmark.py` | Analytical token savings formula | No | Free |
| `benchmark_retrieval.py` | RAG retrieval quality (Hit@K, MRR) | No | Free* |
| `benchmark_cost.py` | Real-world cost savings vs naive baseline | Yes | ~$0.05–$0.50 |
| `benchmark_claude_code.py` | Claude Code vs `claude_light` vs naive baseline | Yes | Varies with live API/tool runs |

\* Downloads HuggingFace dataset + sentence-transformers on first run (~1.5 GB).

---

## 1 · `benchmark.py` — Analytical Token-Savings Benchmark

### Purpose

Answers: *"How much cheaper is claude_light than sending the whole codebase on
every query?"* using pure arithmetic — no API calls, no network, no GPU.

It replicates every formula from `claude_light.py` (token proxy, TOP_K,
retrieval k, cost calculation) on synthetic Java repositories generated
identically to `--test-mode`.

### Methodology

Three cost scenarios are compared for every (preset × query × effort) combination:

| Scenario | What it represents | Token price used |
|---|---|---|
| **Naive** | Sending the full repo on every single query | $3.00/M (full input) |
| **Cold** | First query of a session — cache write + retrieval | $3.75/M write |
| **Warm** | Follow-up queries — cache read + retrieval | $0.30/M read |

Formulas replicated from `claude_light.py`:

```
token_count  = len(text) // 4
TOP_K        = max(2, min(15, round(6000 / avg_chunk_tokens)))
retrieved_k  = max(2, round(TOP_K * budget / TARGET_RETRIEVED_TOKENS))
retrieved_k  = min(retrieved_k, total_chunks)
```

Pricing constants:  `PRICE_INPUT=3.00`, `PRICE_WRITE=3.75`, `PRICE_READ=0.30`,
`PRICE_OUTPUT=15.00` (per million tokens).

A 10-turn session model `[low×3, medium×3, high×3, max×1]` is used to compute
realistic session costs (turn 1 = cold, turns 2–10 = warm).

### Presets

| Preset | Files | Methods/file | Approx. naive tokens |
|---|---|---|---|
| small | 5 | 10 | ~2.3K |
| medium | 50 | 15 | ~36K |
| large | 200 | 20 | ~193K |
| extra-large | 1000 | 20 | ~966K |

### Running

```bash
# All presets (default)
python benchmark.py

# One preset
python benchmark.py --preset large

# Machine-readable JSON
python benchmark.py --json

# Fast mode (uses precomputed token counts — skips the iteration loop)
python benchmark.py --fast
```

### Interpreting Results

The four output tables are:

1. **Preset overview** — files, chunks, avg chunk tokens, TOP_K, skeleton tokens,
   naive tokens for that preset.
2. **Per-query table** — one row per query × effort combination showing naive / cold /
   warm tokens and costs with percentage savings.
3. **Per-effort aggregate** — mean Cold save% and Warm save% per effort tier,
   averaged across all presets.
4. **10-turn session projection** — what a realistic coding session costs in each
   scenario and the Naive/Warm ratio.

Key insight from the extra-large preset:

```
Naive 10-turn session:  $27.61
Warm  10-turn session:  $0.08
Ratio:                  346x
```

### Assumptions / Limitations

- Synthetic data only — all files are identical Java templates. Real codebases
  have variable chunk sizes and more complex skeletons.
- Skeleton compression for synthetic data is exact (all files brace-group to one
  line). Real skeletons vary by directory depth.
- Does not model cache expiry (heartbeat keeps cache alive; no query gap > 4 min
  in a real session of ten turns).

---

## 2 · `benchmark_retrieval.py` — RAG Retrieval Quality Benchmark

### Purpose

Answers: *"Does claude_light's RAG pipeline retrieve the right code?"* using real
bug-fix commits from the [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)
dataset as ground truth.

Given a natural-language issue description, the benchmark checks whether the
correct source files appear in the top-K retrieved chunks — the same chunks
claude_light would send to Claude.

### Methodology

1. Download SWE-bench Lite from HuggingFace (`princeton-nlp/SWE-bench_Lite`).
2. For each instance, clone the target repo at the base commit using a treeless
   shallow clone (`git clone --filter=blob:none --no-checkout`).
3. Parse the `patch` field to extract gold files (the files modified by the fix).
4. Chunk all source files with the same tree-sitter pipeline used by
   `claude_light.py` (method-level, with preamble injection).
5. Embed with the same auto-selected sentence-transformers model.
6. Query with the issue description text and rank chunks by cosine similarity.
7. Measure Hit@K, Recall@K, Precision@K, and MRR at K = [5, 10, 15].

### Metrics

| Metric | Definition |
|---|---|
| **Hit@K** | Fraction of instances where at least one gold file appears in top-K chunks |
| **Recall@K** | Fraction of all gold files retrieved in top-K (averaged per instance) |
| **Precision@K** | Fraction of top-K chunks that are gold files (averaged per instance) |
| **MRR** | Mean Reciprocal Rank — 1/rank of the first retrieved gold chunk |

Higher is always better. Hit@K and MRR are the primary signals.

### Repos Covered (SWE-bench Lite subset)

The dataset spans 6 Python repositories: `django/django`, `astropy/astropy`,
`sympy/sympy`, `scikit-learn/scikit-learn`, `matplotlib/matplotlib`,
`pytest-dev/pytest`. Some instances may fail to clone or checkout if the exact
commit no longer exists on the remote.

### Running

```bash
# Install dependencies
pip install datasets sentence-transformers numpy

# Full dev split (23 instances — fast, no API key needed)
python benchmark_retrieval.py --split dev

# Subset of the test split (300 instances total)
python benchmark_retrieval.py --split test --n 50

# One specific repo only
python benchmark_retrieval.py --split test --repo django/django

# Custom K values
python benchmark_retrieval.py --k 5 10 20

# Machine-readable JSON
python benchmark_retrieval.py --json

# Use pre-cloned repos (avoids re-cloning)
python benchmark_retrieval.py --repos-dir /path/to/repos
```

### Disk Space

| Item | Size |
|---|---|
| HuggingFace dataset | ~50 MB |
| Embedding model (MiniLM, < 50 files) | 22 MB |
| Embedding model (mpnet, 50-199 files) | 420 MB |
| Embedding model (nomic, 200+ files) | ~550 MB |
| Repo clones (treeless, all 6 repos) | ~300 MB |
| Embedding cache | ~50–200 MB |

### Interpreting Results

The output table shows per-repo and overall metrics at each K:

```
Results: dev split  23 instances  K=[5,10,15]
+---------------------------+----------+----------+----------+----------+
| Repo                      | Hit@5    | Hit@10   | MRR      | Recall@10|
+---------------------------+----------+----------+----------+----------+
| django/django             | 0.714    | 0.857    | 0.542    | 0.612    |
| ...                       | ...      | ...      | ...      | ...      |
| OVERALL                   | 0.652    | 0.783    | 0.498    | 0.561    |
+---------------------------+----------+----------+----------+----------+
```

- **Hit@10 > 0.70** is a strong result for method-level chunking.
- **MRR > 0.50** means the correct file typically appears in the first two results.
- Low Recall@K with high Hit@K means multi-file bugs where only one file is
  retrieved — a known limitation of top-K retrieval.

### Assumptions / Limitations

- Gold files come from the patch diff — new files added by the fix are excluded
  (they don't exist at the base commit). Deleted files are also excluded.
- The issue text is used directly as the query; `claude_light.py` would let
  Claude rephrase or disambiguate first, which could improve recall.
- `MIN_SCORE` (0.45) cosine threshold is **not applied** in this benchmark so
  that recall at all K values is measurable. In production, low-scoring chunks
  are dropped, which improves precision at the cost of recall.
- Some SWE-bench instances touch only configuration files or test files that are
  outside the indexed extensions — these count as misses.

---

## 3 · `benchmark_cost.py` — Real-World Cost Benchmark

### Purpose

Answers: *"How much money does claude_light actually save on real code?"* by
running the tool against four popular Python open-source projects and comparing
the actual API spend to a naive baseline (sending the whole repo every query).

This is the benchmark most useful for sharing with other developers: it shows
real dollar amounts on real codebases.

### Methodology

1. Clone 4 Python repos at fixed stable tags (shallow, `--depth 1`).
2. Count total source tokens in each repo (same extensions and SKIP_DIRS as
   `claude_light.py`) to compute the naive baseline.
3. For each of 10 standard queries, run `claude_light.py` in one-shot mode:
   ```bash
   python claude_light.py "query"
   ```
   with `stdin=/dev/null` so it runs non-interactively.
4. Strip ANSI codes from stderr and extract the `[Stats]` and `[Cost]` lines.
5. Compute true savings vs the naive baseline:
   ```
   nocache_cost = actual_cost + saved_vs_nocache   (from [Cost] line)
   output_cost  = nocache_cost - (total_input / 1M) * 3.00
   naive_cost   = (naive_tokens / 1M) * 3.00 + output_cost
   true_savings = (naive_cost - actual_cost) / naive_cost
   ```

### Repos Benchmarked

| Repo | Tag | Description |
|---|---|---|
| `psf/requests` | v2.31.0 | HTTP library |
| `pallets/flask` | 3.0.0 | WSGI web framework |
| `encode/httpx` | 0.25.2 | Next-gen HTTP client |
| `bottlepy/bottle` | 0.13.2 | Fast micro web-framework |

These repos were chosen because they are:
- Pure Python (no generated code or large binary assets)
- Well-maintained with stable tags
- Representative of real-world project sizes (small to medium)

### Standard Queries

10 generic queries spanning all 4 effort tiers are run against each repo:

| # | Effort | Query |
|---|---|---|
| 1 | low | List all public classes or modules in this codebase |
| 2 | low | Where is the main entry point defined? |
| 3 | low | What Python version is required by this project? |
| 4 | medium | Explain how HTTP sessions or connections are managed |
| 5 | medium | Summarize the error handling and exception hierarchy |
| 6 | medium | Describe how routing or request dispatching works |
| 7 | high | Show me an example of adding a custom middleware or hook |
| 8 | high | Implement a helper function that retries a request on 5xx errors with exponential backoff |
| 9 | high | Identify potential thread-safety issues and suggest fixes |
| 10 | max | Evaluate the overall architecture trade-offs and scalability of this codebase |

### Running

```bash
# Install API key (real costs incurred)
export ANTHROPIC_API_KEY=sk-ant-...

# Full run (all 4 repos × 10 queries = 40 API calls)
python benchmark_cost.py

# One repo only
python benchmark_cost.py --repo pallets/flask

# Dry-run (clone repos, count tokens, no API calls)
python benchmark_cost.py --dry-run

# Use already-cloned repos
python benchmark_cost.py --repos-dir /path/to/repos

# JSON output
python benchmark_cost.py --json
```

### Estimated Costs

A full run (40 API calls) costs approximately:
- **$0.05–$0.15** if cache hits land well (queries 2–10 reuse the warm cache).
- **$0.20–$0.50** worst case (cold cache on every query, max-effort responses).

The first query per repo is always a cache-write. Subsequent queries benefit from
warm cache reads at $0.30/M vs $3.75/M write.

### Interpreting Results

**Per-repo table:**

```
Results for psf/requests:
+--+-----------------------------------------------+------+--------+---------+------+---------+---------+----------+
|# | Query (truncated)                             |Effort|Model   |Input tok|Cache%|Actual $ |Naive $  |True save%|
+--+-----------------------------------------------+------+--------+---------+------+---------+---------+----------+
|1 |List all public classes or modules...          |low   |haiku   |   12,450|  0.0%|  $0.0005|  $0.0421| 98.8%    |
|2 |Where is the main entry point defined?         |low   |haiku   |   11,200| 84.3%|  $0.0001|  $0.0393| 99.7%    |
...
```

**Key columns:**
- **Cache%**: Fraction of input tokens served from cache on that query.
- **Actual $**: What was actually charged to your API key.
- **Naive $**: What it would cost to send the entire repo on this query.
- **True save%**: `(Naive - Actual) / Naive`. The headline number.

**Aggregate table:**

```
AGGREGATE RESULTS ACROSS ALL REPOS
+------------------------+------------------------------+-------+---------+---------+----------+
|Repo                    |Desc                          |Queries|Actual $ |Naive $  |True save%|
+------------------------+------------------------------+-------+---------+---------+----------+
|psf/requests            |HTTP library for Python       |10     |$0.0025  |$0.4210  |99.4%     |
|pallets/flask           |Lightweight WSGI web framework|10     |$0.0031  |$0.5820  |99.5%     |
...
|TOTAL                   |                              |40     |$0.0123  |$2.1380  |99.4%     |
+------------------------+------------------------------+-------+---------+---------+----------+
Naive/Actual ratio: 174x
```

### Assumptions / Limitations

- **Cache warm-up**: Query 1 per repo is always cold. In real usage, developers
  run many more than 10 queries per session, so the effective cache hit rate is
  higher.
- **Naive baseline**: Assumes the entire repo is re-sent on every query at full
  $3.00/M input price. This is the absolute worst case. Other tools may use
  smaller context windows but still be more expensive than claude_light's RAG.
- **Model routing**: claude_light's router picks model and effort automatically.
  The benchmark does not force a specific model — it measures the tool as-is.
- **tree-sitter**: If tree-sitter is installed, chunks are method-level. Without
  it, whole-file chunks are used. Whole-file mode increases retrieved tokens and
  slightly reduces true savings. Install tree-sitter for best results.
- **Output tokens**: Output tokens are included in `actual_cost` but not in
  `naive_cost` estimation for the input-only naive baseline. The formula adds
  them back correctly via `output_cost`.

---

## 4 · `benchmark_claude_code.py` — Claude Code vs Claude Light

### Purpose

Answers: *"How does `claude_light` compare to Claude Code on the same real
queries?"* by combining live Claude Code runs with the saved JSON output from
`benchmark_cost.py`.

This is the benchmark that feeds the README headline comparison table.

### What It Compares

For each repo/query pair it compares:

- **Claude Code**: `claude --print` in a fresh isolated session
- **claude_light**: JSON results captured from `benchmark_cost.py --json`
- **Naive baseline**: sending the entire repo every time

### Repo Set

The current published comparison uses:

| Repo | Tag | Description |
|---|---|---|
| `psf/requests` | v2.31.0 | HTTP library |
| `pallets/flask` | 3.0.0 | WSGI web framework |
| `encode/httpx` | 0.25.2 | Next-gen HTTP client |
| `bottlepy/bottle` | 0.13.2 | Fast micro web-framework |

### Running

```bash
# First capture claude_light's real-cost benchmark output
python benchmark_cost.py --json > cl_results.json

# Then run the Claude Code comparison
python benchmark_claude_code.py --claude-light-results cl_results.json

# One repo only
python benchmark_claude_code.py --repo pallets/flask --claude-light-results cl_results.json
```

### Published Result Snapshot

The current README numbers are:

| Repository | Claude Code | claude_light | Savings |
|---|---:|---:|---:|
| `psf/requests` | $0.91 | $0.23 | **75%** |
| `pallets/flask` | $1.62 | $0.17 | **89%** |
| `encode/httpx` | $1.75 | $0.45 | **74%** |
| `bottlepy/bottle` | $1.52 | $0.30 | **80%** |
| **Total** | **$5.81** | **$1.16** | **80% / 5× cheaper** |

## Running All Four Benchmarks

```bash
# 1. Analytical (free, instant)
python benchmark.py

# 2. Retrieval quality (free, needs GPU/CPU for embeddings)
pip install datasets sentence-transformers numpy
python benchmark_retrieval.py --split dev

# 3. Real-world cost (requires API key, incurs ~$0.10 in costs)
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark_cost.py --dry-run  # preview without API calls
python benchmark_cost.py            # real run

# 4. Claude Code comparison (requires Claude Code plus benchmark_cost JSON)
python benchmark_cost.py --json > cl_results.json
python benchmark_claude_code.py --claude-light-results cl_results.json
```

## Sharing Results

To share benchmark results with other claude_light users:

```bash
# Capture all three
python benchmark.py --json       > results_analytical.json
python benchmark_retrieval.py \
    --split dev --json           > results_retrieval.json
python benchmark_cost.py \
    --json 2>/dev/null           > results_cost.json
```

The JSON outputs are self-contained and include all metadata needed to reproduce
or compare results across runs.

## CI Regression Testing

`claude_light` includes an automated GitHub Actions pipeline that ensures token costs do not spike and retrieval accuracy does not degrade over time!
It implements a hard block on any pull request that reduces `Hit@10` / `Recall@10` by >2% or increases analytical token costs by >1%.

If a developer legitimately changes the chunking behavior for the better, they simply run the benchmarks locally, overwrite the `baseline_*.json` files, and commit them.

**To manually update the baselines:**
```bash
# 1. Update analytical token baseline
python tests/benchmark.py --json > tests/baseline_token_stats.json

# 2. Update retrieval performance baseline (matches the CI marshmallow subset)
python tests/benchmark_retrieval.py --repo marshmallow-code/marshmallow --output tests/baseline_retrieval_stats.json

# 3. Commit the new baseline files
git add tests/baseline_token_stats.json tests/baseline_retrieval_stats.json
git commit -m "chore: Update benchmark baselines"
```
