# LLMLingua Integration Plan

Branch: `llmlingua-integration`

## Summary

Add [LLMLingua-2](https://github.com/microsoft/LLMLingua) (Microsoft, MIT) as an
optional pre-send compressor for the **retrieved RAG context** that is embedded
in each query. Track tokens-before-compression vs. tokens-after-compression so
the existing session summary and global lifetime savings reflect the extra
compression on top of prompt caching.

## Why LLMLingua (and where it actually pays)

`claude_light` already has two token-saving strategies:

1. **Prompt caching** — skeleton (system) and retrieved context (user) are
   written to the ephemeral cache (`llm.py:218,544-550`). Repeat queries over
   the same chunks read at $0.30/M instead of $3.00/M.
2. **History compression** — old turns summarised via Haiku
   (`_summarize_turns` in `llm.py`).

LLMLingua is useful **exactly where caching is not**: the retrieved-context
block changes on every query (different top-K chunks → different cache key),
so it almost never hits the cache across queries. That block is sized
1 500–9 000 tokens per query (`_RETRIEVAL_BUDGET` in `config.py`). LLMLingua-2
is reported to achieve 2×–5× compression at minor quality cost on QA/code
tasks, so this block is the highest-leverage target.

**Do not** compress the skeleton or the user query:
- Skeleton is cached; per-query marginal cost is already ~$0.30/M. Compressing
  only saves on the initial write and risks destroying the literal directory
  listings and markdown structure the model relies on.
- User query is short and compression could distort intent.
- Conversation history is already summarised.

## Dependency / footprint

- `pip install llmlingua` (MIT).
- Pulls in `transformers`; the project already depends on `sentence-transformers`
  which installs `transformers` and `torch`. Net new download: the LLMLingua-2
  model, `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (~560 MB) or the
  smaller `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
  (~280 MB).
- CPU-capable (BERT-class encoder). Expect **~100–400 ms per compress** on
  modern CPU for 5 000-token inputs; faster on GPU. Sentence-transformers
  already runs on CPU here, so no new hardware assumption.
- Make it **optional**: fall back silently if `llmlingua` is not installed
  (same pattern used for `tree-sitter`, `rich`, `prompt_toolkit`).

## Design

### New module: `claude_light/compressor.py`

```python
_COMPRESSOR = None
_COMPRESSOR_LOAD_ERROR = None

def get_compressor():
    """Lazy-load LLMLingua-2 on first use. Thread-safe singleton."""

def compress_context(text: str, rate: float = 0.5, force_tokens=None) -> tuple[str, dict]:
    """
    Returns (compressed_text, info) where info includes:
      - tokens_before (LLMLingua's count)
      - tokens_after
      - ratio
      - elapsed_ms
    On any failure: returns (text, {...noop}) — never raises into the hot path.
    """
```

Uses the smaller BERT model by default. Force-keep newline, code fences
(```` ``` ````), and `::` to preserve chunk structure. Call
`structured_compress_prompt` so per-chunk preambles aren't merged across files.

### Config (`claude_light/config.py`)

```python
LLMLINGUA_ENABLED      = True     # On by default; no-op if llmlingua not installed
LLMLINGUA_MODEL        = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
LLMLINGUA_TARGET_RATE  = 0.5      # Keep 50% of tokens
LLMLINGUA_MIN_TOKENS   = 800      # Below this, skip compression — overhead > savings
LLMLINGUA_FORCE_TOKENS = ["\n", "```", "::", "//"]
```

Env overrides: `CLAUDE_LIGHT_LLMLINGUA=0` to opt out, `CLAUDE_LIGHT_LLMLINGUA_RATE=0.4` to tune the keep rate.

### Wire-in point (single site)

`llm.py:544-550`, right after `retrieve()` returns and before the block is
built:

```python
if retrieved_ctx and config.LLMLINGUA_ENABLED:
    compressed, info = compress_context(
        retrieved_ctx, rate=config.LLMLINGUA_TARGET_RATE,
    )
    retrieved_ctx = compressed
    _accumulate_compression_stats(info)   # see below
```

No change to caching — the compressed block still gets `cache_control: ephemeral`.

### Stats — extend existing machinery

Current state (`state.py`):

```python
"total_tokens_full":   int   # non-cached billed tokens
"total_tokens_saved":  int   # cache-read tokens
"total_dollars_saved": float
```

Add three siblings:

```python
"total_tokens_pre_compress":   int   # what we would have sent
"total_tokens_post_compress":  int   # what we actually sent
"total_dollars_saved_llmlingua": float
```

Update in `_accumulate_compression_stats` (mirror of `_accumulate_usage`):

```python
delta_tokens = info["tokens_before"] - info["tokens_after"]
# Compressed tokens still flow through the cache-write path on first send.
# Conservatively price the saving at PRICE_WRITE (cache write) — this is the
# path the retrieved block actually takes. Using PRICE_INPUT would overstate
# savings in the (rare) case a query re-hits the exact cache.
dollars = (delta_tokens / 1_000_000.0) * PRICE_WRITE
state.global_stats["total_tokens_pre_compress"]    += info["tokens_before"]
state.global_stats["total_tokens_post_compress"]   += info["tokens_after"]
state.global_stats["total_dollars_saved_llmlingua"] += dollars
```

### UI (`ui.py`)

Per-turn `print_stats` gets one extra segment when compression ran:

```
... · compressed 4,812→1,930 (60%) · saved $0.011
```

`print_session_summary` gets a third block below "Global Lifetime Savings":

```
┌──────────── LLMLingua Compression ────────────┐
│ Pre-compress tokens:  <N>                     │
│ Post-compress tokens: <M> (<pct>%)            │
│ Extra dollars saved:  $<X.XX>                 │
└───────────────────────────────────────────────┘
```

Both blocks sum to the **true** savings story:
`cache savings + llmlingua savings`.

### Backwards compatibility of stats file

`state.save_global_stats()` / `load_global_stats()` already ignore unknown
keys. Reading an old `~/.claude_light_stats.json` missing the three new keys
must default them to 0 in `state.py` — already the pattern there.

## Token-savings math (for the PR description)

Per query, with default rate=0.5 and a `medium` effort budget (3 000 tokens):

| | Tokens sent | Cost per query¹ |
|---|---:|---:|
| Baseline (no cache, no LLMLingua) | 3 000 | $0.00900 |
| Cache only (read hit)             | 3 000 | $0.00090 |
| Cache miss + LLMLingua (0.5)      | 1 500 | $0.00563 (cache write) |
| Cache hit  + LLMLingua (0.5)      | 1 500 | $0.00045 |

¹ PRICE_INPUT=$3.00/M, PRICE_WRITE=$3.75/M, PRICE_READ=$0.30/M.

So LLMLingua's main win is on the **cache-miss path** (new query → new
chunks): halves the $3.75/M write. On the cache-hit path it still halves
cost, but at an already-cheap $0.30/M.

For a 100-query session with a 40% cache-miss rate on the retrieved block
(distinct queries), default 3 000-token budget:

- Cache savings today:            60 × 3 000 × ($3.00 − $0.30) / 1M = **$0.486**
- Added LLMLingua savings (0.5): 40 × 1 500 × $3.75 / 1M + 60 × 1 500 × $0.30 / 1M = **$0.225 + $0.027 = $0.252**
- Combined: **~$0.74 / 100 queries** vs. $0.486 today — **+52% savings** on the retrieved block.

Numbers scale linearly with `TARGET_RETRIEVED_TOKENS` / effort.

## Benchmarks

Extend `tests/benchmarks/benchmark_retrieval.py` with a `--llmlingua` flag
that compresses the retrieved context before computing Hit@K / MRR — proves
compression doesn't tank retrieval quality (LLMLingua operates post-retrieval,
so Hit@K is unaffected, but end-task accuracy may drop).

Extend `tests/benchmarks/benchmark_cost.py` with a baseline/cache-only/cache+LLMLingua
three-way compare on the same query trace.

Extend `tests/benchmarks/benchmark.py` analytical model with the math above
so CI prints the projected savings envelope.

## Risks / open questions

1. **Quality regression on code** — LLMLingua-2 was trained on MeetingBank
   (meeting transcripts). Code token importance ≠ English token importance.
   Mitigation: `force_tokens=["\n","```","::","//"]`, small default rate
   (0.5 not 0.3), and benchmark on `benchmark_retrieval.py`'s code corpus
   before flipping `LLMLINGUA_ENABLED=True` by default.
2. **Cold-start latency** — first `compress_context()` call loads the model
   (~2–5 s CPU). Mitigation: warm on first heartbeat (reuse the existing
   heartbeat thread in `main.py`) or on first `retrieve()` result.
3. **Per-query latency** — 100–400 ms on CPU is perceptible. Skip compression
   below `LLMLINGUA_MIN_TOKENS` (savings ≤ overhead). Expose `/compress on|off`
   interactive command for debugging.
4. **Double-counting** — the existing `total_dollars_saved` metric measures
   cache savings; the new `total_dollars_saved_llmlingua` measures
   compression savings. They are additive and must be displayed separately,
   not summed into one "saved" number, or the Stats panel becomes
   uninterpretable.
5. **Windows / Bun CLI path** — OAUTH mode sends the full system prompt via
   `claude` subprocess (`llm.py:223` onwards), bypassing the API-key path.
   Compression must run before the prompt is serialised for either path.
   Current wire-in site (`chat()`, just after `retrieve()`) satisfies this.

## Implementation order

1. `compressor.py` + unit test with a synthetic fixture (mock `PromptCompressor`).
2. Config flags + state fields + stats persistence.
3. Wire into `chat()` in `llm.py`.
4. UI updates in `ui.py` (per-turn + session summary).
5. Update `benchmark.py` analytical model; add `--llmlingua` to
   `benchmark_retrieval.py` and `benchmark_cost.py`.
6. Update `CLAUDE.md` and `docs/architecture.md`.
7. Default `LLMLINGUA_ENABLED=True` — safe because `compress_context()` is a
   silent no-op when the `llmlingua` package isn't installed. Users who have
   it installed get the savings automatically; anyone who wants to opt out
   sets `CLAUDE_LIGHT_LLMLINGUA=0`.
