# LLMLingua-2 Prompt Compression

> Part of the [architecture index](ARCHITECTURE.md). Source: the original `docs/architecture.md`, split by topic.

### LLMLingua-2 Prompt Compression *(optional)*

**What it is.** [LLMLingua-2](https://github.com/microsoft/LLMLingua) (Microsoft, MIT-licensed) is a prompt-compression library. It runs a small BERT-class encoder over a prompt and drops tokens predicted to be least important, preserving semantic content at a configurable rate (e.g. keep 50%). We use the `llmlingua-2-bert-base-multilingual-cased-meetingbank` model by default (~280 MB, CPU-capable, ~100–400 ms per 5 000-token input).

**Why we use it.** `claude_light` already has two token-saving strategies — **prompt caching** (skeleton + retrieved-context blocks cached at $0.30/M vs $3.00/M input) and **history summarisation** (old turns collapsed via Haiku). LLMLingua-2 closes a gap that neither of those helps with: the **retrieved-context block changes on every query** (different top-K chunks → different cache key), so it almost never hits the cross-query cache. Compressing that block 2× before sending halves what the API has to read and bill on the cache-write / cache-miss path.

**Where it plugs in.** Exactly one spot in the pipeline, after retrieval and before the message is assembled:

```
retrieve()  →  compress_context()  →  cache_control: ephemeral block  →  API
```

Skeleton is **not** compressed (already cached across many queries, so per-query marginal cost is ~$0.30/M and compressing would destroy the literal directory listings and markdown structure the model relies on). User query and conversation history are not compressed either.

**How it's wired.**

| File | Role |
|---|---|
| `claude_light/tools.py` | Read-only tools for API-key turns (`read_file`, `grep`, `list_files`); `llm._api_tool_loop()` streams each request, runs any `tool_use` blocks, sends all results back in one user message and repeats (max 8 rounds), never on a `max_tokens` or `refusal` stop |
| `claude_light/compressor.py` | Thread-safe lazy-loaded `PromptCompressor` singleton + `compress_context(text, rate, force_tokens)` that never raises into the hot path |
| `claude_light/llm.py` | Calls `compress_context()` in `chat()` and `one_shot()` right after `retrieve()`; routes stats via `_accumulate_compression_stats()` |
| `claude_light/state.py` | Three new `global_stats` fields: `total_tokens_pre_compress`, `total_tokens_post_compress`, `total_dollars_saved_llmlingua` — additive with existing cache savings, tracked separately so the display is never ambiguous |
| `claude_light/ui.py` | Per-turn `print_stats` gains a `LLMLingua N→M (pct%)` segment; session summary gains an `LLMLingua-2 Compression` panel below Global Lifetime Savings |
| `claude_light/config.py` | `LLMLINGUA_ENABLED`, `LLMLINGUA_MODEL`, `LLMLINGUA_TARGET_RATE`, `LLMLINGUA_MIN_TOKENS`, `LLMLINGUA_FORCE_TOKENS` |

**Safety properties.**

* Optional dependency — if `pip install llmlingua` hasn't been run, `compress_context()` returns the input unchanged with `skipped=True`.
* Any `ImportError` or `RuntimeError` inside the compressor is swallowed; original text is returned.
* Skipped below `LLMLINGUA_MIN_TOKENS` (800 estimated tokens, override with `CLAUDE_LIGHT_LLMLINGUA_MIN_TOKENS`) — below that, 100–400 ms of compression overhead is larger than the token saving.
* Sanity check: if LLMLingua ever returns a **longer** output, we ignore its result and send the original.
* `force_tokens=["\n", "```", "::", "//"]` preserves newlines, fenced-code markers, `::` chunk-ID separators, and comment delimiters so RAG chunk structure survives compression.
* **Non-blocking model load.** The ~280 MB BERT model is loaded in a background daemon thread launched from `main()` so indexing and skeleton-building run in parallel with the download. If a query arrives before the loader finishes, `compress_context()` skips that turn (returns the uncompressed text with `reason="still_loading"`) rather than blocking the query. This prevents the "Processing…" hang on first run — compression simply kicks in from the turn after the load completes.
* **Enabled by default** (`LLMLINGUA_ENABLED=True`) once `pip install llmlingua` has been run. On machines without the package, `compress_context()` is a silent no-op, so the default causes no behavioural change on fresh installs. Opt out with `CLAUDE_LIGHT_LLMLINGUA=0` (env) or by editing `config.py`.

**Savings math** (at rate=0.5, derived from the same pricing as the rest of the pipeline):

| Path | Billed price | Effect of 2× compression |
|---|---|---|
| Cache miss / first query (cache write) | $2.50/M on Sonnet 5.5 | Halves the write cost on the retrieved block |
| Cache hit (cache read) | $0.10/M on Sonnet 5.5 | Halves the (already-cheap) read cost |
| Skeleton | unchanged | Not touched |

On a session with frequent distinct queries (cache miss on retrieved block), LLMLingua adds **~50% extra savings on top of caching** for the retrieved-context portion. On a session dominated by repeat queries over the same chunks, the extra saving is small in absolute dollars because caching is already doing most of the work — the projection table in `tests/benchmarks/benchmark.py` (`LLMLINGUA-2 PROJECTION` section) shows this clearly.

See [docs/llmlingua_plan.md](../llmlingua_plan.md) for the full design rationale and trade-off notes.

---

