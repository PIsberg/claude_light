# Configuration reference

Constants live in `claude_light/config.py`. Environment variables override the defaults where shown. How each part is used is in [architecture/](architecture/ARCHITECTURE.md).

| Variable | Purpose |
|---|---|
| `TARGET_RETRIEVED_TOKENS` | Token budget for retrieved context per query (default 6 000) |
| `MIN_SCORE` | Absolute cosine similarity floor to include a chunk (default 0.35; 0.45 dropped most natural-language questions) |
| `RELATIVE_SCORE_FLOOR` | Drop chunks below this fraction of the top chunk's score (default 0.60) |
| `HYBRID_RETRIEVAL` | Static + dense index (default on); `CLAUDE_LIGHT_HYBRID=0` restores one auto-tuned model and the two floors above |
| `FAST_EMBED_MODEL` / `DENSE_EMBED_MODEL` | `static-retrieval-mrl-en-v1` / `all-mpnet-base-v2` |
| `HYBRID_MIN_SCORE` | Per-model cosine floor for "anything related at all" (static 0.22, mpnet 0.21: just above the best chunk for off-topic questions) |
| `HYBRID_SCORE_GAP` / `HYBRID_FAST_WEIGHT` | Keep chunks within this fused z-score of the top (1.5) / static model's share of the fused score (0.5) |
| `_RETRIEVAL_BUDGET` | Per-effort token budgets: low=1500, medium=3000, high=6000, max=9000 |
| `MAX_HISTORY_TURNS` | Compress and cap when stored turns exceed this (default 6) |
| `SUMMARIZE_BATCH` | Old turns collapsed per summary call (default 3) |
| `GLOBAL_STATS_FILE` | Path to global savings persistence (`~/.claude_light_stats.json`; override with `CLAUDE_LIGHT_STATS_FILE`) |
| `HEARTBEAT_SECS` | How often the heartbeat thread wakes (default 30) |
| `CACHE_TTL_SECS` | Idle seconds before the heartbeat warms the cache (default 240) |
| `MODEL_HAIKU` / `MODEL_SONNET` / `MODEL_OPUS` | Model ID constants used by the router |
| `SUMMARY_MODEL` | Model used for history compression (set to `MODEL_HAIKU`) |
| `MODEL_PRICES` / `model_prices()` | Per-model USD per M tokens (input, 5 m and 1 h cache write, read, output); every call is priced at the model that ran it. `PRICE_INPUT/WRITE/READ/OUTPUT` are the default model's rates (Sonnet 5.5: $2.00 / $2.50 / $0.10 / $10.00) |
| `SKIP_DIRS` | Directory names excluded from indexing |
| `EMBED_MODEL`, `TOP_K` | Set at runtime by `auto_tune()`; do not set manually |
| `LLMLINGUA_ENABLED` | **On** by default (silent no-op if `llmlingua` isn't installed). Opt out via `CLAUDE_LIGHT_LLMLINGUA=0` |
| `LLMLINGUA_MODEL` | Default `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (~280 MB). Override with `CLAUDE_LIGHT_LLMLINGUA_MODEL` |
| `LLMLINGUA_TARGET_RATE` | Fraction of tokens to **keep** (default 0.5 = halve). Override with `CLAUDE_LIGHT_LLMLINGUA_RATE` |
| `LLMLINGUA_MIN_TOKENS` | Skip compression below this size, since overhead exceeds savings (default 800). Override with `CLAUDE_LIGHT_LLMLINGUA_MIN_TOKENS`; the retrieval budgets (1 500–9 000) are all above it |
| `LLMLINGUA_FORCE_TOKENS` | Tokens preserved verbatim (default `["\n", "```", "::", "//"]` so chunk structure survives) |
| `AGENT_SYSTEM_PROMPT` | Replaces `SYSTEM_PROMPT` in OAuth mode when `ALLOWED_COMMANDS` is non-empty. See [architecture/components.md](architecture/components.md#agent-mode-and-allowed-commands) |
| `ALLOWED_COMMANDS` | Shell command prefixes the OAuth agent may run. From `--allow-command` or `CLAUDE_LIGHT_ALLOW_COMMANDS`. See [architecture/components.md](architecture/components.md#agent-mode-and-allowed-commands) |
