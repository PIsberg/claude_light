# Configuration reference

Constants live in `claude_light/config.py`. Environment variables override the defaults where shown. How each part is used is in [architecture/](architecture/ARCHITECTURE.md).

| Variable | Purpose |
|---|---|
| `--provider {claude,gemini,codex}` | Command-line flag, default `claude`. Selects the agent (`claude_light/providers/__init__.py: NAMES`, `DEFAULT`) |
| `providers/runner.py: TIMEOUT_SECS` | Seconds a Gemini or Codex run may take before its process tree is killed (default 600) |
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
| `SESSION_DIR` | Where `session.py` saves each project's conversation for `--continue`, one file per project named by a hash of its path (`~/.claude_light_sessions`; override with `CLAUDE_LIGHT_SESSION_DIR`). In the home directory so a cloned repo cannot supply a conversation |
| `HEARTBEAT_SECS` | How often the heartbeat thread wakes (default 30) |
| `CACHE_TTL_SECS` | Idle seconds before the heartbeat warms the cache (default 240) |
| `MODEL_HAIKU` / `MODEL_SONNET` / `MODEL_OPUS` | Model ID constants used by the router |
| `MODEL_FABLE` | `claude-fable-5-1`. Never routed to: offered, with Opus 5.5 and Sonnet 5.5, when a question routes to Opus |
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
| `CALL_GRAPH` | Call-graph retrieval expansion: adds callers or callees of the best hit within the token budget (default off; skipped at low effort). Off because it sent 20-28% more tokens for no extra right-file hits, see [BENCHMARKS.md](BENCHMARKS.md). Enable with `CLAUDE_LIGHT_CALLGRAPH=1` |
| `CALL_GRAPH_MAX` | Most related chunks added per query by the call-graph expansion (default 3) |
| `ROUTE_ONCE` | Keep the model the session's first question routes to for every later turn; effort is still routed per turn, `/budget` still drops to Haiku at the cap, and `/clear` routes the model again. Default on. In `benchmark_session_cache.py` a session opening on Sonnet cost 71% more than per-turn routing, and one opening on Haiku cost 91% less because every later question, hard ones included, ran on Haiku. `CLAUDE_LIGHT_ROUTE_ONCE=0` routes the model per turn |
| `WORKTREE_BOOST` | Z-score boost for chunks in modified, staged or untracked files; hybrid path only (default 1.0, tuned with `benchmark_models.py --worktree`; 1.5 starts losing answers). Disable with `CLAUDE_LIGHT_WORKTREE_BOOST=0` |
| `CLAUDE_LIGHT_RUN_TIMEOUT` | Timeout in seconds for the `shell` tool and the executor's commands (default 120; 0 or blank disables the cap) |
