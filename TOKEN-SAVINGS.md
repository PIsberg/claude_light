# Token savings plan

Goal: reduce what claude_light spends per turn without hurting answer quality.
Work the levers in order. Change one at a time and measure before and after.
A lever ships only when its measurement shows a gain.

## Levers

- [ ] **1. Keep the prompt cache warm.** Avoid switching models between consecutive turns, since a switch may pay a fresh cache write. Check whether a stronger model is worth that write on the next turn.
  - Open question: is Anthropic's prompt cache scoped per model? Not verified against the API docs. A web search did not settle it either.
  - Status: not changed. Measuring it needs live API calls that show cache writes and reads across model switches. A sticky-model rule also trades cost against answer quality, and nothing offline measures quality.
- [x] **2. Confirm LLMLingua is running.** Done: the CPU-only torch build could not load the model (llmlingua defaults to CUDA), so nothing compressed. Fixed; a 2,537-token sample compressed to 1,324. It is a silent no-op until `pip install llmlingua`. Check that `LLMLINGUA_MIN_TOKENS` (default 800) is low enough to trigger on typical retrieval sizes.
- [x] **3. Lower the retrieval budget where answers are short.** `_RETRIEVAL_BUDGET` is set by effort (low 1500, medium 3000, high 6000, max 9000). Measured offline with `benchmark_models.py --budgets`: going below the current budgets loses answers, so nothing changed. See Results.
- [x] **4. Tighten escalation.** Measured offline with `benchmark_router.py`: no query on the repo set lands in the band that `ROUTER_ESCALATE_MARGIN` can change, at any margin tried. Nothing changed. See Results.
- [ ] **5. Route large file reads to a cheaper model (Spotify-style).** Only after measuring how many tokens go to agent or tool reads. Targets: API mode `tools.py` `read_file`; subscription mode needs a settings-file hook. Files needed for a SEARCH/REPLACE edit must stay exact.
  - Status: not changed. The read-token count needs live API runs in API mode, which this offline pass did not make.

## How to measure

- Use `/cost` and the stats file `~/.claude_light_stats.json` before and after each change.
- Run the same set of questions each time, and record: input tokens, cached tokens, output tokens, cost, and turns per question.
- Offline, without an API key: `python tests/benchmarks/benchmark_models.py --budgets` (recall and tokens per effort budget) and `python tests/benchmarks/benchmark_router.py` (tier and model per question, with `--margin` to replay another escalation margin). Both measure routing and retrieval, not answer quality.
- Keep a result line per lever here once measured.

## Results

| Lever | Change | Input tokens | Cost | Turns | Notes |
|---|---|---|---|---|---|
| baseline | none | | | | |
| 3. budget sweep | none | not measured offline | not measured offline | n/a | Answering file reached the context: low 13/23 identifier and 8/12 paraphrase, medium 14/23 and 9/12, high 16/23 and 9/12. Median tokens sent: low 1,141, medium 1,901, high 1,997. Kept the budgets. |
| 4. router replay | none | not measured offline | not measured offline | n/a | 41 repo questions: 32 low, 2 medium, 7 high, 0 max. Zero scores in the band at margins 0.15, 0.30 and 0.50. Kept the margin. |
| retrieval: symbol-only embed text | reverted | not measured offline | not measured offline | n/a | Identifier recall 14/23 to 17/23, but off-topic questions got context (6/6 to 2/6) and median tokens at medium rose 1,901 to 2,571. Not a gain, so reverted. See docs/BENCHMARKS.md. |
| retrieval: call-graph expansion | off by default | not measured offline | not measured offline | n/a | 43 questions: right-file hits unchanged at every budget (medium 19/25 and 11/12 both ways). Mean tokens rose 1,802 to 2,169 at medium, 2,829 to 3,503 at high, 3,586 to 4,603 at max. Opt-in with `CLAUDE_LIGHT_CALLGRAPH=1`. See docs/BENCHMARKS.md. |
| retrieval: working-tree boost | shipped at 1.0 | not measured offline | not measured offline | n/a | Medium budget. Answering file modified: identifier 19/25 to 25/25, mean tokens 1,802 to 1,658. Unrelated file modified: unchanged at 19/25 and 11/12, mean tokens 1,802 to 1,829. Boost 1.5 drops paraphrase to 9/12 in that case. See docs/BENCHMARKS.md. |
| tools: `fetch_chunk` and `shell` schemas | shipped | about +320 per request, estimated | cached prefix | not measured | Both schemas together are 1,282 characters (`_estimate_tokens`: 182 + 138). They sit in the cached prefix, so after the first turn they bill at the cache-read rate. Whether `fetch_chunk` saves reads or turns needs live API runs. |
| retrieval: exclude test directories for package questions | shipped | not measured offline | not measured offline | n/a | Same run, 43 questions, medium budget: identifier 15/25 to 20/25, paraphrase 8/12 to 11/12, off-topic 6/6 both, median tokens 1,901 to 1,570. Lower or equal tokens and higher recall at every budget. Questions that mention tests keep them. See docs/BENCHMARKS.md. |

## Decisions

- `LLMLINGUA_MIN_TOKENS` stays at 800 (now settable with `CLAUDE_LIGHT_LLMLINGUA_MIN_TOKENS`). Every retrieval budget is 1 500 or more, so a full retrieval always qualifies.
- Retrieval budgets stay as they are (lever 3). Low drops one identifier and one paraphrase answer against medium, a recall loss rather than a gain. High adds two identifier answers over medium for about 100 more median tokens.
- `ROUTER_ESCALATE_MARGIN` stays at 0.15 (lever 4). Only scores between 12.0 and 12.0 × (1 + margin) change tier, and no repo question scores there. A larger margin would move questions off Opus on no evidence.
- Call-graph expansion is off by default. It sent 20-28% more tokens for no extra right-file hit. `fetch_chunk` still gives the model callers and callees when it asks for them.
- `WORKTREE_BOOST` is 1.0, up from an untuned 0.5. It is the largest value that lost nothing when an unrelated file was the modified one.
- Max budget (9 000): not changed. On the repo set, max does not beat high on recall, and it sends about 700 more mean tokens (3,443 against 2,732). No repo question routes to max, so that result says nothing about the questions that do. Measure with max-tier questions before lowering it.
