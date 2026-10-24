# Token savings plan

Goal: reduce what claude_light spends per turn without hurting answer quality.
Work the levers in order. Change one at a time and measure before and after.

## Levers

- [ ] **1. Keep the prompt cache warm.** Avoid switching models between consecutive turns, since a switch may pay a fresh cache write. Check whether a stronger model is worth that write on the next turn.
  - Open question: is Anthropic's prompt cache scoped per model? Not verified against the API docs.
- [ ] **2. Confirm LLMLingua is running.** It is a silent no-op until `pip install llmlingua`. Check that `LLMLINGUA_MIN_TOKENS` (default 800) is low enough to trigger on typical retrieval sizes.
- [ ] **3. Lower the retrieval budget where answers are short.** `_RETRIEVAL_BUDGET` is set by effort (low 1500, medium 3000, high 6000, max 9000). Check whether the default effort is higher than narrow questions need.
- [ ] **4. Tighten escalation.** Check how often `ROUTER_ESCALATE_MARGIN` sends a query to a bigger model. Raise the threshold if cheaper answers are good enough.
- [ ] **5. Route large file reads to a cheaper model (Spotify-style).** Only after measuring how many tokens go to agent or tool reads. Targets: API mode `tools.py` `read_file`; subscription mode needs a settings-file hook. Files needed for a SEARCH/REPLACE edit must stay exact.

## How to measure

- Use `/cost` and the stats file `~/.claude_light_stats.json` before and after each change.
- Run the same set of questions each time, and record: input tokens, cached tokens, output tokens, cost, and turns per question.
- Keep a result line per lever here once measured.

## Results

| Lever | Change | Input tokens | Cost | Turns | Notes |
|---|---|---|---|---|---|
| baseline | none | | | | |

## Decisions

- (none yet)
