# HANDOVER

Read this first in a new session. It records where the unique-features work stands.

## Where we are

The developer asked which unique features to add beyond Claude Code. Six were proposed and all were chosen. Five shipped in PR #87 (merged 2026-10-25). Feature 4, cache-aware routing, is on branch `feature/cache-stable-routing`: the effort half is built, the model half was measured and rejected.

History: the first sessions had no shell tool, and claude_light auto-committed their edits straight onto local `main` as seven `Claude: ...` commits. On 2026-10-25 those commits were moved onto this branch and local `main` was reset to `origin/main`. Nothing reached `origin/main`.

### Verified on 2026-10-25 (Windows, Python 3.11.4; CI runs 3.12)

- In CI order: `test_editor.py` 46 passed, `test_config.py` 9 passed, the rest of `tests/unit` 852 passed and 1 skipped (a symlink test that needs a Windows privilege; it runs on CI's Linux). `tests/integration` and `tests/utilities`: 18 passed.
- Token benchmark and regression check: warm total $0.0622 against a $0.0622 baseline. Retrieval fixture: Hit@10 1.000, Recall@10 1.000, the same as the baseline.
- One test was wrong: `test_timeout_kills_command` put `;` in its command, so the forbidden-character check refused it and the timeout path never ran. The test is fixed and now bounds the elapsed time at 4 s. With the tree-kill removed it fails at 6.0 s; with the real code it passes at 1.25 s.

## Feature status

| # | Feature | Status | Where |
|---|---|---|---|
| 1 | `fetch_chunk(id)` tool: up to 5 method bodies by `path::symbol` or bare name, with callers and callees | Shipped | `claude_light/tools.py` |
| 2 | Call-graph retrieval expansion | Built, **off by default**: no recall gain, 20-28% more tokens. `CLAUDE_LIGHT_CALLGRAPH=1` turns it on | `claude_light/callgraph.py`, `retrieval.py::_related_pairs` |
| 3 | Working-tree boost for modified, staged or untracked files. Hybrid path only | Shipped, tuned to 1.0 | `retrieval.py` (`_modified_paths`, `_boost_modified_chunks`) |
| 4 | Cache-aware routing | Effort: fixed top-level effort plus an effort message per turn. Model: the first question's model is kept by default, the developer's choice (`ROUTE_ONCE`; +71% opening on Sonnet, -91% opening on Haiku; `CLAUDE_LIGHT_ROUTE_ONCE=0` routes per turn). Pinning to the strongest model cost 48-67% more and was rejected | `llm.py::_apply_effort`, `_finalize_route`, `warm_cache` |
| 5 | Run only the affected tests after an edit | Shipped | `claude_light/affected_tests.py`, hook in `editor.apply_edits` |
| 6 | `/budget` command and per-turn cost estimate | Shipped | `state.py`, `llm.py::_finalize_route`, `main.py` |
| extra | Top-tier prompt: a question routed to Opus asks Opus 5.5, Fable 5.1 or Sonnet 5.5 (Sonnet without a terminal) | Built, stacked on #91 | `llm.py::_top_tier_model`, `_finalize_route` |
| extra | `shell` tool (allowlisted commands only) | Shipped | `claude_light/tools.py` |

## Measurements

All offline, with `tests/benchmarks/benchmark_models.py` on this repo's 43 questions. Tables are in `docs/BENCHMARKS.md` section 2b; result lines are in `TOKEN-SAVINGS.md`.

- **Call graph (`--callgraph`):** right-file hits unchanged at every budget. Mean tokens: medium 1,802 to 2,169, high 2,829 to 3,503, max 3,586 to 4,603. Hence off by default.
- **Worktree boost (`--worktree`):** with the answering file modified, identifier hits go from 19/25 to 25/25 at 1.0. With an unrelated file modified, nothing is lost up to 1.0, and 1.5 drops paraphrase from 11/12 to 9/12. Hence 1.0.
- **`fetch_chunk` and `shell` schemas:** about 320 more tokens per request (estimated from characters), in the cached prefix. Whether `fetch_chunk` saves reads or turns needs live API runs (#85).

`tests/benchmarks/benchmark_retrieval.py` cannot measure any of this. It reimplements chunking, ranks by cosine alone and never imports `claude_light.retrieval`, so the `CLAUDE_LIGHT_CALLGRAPH` and `CLAUDE_LIGHT_WORKTREE_BOOST` env vars have no effect on it. The earlier plan to measure with it was wrong.

## Design notes

- **fetch_chunk:** reads `state.chunk_store` and errors if the index is not ready. Ambiguous or missing ids return a message.
- **Call graph:** matches `name(` in chunk bodies against chunk symbol names. Same-file definitions win. Names defined in more than 3 places are skipped. Nothing is written to disk. Config: `CALL_GRAPH`, `CALL_GRAPH_MAX`.
- **Worktree boost:** `_modified_paths()` uses `git_manager.get_modified_files()` and caches for 5 s. Test-file exclusion still wins over the boost. Matching compares whole path components (`_same_or_inside`).
- **shell tool:** see `docs/architecture/components.md`, "Agent mode and allowed commands". It is always in `TOOLS`, so the prompt-cache prefix written by `warm_cache` stays stable.
- **/budget** and **affected tests:** see `CLAUDE.md`, "Tools, `/budget` and affected tests".

## Open work

- PR #89 (`feature/cache-stable-routing`) and the route-once PR stacked on it (`feature/session-model-pin`): the merges are the developer's call, #89 first, with merge commits.
- Effort messages are verified offline only (mocks and the SDK stub server). A live run must confirm the API accepts them on Haiku, Sonnet and Opus 5.5 and that `cache_read_input_tokens` holds across an effort change; see the issue linked from that PR.
- #85: measure `fetch_chunk` per-turn tokens with live API runs.
- #86: decide whether call-graph expansion helps answer quality, which the offline benchmark cannot see. Delete the expansion path if it does not.

## Gotchas

- Every source edit changes the chunk set, so `benchmark_models.py` re-embeds all chunks with mpnet: about 10 minutes on CPU.
- The suite runs with `CLAUDE_LIGHT_HYBRID=0`, so the worktree boost (hybrid path only) is covered by `test_worktree_boost.py` and `benchmark_models.py`, not by the general retrieval tests.
- CI runs `test_editor.py` and `test_config.py` first, before `MockManager` patches `Path`.
- Invariants in `CLAUDE.md` still apply: background threads never print, shared state goes through `state.py` under `state.lock`, and tests never call the real API.
