# HANDOVER

Read this first in a new session. It records where the `feature/chunk-tools-callgraph` work stands.

## Where we are

The developer asked which unique features to add beyond Claude Code. Six were proposed and all were chosen. The code for most of them is written and sits **uncommitted on `main`**.

**Nothing has been run.** The sessions that wrote this code had no shell tool. No branch was created, pytest was not run, and nothing was committed. Treat all of it as untested.

### Static-review fixes (2026-10-25, also not run)

A read-through without a shell found and fixed these. Each has a new regression test.

- **`tools._shell`:** the forbidden-character check ran *after* whitespace was collapsed, so a newline turned into a space and `echo a\necho b` got through. The check now runs on the raw command. Test: `test_newline_refused_before_whitespace_is_collapsed`.
- **`tools._shell` timeout:** `subprocess.run(shell=True, timeout=…)` killed only `cmd.exe` on Windows. The child kept the pipe open, so the timeout had no effect. It now uses `Popen` + `communicate(timeout)` and kills the whole tree (`taskkill /T` on Windows, `killpg` on POSIX via `start_new_session`). `_kill_tree` is a copy of the helper in `llm.py`, because importing `llm` runs the auth check.
- **`retrieval._boost_modified_chunks`:** a plain `endswith` boosted `src/oauth.py` when `auth.py` was modified. Matching now goes through `_same_or_inside()`, which compares whole path components and also handles git paths relative to a repo root above the cwd. Tests: `test_boost_matches_whole_path_components`, `test_boost_git_path_relative_to_repo_root_above_cwd`.
- **`affected_tests.is_test_file`:** it counted `conftest.py` and every `.py` under `tests/` (fixtures, helpers) as a test file. Passed to pytest alone, those collect nothing and exit with code 5, which read as a failure. Only `test_*.py` and `*_test.py` count now. Test: `test_helpers_under_tests_are_not_run_directly`.
- **`editor.apply_edits`:** the affected-tests run (and its "revert the last commit?" offer) now happens only if `auto_commit` returned True. Otherwise the revert could target an older commit.
- **`llm._route_result`:** over budget, a query already routed to Haiku kept its high or max effort. It now drops to low effort too, as documented.

### Session log (latest first)

- **2026-10-25, third session (no shell tool again):** The `shell` tool was reviewed by reading `claude_light/tools.py`. It is **already built**: schema in `TOOLS`, dispatch in `run_tool`, allowlist, forbidden-character check, tree-kill timeout and output trimming are all there. Do not build it again. No code changed. Still nothing run, no branch, no commit. Next action is step 1 under "Do this next".

## Feature status

| # | Feature | Status | Where |
|---|---|---|---|
| 1 | `fetch_chunk(id)` tool: up to 5 method bodies by `path::symbol` or bare name, with callers and callees | Written, not run | `claude_light/tools.py` |
| 2 | Call-graph retrieval expansion: adds up to 3 callers or callees of the best hit within the token budget. Low effort skips it | Written, not run | `claude_light/callgraph.py`, `retrieval.py::_finish()` |
| 3 | Working-tree boost: raises the score of chunks in modified, staged or untracked files. Hybrid path only | Written, not run | `retrieval.py` (`_modified_paths`, `_boost_modified_chunks`) |
| 4 | Cache-aware routing | **Not built** | n/a |
| 5 | Run only the affected tests after an edit | Written, not run | `claude_light/affected_tests.py`, hook in `editor.apply_edits` |
| 6 | `/budget` command and per-turn cost estimate | Written, not run | `state.py`, `llm.py::_route_result`, `main.py` |
| extra | `shell` tool (allowlisted commands only) | Written, not run | `claude_light/tools.py` |

## Design notes

- **fetch_chunk:** reads `state.chunk_store` and errors if the index is not ready. Ambiguous or missing ids return a message.
- **Call graph:** matches `name(` in chunk bodies against chunk symbol names. Same-file definitions win. Names defined in more than 3 places are skipped. Nothing is written to disk. `CLAUDE_LIGHT_CALLGRAPH=0` disables it. Config: `CALL_GRAPH`, `CALL_GRAPH_MAX`.
- **Worktree boost:** `_modified_paths()` uses `git_manager.get_modified_files()` and caches for 5 s. Test-file exclusion still wins over the boost. `WORKTREE_BOOST` defaults to 0.5, which is a guess. `CLAUDE_LIGHT_WORKTREE_BOOST=0` disables it.
- **shell tool:**
  - It runs one command from the project root, and only if it starts with a prefix given via `--allow-command`. Prefixes match whole words.
  - Commands containing `; & | < > `` ` `` $` or a newline are rejected.
  - A `powershell:` entry runs in PowerShell.
  - The timeout is 120 s, or `CLAUDE_LIGHT_RUN_TIMEOUT`. Output keeps the first 20 and last 80 lines. A non-zero exit returns an error result.
  - It is always in `TOOLS`, even with no allowlist, so the prompt-cache prefix written by `warm_cache` stays stable.
- **/budget:** `/budget 2.50` sets a cap, `/budget off` removes it, `/budget` shows cap and spend. At the cap, routing drops to Haiku at low effort. The first time in a session, the user is asked to continue or skip. While a budget is set, the route line shows an estimate such as `~$0.012`.
- **Affected tests:** finds test files that mention the edited module's name. An edited test file maps to itself. It runs `<allowed pytest prefix> <files> -q` after the auto-commit, and only if a pytest command was allowed. On failure it prints the output and, in a terminal, offers to revert the commit. It never reverts on its own. It does not run after OAuth-agent edits (`_commit_agent_edits`).
- **Tests added or changed:**
  - New: `tests/unit/test_callgraph.py`, `test_worktree_boost.py`, `test_shell_tool.py`, `test_budget.py`, `test_affected_tests.py`.
  - Updated: three tool-name assertions (`test_tools.py`, `test_llm.py`, `test_api_tool_loop.py`) now include `fetch_chunk` and `shell`.

## Why feature 4 is not built

A rule like "stay on the previous model to keep the cache warm" cannot be checked without live API runs showing cache writes and reads across model switches. It also trades answer quality against cost, and nothing offline measures quality. See lever 1 in `TOKEN-SAVINGS.md`. Do not ship a rule that cannot be measured.

## Do this next

1. Create the branch and run the tests (PowerShell):
   ```powershell
   cd C:\dev\private\claude_light
   git checkout -b feature/chunk-tools-callgraph
   python -m pytest tests/unit/test_editor.py tests/unit/test_config.py -q
   python -m pytest tests/unit/test_callgraph.py tests/unit/test_worktree_boost.py tests/unit/test_shell_tool.py tests/unit/test_budget.py tests/unit/test_affected_tests.py tests/unit/test_tools.py -x -q
   python -m pytest tests/unit tests/integration -q --timeout=120
   ```
2. Fix any failures. Both earlier suspects were reviewed by reading the code:
   - `test_shell_tool.py`: quoting `sys.executable` on Windows looks right. Python wraps a `shell=True` command as `cmd /c "<command>"`, and cmd strips only the outer quotes. If it still fails, check paths that contain spaces.
   - `test_editor.py`: `check_after_edit` returns early when `config.ALLOWED_COMMANDS` is empty, which is the suite default, and it swallows exceptions. `MockManager` patching `Path` should not reach it. If an editor test mocks `auto_commit` with `return_value=None`, the hook is now skipped, which is fine.
   - `test_timeout_kills_command` should now finish in about 1 s on Windows. Before the fix it took about 30 s.
3. Commit only specific paths. Leave out `diff_dump.txt` and `dist/`:
   ```powershell
   git add claude_light tests/unit tests/integration HANDOVER.md CLAUDE.md docs/CONFIG.md
   git commit -m "Add fetch_chunk, call graph, worktree boost, shell tool, /budget, affected tests"
   ```
4. ~~Update the docs~~ Done (not committed): `docs/CONFIG.md` and `CLAUDE.md` now cover the new settings, tools, `/budget` and affected tests. They are included in the `git add` in step 3.

## Measurements still owed

The repo rule is that a token change needs a before/after measurement. See `docs/BENCHMARKS.md`. Keep a change only if it shows a gain.

- **Call graph:** run `tests/benchmarks/benchmark_retrieval.py` with `CLAUDE_LIGHT_CALLGRAPH=1` and `=0`. Keep expansion only if Hit@10 and Recall@10 improve.
- **Worktree boost:** same benchmark with `CLAUDE_LIGHT_WORKTREE_BOOST=0` against `0.5`. Tune the value from the result.
- **fetch_chunk:** measure per-turn token counts using `test_api_tool_loop`.
- Watch for failures in exact-output retrieval tests or token baselines in `tests/fixtures/`.

## Gotchas

- The suite runs with `CLAUDE_LIGHT_HYBRID=0`, so the worktree boost (hybrid path only) is not exercised by existing tests. `test_hybrid.py` turns hybrid on for its own tests.
- CI runs `test_editor.py` and `test_config.py` first, before `MockManager` patches `Path`.
- Invariants in `CLAUDE.md` still apply: background threads never print, shared state goes through `state.py` under `state.lock`, and tests never call the real API.

## Possible later work

Feature 4 (cache-aware routing), only once live API runs can measure it.
