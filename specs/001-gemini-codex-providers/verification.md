# Manual verification (T026)

Run 2026-10-30 on win32, claude_light branch 001-gemini-codex-providers, with gemini 0.44.1 and
codex-cli 0.162.1. Scratch project: a git repo with one file, `order.py`, holding `OrderService`
(`place`, `cancel`), clean tree. `ANTHROPIC_API_KEY` unset for every run (`env -u`). The
interactive steps (3 and the `/budget` and `/cost` half of 4) were not run interactively: no
terminal session was available, so one-shot runs stand in and the gaps are stated.

## Step 0: preconditions

```
$ gemini --version        -> 0.44.1
$ codex --version         -> codex-cli 0.162.1
$ codex login status      -> Logged in using ChatGPT
```

## Gemini

### Step 2: question

```
$ time python claude_light.py --provider gemini "What does OrderService do?"
◆ Building context...
⏺ Scanning project files...
  ◆  gemini  ·  CLI default model  ·  effort unavailable for gemini
  ⏺  order.py::cancel
⏺ Index ready — 1 files → 2 chunks (2 embedded with the static model; 2 still need dense vectors)
✗ API call failed: gemini failed (exit 1): An unexpected critical error occurred:IneligibleTierError: This client is no longer supported for Gemini Code Assist for individuals. To continue using Gemini, please migrate to the Antigravity suite of products: https://antigravity.google. Run gemini on its own to see the full error.
real	0m37.207s
user	0m0.000s
sys	0m0.015s
exit=1
real 0m37.207s   exit=1
```

The route line printed as the contract says and no Claude request was made. The tool rejected the
account (`research.md` R3). The error names the provider and the cause, and keeps the exit status 1.

Steps 3 and 4: not run, no answer to build on.

**Verdict: fail** (IneligibleTierError from Google for this account; decision D9).

## Codex

### Step 2: question

```
$ time python claude_light.py --provider codex "What does OrderService do?"
  ◆  codex  ·  CLI default model  ·  effort unavailable for codex
  ⏺  order.py::cancel
◆ I can't tell from the provided context: it shows only `order.py`'s `cancel(self, order)` method, not an `OrderService` definition.
  tokens, cost and caching unavailable for codex
real 0m15.129s   exit=0
```

Non-empty answer in 15 s, route line and unavailable-figures line as in the contract, no Claude
request. The answer is weak because retrieval returned only the `cancel` chunk of this tiny
project; that is retrieval, not the provider. Step 2: pass.

### Step 3: edit and undo (one-shot, not 3 interactive trials)

```
$ python claude_light.py --provider codex -y "In order.py ... add a one-line docstring 'Cancel the order.' to cancel"
  ◆  codex  ·  CLI default model  ·  effort unavailable for codex  ·  top tier unavailable for codex
◆ `cancel` has no docstring in the provided context. I can't safely produce an exact SEARCH/REPLACE block without the surrounding file text.
$ git status --short && git log --oneline -2
?? .claude_light_cache/
a520102 init
```

No edit was produced, so no commit and nothing to `/undo`. 0 of 3 trials completed (1 attempted).
SC-004 is not shown. Not run: `/undo`, trials 2 and 3.

### Step 4: unavailable features and shell

```
$ python claude_light.py --provider codex --allow-command "python -m pytest" "Run the shell command echo SHELLRAN > marker.txt"
  ◆  shell access is disabled for codex: its agent cannot run commands. --allow-command only lets claude_light run the affected tests.
◆ I can't run that shell command in this environment.
$ ls marker.txt
ls: cannot access 'marker.txt': No such file or directory
```

Shell notice and absent `marker.txt`: as expected. `/budget 2.50` and `/cost`: not run (interactive only).

**Verdict (first pass): fail** for the edit requirement (step 3 produced no edit, SC-004 not met);
steps 2 and the shell half of 4 passed. Superseded by the second pass below.

## Step 5: Claude unchanged

Not run in this pass (token benchmarks belong to the gatekeeper round). The gatekeeper ran them:
token gate warm total $0.0622 against the $0.0622 baseline, retrieval Hit@10 and Recall@10 1.000.

## Second pass (2026-10-30, after cee75d1)

Run by the main session against the same `gemini` 0.44.1 and `codex` 0.162.1, in a scratch git
repo holding one file, `order.py`: `class OrderService` with `__init__`, `place` and `cancel`.
`ANTHROPIC_API_KEY` unset; temp `CLAUDE_LIGHT_STATS_FILE` and `CLAUDE_LIGHT_SESSION_DIR`.

### Codex step 3: edit and undo, 3 trials

The edit could not use the interactive `/undo`, so each trial ran the edit one-shot and then called
`git_manager.undo_last_commit()`, the function `/undo` calls (`main.py:387`).

```
$ python claude_light.py --provider codex -y "In order.py add a one-line docstring 'Cancel the order.' to cancel"
$ python -c "from claude_light import git_manager; git_manager.undo_last_commit()"
start=c99a3b9
trial 1: 9c6f4ae Claude: Added the docstring to cancel.            -> Undone, HEAD=c99a3b9, status clean
trial 2: 15c8c8e Claude: Added the one-line docstring to cancel.   -> Undone, HEAD=c99a3b9, status clean
trial 3: f4a94d3 Claude: Added the docstring to cancel.            -> Undone, HEAD=c99a3b9, status clean
```

Two earlier edit runs of the same request also applied and committed (`7d91921`, `fe6dcdd`). The
first-pass failure did not reproduce in 5 runs; it was a single refusal by Codex.

**Verdict: pass** for SC-004 under Codex, 3 of 3.

### Codex step 2: question, after the low-effort context fix

Before cee75d1, a low-effort route sent Codex only the summary line `order.py :: place — def
place(self, order):`, and Codex answered that the context lacked the implementation. Low-effort
rendering is summary-only because Claude can fetch bodies with `fetch_chunk`; Gemini and Codex run
with tools off (D1) and cannot. cee75d1 renders low as medium for a provider without `EFFORT`.

```
$ time python claude_light.py --provider codex "What does OrderService do?"
◆ `OrderService` places and cancels orders. Its `place` method saves an order through `repo` and returns the order's ID.
exit=0, 13 s
```

**Verdict: pass** (SC-001 under Codex: answer about the project's code within 60 s).

### Gemini: error message after the T006 fix

```
$ python claude_light.py --provider gemini "What does OrderService do?"
✗ API call failed: gemini failed (exit 1): An unexpected critical error occurred:IneligibleTierError: This client is no longer supported for Gemini Code Assist for individuals. ... Run gemini on its own to see the full error.
exit=1, 11 s
```

FR-011 holds: the message names the provider and the real error, not a Node stack frame. The
verdict stays **fail** (D9): this account cannot use Gemini.

### Hostile repository config (FR-010, constitution V)

A scratch repo with `.gemini/settings.json` holding a `hooks.SessionStart` command and a
`tools.discoveryCommand`, each writing a marker file, and `.codex/config.toml` holding `notify` and
an MCP server doing the same.

- Gemini argv run in the repo directory, before 9487d40: both Gemini markers written.
- Gemini adapter (`_run`) called from inside the repo, after 9487d40: no marker written.
- Codex argv run in the repo directory: no marker written (untrusted folder, `--ignore-user-config`).
  Codex also runs in an empty temporary directory since 9487d40.

### Not run

`/budget 2.50` and `/cost` interactively (covered by `tests/unit/test_provider_route.py` and
`tests/unit/test_provider_usage.py`); every Gemini step past the auth failure.
