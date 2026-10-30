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

**Verdict: fail** for the edit requirement (step 3 produced no edit, SC-004 not met); steps 2 and
the shell half of 4 passed.

## Step 5: Claude unchanged

Not run in this pass (token benchmarks belong to the gatekeeper round).
