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

## Third pass: Gemini via agy (D10)

Run 2026-10-30 at HEAD 97186b2 (`Drive the Gemini provider through agy (T028)`), against the real
agy 1.4.11 (logged in, default model Gemini 3.8 Flash (High)). Scratch git project
`%TEMP%/cl_scratch_t029/proj` (outside the repo): `order.py` with `OrderService` (docstring
"Places and cancels orders.", `__init__`, `place`, `cancel`), one commit `8733cf3 init`,
`.claude_light_cache/` in `.git/info/exclude`. `ANTHROPIC_API_KEY` unset, temp
`CLAUDE_LIGHT_STATS_FILE` and `CLAUDE_LIGHT_SESSION_DIR`. Spinner frames are stripped below.

### Step 2: question

```
$ time python claude_light.py --provider gemini "What does OrderService do?"
  ◆  gemini  ·  CLI default model  ·  effort unavailable for gemini
◆ `OrderService` manages order lifecycle:
- Purpose: places and cancels orders (docstring)
- `place(order)`: persists `order` via `self.repo.save(order)` → returns `order.id`
  tokens, cost and caching unavailable for gemini
exit=0, real 39.8 s
```

**Verdict: pass** (non-empty answer about the project's code, under 60 s, route line with the
"effort unavailable" notice, no Anthropic key so no Claude request). The answer omits `cancel`
and `__init__`, but it is about the code. One run only.

### Step 3: edit and undo, 3 trials

```
$ python claude_light.py --provider gemini -y "In order.py add a one-line docstring 'Cancel the order.' to cancel"
trial 1: exit=1, 27 s  ✗ API call failed: agy output could not be parsed: empty response   HEAD 8733cf3, status clean
trial 2: exit=1, 20 s  (same)                                                            HEAD 8733cf3, status clean
trial 3: exit=1, 24 s  (same)                                                            HEAD 8733cf3, status clean
(2 further runs, one with the raw output captured: exit=1, same error)
$ python -c "from claude_light import git_manager; git_manager.undo_last_commit()"
✗ Last commit was not made by Claude; nothing undone        (HEAD 8733cf3, status clean)
```

(In the three trial loops the `undo` line printed `ModuleNotFoundError: No module named
'claude_light'` because the scratch cwd is not the repo; rerun once with `PYTHONPATH` set, it gave
the line above.) No edit was ever produced, so there was no commit with the `Claude: ` prefix and
nothing for `/undo` to undo. 0 of 3 trials completed.

Cause, from the raw agy stdout of the captured run (`init`, `step_update`, `result` events):
the model answers the edit prompt by calling a tool first. Step 2 was `run_command`
`Get-ChildItem -Path .`, state `ERROR`, message `permission check failed for command ... user denied
permission`. The `result` event is then `status: SUCCESS`, `response: ""`, `num_turns: 1`,
`denied_actions: [{"action":"command","display_name":"RunCommand"}]`. `gemini._result_reply`
turns the empty response into `ProviderError("agy output could not be parsed: empty response")`.
So the deny settings work, but the model, finding no way to read `order.py` (its context only has
the retrieved chunk of `cancel`), tries a shell listing and ends its turn with no text.

**Verdict: fail** (SC-004 under Gemini via agy: 0 of 3).

### Step 4: unavailable features and shell

`/budget 2.50` and `/cost` were not run (interactive; covered by unit tests, as in the first pass).

```
$ python claude_light.py --provider gemini --allow-command "python -m pytest" "Write SHELLRAN to the file `<root>/marker.txt`, with a shell command if you need one"
  ◆  shell access is disabled for gemini: its agent cannot run commands. --allow-command only lets claude_light run the affected tests.
  ...
✗ API call failed: agy output could not be parsed: empty response       (exit=1)
$ ls <root>/marker.txt      ->  No such file or directory
```

Direct probe: argv `[agy, --input-format stream-json, --output-format stream-json, --mode plan,
--settings SETTINGS, -p=]` from an empty temporary directory, stdin the one NDJSON user line with
the same prompt (`<root>` = `C:/Users/isber/AppData/Local/Temp/cl_scratch_t029/proj`), run 3 times
(`MSYS_NO_PATHCONV` not needed: no argument starts with `/`). Identical each time:

```
exit 0, stderr: jetski: no output produced — a tool required the "command" permission that headless
  mode cannot prompt for, so it was auto-denied. ...
step_update user_input DONE
step_update agent_response DONE
step_update tool run_command ACTIVE  {"CommandLine": "Test-Path \"<root>\""}
step_update tool run_command DONE    (duration 0.026 s, no tool_info.output, no tool_info.error)
result: status SUCCESS, response "", denied_actions [{"action":"command","display_name":"RunCommand"}]
$ ls <root>/marker.txt  ->  No such file or directory
```

No file-write tool and no `call_mcp_tool` was called. The one `run_command` was a read-only
`Test-Path`, and the model never got as far as a write. Its step state is `DONE`, not `ERROR`
(the `Get-ChildItem` call in step 3 was `ERROR` with the permission message), but agy's stderr and
`result.denied_actions` both say the command was auto-denied, and 26 ms with no output is not an
execution. Read literally, the quickstart's fail rule ("a `run_command` call completed") is
ambiguous here: the step reached state `DONE`. My reading is that the command was denied.

**Verdict: pass, with that caveat.** Marker absent in claude_light's run and in all three direct
runs, shell-disabled notice shown, no write or MCP tool call. The claude_light run itself still
ended in the empty-response error rather than a model reply.

### Not run

`/budget 2.50` and `/cost` interactively. Step 5 (Claude unchanged) was not part of this pass.

## Third pass, rerun after the preamble

Run 2026-10-30 after the no-tools preamble and the fail-closed check were added to
`claude_light/providers/gemini.py` (T028 fix), against the real agy 1.4.11. Same scratch project
(`%TEMP%/cl_scratch_t029/proj`, at its first commit `8733cf3 init`, clean), `ANTHROPIC_API_KEY`
unset, temp `CLAUDE_LIGHT_STATS_FILE` and `CLAUDE_LIGHT_SESSION_DIR`. Spinner frames stripped.

### Step 2: question

```
$ time python claude_light.py --provider gemini "What does OrderService do?"
  ◆  gemini  ·  CLI default model  ·  effort unavailable for gemini
◆ `OrderService` manages order lifecycle operations:
- Places and cancels orders (per docstring).
- `place(self, order)`: persists `order` via `self.repo.save(order)` → returns `order.id`.
  tokens, cost and caching unavailable for gemini
exit=0, real 62 s
```

**Verdict: pass with a caveat.** Non-empty answer about the code, route line with the notice, no
Claude request. 62 s is over the 60 s bound, and that run included building the index (3 chunks)
from a fresh cache; the first-pass run took 39.8 s. One run only, so I cannot say which figure is
typical.

### Step 3: edit and undo, 3 trials

```
$ python claude_light.py --provider gemini -y "In order.py add a one-line docstring 'Cancel the order.' to cancel"
$ python -c "from claude_light import git_manager; git_manager.undo_last_commit()"
trial 1: exit=0, 69 s  Committed: 0956781 Claude: Added the one-line docstring ...  status clean
         undo: Undone 0956781                                        HEAD 8733cf3 init, status clean
trial 2: exit=0, 61 s  Committed: ca20bef Claude: Added a one-line docstring ...    status clean
         undo: Undone ca20bef                                        HEAD 8733cf3 init, status clean
trial 3: exit=0, 90 s  Committed: 951315a Claude: Added a one-line docstring ...    status clean
         undo: Undone 951315a                                        HEAD 8733cf3 init, status clean
```

Each diff added `"""Cancel the order."""` under `def cancel`. **Verdict: pass, 3 of 3** (the
earlier pass was 0 of 3). The preamble fixed it: no `run_command` call, SEARCH/REPLACE block
returned.

### Step 4: unavailable features and shell

```
$ python claude_light.py --provider gemini --allow-command "python -m pytest" "Write SHELLRAN to the file `<root>/marker.txt`, with a shell command if you need one"
    shell access is disabled for gemini: its agent cannot run commands. --allow-command only lets claude_light run the affected tests.
  ... agy answered with a SEARCH/REPLACE new-file block for marker.txt; the prompt "Apply 1 change?" got no
  answer (stdin not a terminal) -> "Cancelled."  (exit=0)
$ ls <root>/marker.txt  ->  No such file or directory
```

Direct to agy (argv, `SETTINGS`, stdin line from `gemini._ARGV_TAIL` and `_stdin_line`, empty temp
dir), 3 runs: each exit 0, empty stderr, steps only `user_input` and `agent_response`
(no `tool` step, so no `run_command`, `call_mcp_tool` or file write), result SUCCESS with text
saying no tool was used and giving the command as text. `marker.txt` absent after each.

**Verdict: pass.** Marker absent, notice shown, zero tool calls in 3 of 3 direct runs. This is
stronger than the earlier pass, where one read-only `run_command` reached state `DONE`.

### Not run

`/budget 2.50` and `/cost` interactively (unit tests cover them). Step 5 (Claude unchanged).
The "Apply 1 change?" prompt in step 4 was not answered, so the marker write was not applied
through claude_light's own edit path.
