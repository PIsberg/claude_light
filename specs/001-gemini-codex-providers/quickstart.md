# Quickstart: manual verification against the real tools

The check FR-015, SC-001, SC-004 and SC-007 ask for, run by hand after the code is done (task
T026). It is not part of CI (constitution I). Record every command and its output in
`specs/001-gemini-codex-providers/verification.md`, one section per tool, each ending in a verdict:
`pass`, `fail` or `not run`.

## 0. Preconditions

```bash
gemini --version          # 0.44.1 here
codex --version           # codex-cli 0.162.1 here
codex login status        # "Logged in using ChatGPT" here
```

A tool that is not logged in gets `not run` and its remaining steps are skipped. Gemini on this
machine is logged in but rejected by Google (`research.md` R3); its verdict follows D9.

## 1. Scratch project

A small git project in a scratch directory, with a committed clean tree and `git config
user.email`/`user.name` set. The spec's question names `OrderService`; if the project has no such
class, ask about one it has and record the substitution.

## 2. Question (US1, SC-001), per provider `<p>` in `gemini codex`

```bash
time python3 /c/dev/private/claude_light/claude_light.py --provider <p> "What does OrderService do?"
```

Pass: a non-empty answer about the project's code within 60 s, the route line of
`contracts/provider.md`, and no Claude request (run once with `ANTHROPIC_API_KEY` unset).

## 3. Edit and undo (US2, SC-004), 3 trials per provider

Interactive, `python3 .../claude_light.py --provider <p>`, then ask for a one-line change to a
named file, accept the commit, run `/undo`, and after each trial:

```bash
git status --short && git log --oneline -1
```

Pass: one commit with the `Claude: ` prefix per trial, and after `/undo` an empty status and the
starting HEAD, 3 out of 3.

## 4. Unavailable features and shell (US3, US2 scenario 4)

In the same session: `/budget 2.50`, then `/cost`. Then restart with
`--allow-command "python -m pytest"` and ask "Run the shell command echo SHELLRAN > marker.txt".
Pass: the messages of `contracts/provider.md`, and `marker.txt` does not exist.

## 5. Claude unchanged (FR-006, constitution III)

```bash
python tests/benchmarks/benchmark.py --json > tests/baseline_token_stats_new.json
python tests/linting/check_regression.py tokens tests/fixtures/baseline_token_stats.json tests/baseline_token_stats_new.json
python tests/benchmarks/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/baseline_retrieval_stats_new.json
python tests/linting/check_regression.py retrieval tests/fixtures/baseline_retrieval_stats.json tests/baseline_retrieval_stats_new.json
```

The committed baselines are the "before"; record both commands' output and exit codes.
