# Testing

How to run the tests, where a new test goes, and what CI runs. For the design of the suite (mocks, flow diagrams, categories), see [docs/test_strategy.md](docs/test_strategy.md). For the CI steps, see [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Run the tests

```bash
pytest                                   # everything pytest collects, about 2.5 min on the dev machine
pytest tests/unit/test_retrieval.py      # one file
pytest tests/unit/test_retrieval.py::test_retrieve_top_k   # one test
pytest tests/unit/ -x --timeout=120      # unit tests only, stop at first failure, per-test timeout
```

No API key is needed. `tests/conftest.py` sets a mock key, so nothing calls the real API.

The CI unit step runs two files first, `test_editor.py` and `test_config.py`, before the rest. Both must run before `MockManager` patches `Path`, so run them first if you see a path-related failure that only appears in a full run.

## Layout

| Directory | What is in it | Run by |
|---|---|---|
| `tests/unit/` | One file per module (`test_retrieval.py` for `retrieval.py`, and so on), plus `test_benchmark_budget.py` for the benchmark helpers | CI `test.yml`, and `pytest` |
| `tests/integration/` | Import and wiring checks across modules, the git manager, the API tool loop, and the benchmark scripts | CI `test.yml`, and `pytest` |
| `tests/utilities/` | Mock and download-progress helpers | CI `test.yml`, and `pytest` |
| `tests/linting/` | Scripts, not pytest tests: `check_regression.py` (the regression gate CI calls) and two syntax-check scripts | `test.yml` runs `check_regression.py`; the syntax scripts are run by hand |
| `tests/test_fuzz.py` | Hypothesis property tests for the parsers and the router | `pytest`; `fuzz.yml` runs it with a larger example budget |
| `tests/benchmarks/` | Scripts that measure token and retrieval changes | Not collected by pytest. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| `tests/fixtures/` | Static data: question sets, baseline stats, a small repo | Not collected by pytest |

`pytest.ini` sets `testpaths = tests` and excludes `tests/fixtures` and `tests/benchmarks`.

CI runs the unit, integration and utilities suites, plus the two benchmark regression checks. The fuzz suite runs in its own workflow.

## Environment the suite sets

`tests/conftest.py` sets these before any test imports the package:

- `ANTHROPIC_API_KEY=sk-ant-test-mock-key`: unit tests never reach the API.
- `CLAUDE_LIGHT_HYBRID=0`: the suite uses one auto-tuned embedding model. `tests/unit/test_hybrid.py` switches hybrid retrieval back on for its own tests.
- `CLAUDE_LIGHT_STATS_FILE` points at a temp file, so a test run never writes the real `~/.claude_light_stats.json`.

If you add a test that needs one of these changed, set it inside the test, not in the module.

Two autouse fixtures in the same file run around every test:

- `_fresh_session_model` clears `state.session_model`, so one test's route does not pin the next test's model.
- `_no_real_llmlingua_loader` fails a test that starts the real LLMLingua loader thread. `main()` starts one, and it keeps loading after the test ends, then sets the compressor singleton under a later test (#74). A test that calls `main()` patches `claude_light.main._start_llmlingua_load`, as `TestMainRouting.setUp` does.

## Where a new test goes

Add the test to the file for the module you changed, in the existing style. A test that only runs on your machine is not coverage, so do not put it in a new directory that CI does not collect.

| You changed | Test in |
|---|---|
| A module under `claude_light/` | `tests/unit/test_<module>.py` |
| Routing (`route_query`) | `tests/unit/test_llm.py`, plus `tests/test_fuzz.py` for inputs |
| Retrieval budgets or selection | `tests/unit/test_retrieval.py`; `test_hybrid.py` for the hybrid path |
| Config constants | `tests/unit/test_config.py` (runs first in CI) |
| Edit parsing or application | `tests/unit/test_editor.py` (runs first in CI) |
| A benchmark script's logic | `tests/unit/test_benchmark_budget.py` or a new `tests/unit/test_<script>.py`. Load the script by path, because pytest does not collect `tests/benchmarks/` |

## Rules for tests

- **Write the failing test first** when you can. Run it and see it fail, then write the fix. A test written after the fix may never have been able to fail.
- **A behaviour change ships with a test.** A bug fix needs a regression test that fails on the old code. If you cannot write one, say why in the change.
- **Test what a caller observes**, not internals. A test pinned to a private helper breaks on every refactor and misses real defects.
- **Check that the test can fail.** Break the fix once on purpose and confirm the test goes red. A suite that passes because the code under test never ran is the most expensive kind of green.
- **Token and retrieval claims are measured, not asserted.** A change to what gets sent to the API needs a before and after from a benchmark, and only gains are kept. See [TOKEN-SAVINGS.md](TOKEN-SAVINGS.md).

## Benchmarks and regression gates

CI runs two benchmark regression checks as part of `test.yml`. They compare freshly generated numbers with committed baselines in `tests/fixtures/`:

- `benchmark.py --json` against `baseline_token_stats.json` (analytical token savings)
- `benchmark_retrieval.py` on the offline fixture against `baseline_retrieval_stats.json`

If one fails, the change moved a number the baseline pins. Confirm the move is intended before you update a baseline, and update the baseline in the same change.

The `*_new.json` files these commands write are scratch output. Do not commit them.

## Fuzz tests

`tests/test_fuzz.py` uses Hypothesis. A plain `pytest` run uses Hypothesis's defaults. The weekly `fuzz.yml` run sets `HYPOTHESIS_MAX_EXAMPLES=500` and a 2 s deadline per example. To reproduce a failure, copy the seed Hypothesis prints.
