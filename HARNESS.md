---
name: claude_light
description: Lets an agent find, change and test claude_light, an interactive CLI that answers questions about and edits a codebase with Claude, using prompt caching and retrieval.
---

You are working on `claude_light`, a Python CLI. It sends a cached project skeleton and the code chunks each question retrieves to Claude, and applies the SEARCH/REPLACE edits Claude returns. Change the package with the tests as the guard, and measure any change that affects tokens before you keep it.

- `claude_light/`: the package, one module per responsibility. Module roles and the data flow are in [docs/architecture/components.md](docs/architecture/components.md).
- `claude_light/providers/`: the agents behind one interface, six modules: `__init__.py` (registry), `base.py` (`Provider`, `ProviderError`, capabilities, `json_events`), `claude.py`, `gemini.py` (the `agy` CLI), `codex.py` and `runner.py` (CLI subprocess runner). Select one with `--provider`. Their tests are `tests/unit/test_providers.py`, `test_provider_runner.py`, `test_provider_gemini.py`, `test_provider_codex.py`, `test_provider_isolation.py`, `test_provider_route.py`, `test_provider_startup.py`, `test_provider_usage.py` and `test_session_provider.py`.
- `tests/`: unit, integration and utility suites, regression fixtures and benchmark scripts. See [TESTING.md](TESTING.md).
- `docs/`: architecture, configuration, benchmarks and CI reference. See [docs/DOCS.md](docs/DOCS.md).
- `installers/`: install scripts for Linux, macOS and Windows.
- `scripts/`: SBOM generation and a startup benchmark.
- `.github/workflows/`: CI. See [docs/WORKFLOW.md](docs/WORKFLOW.md).

Repository rules, invariants and commands are in [CLAUDE.md](CLAUDE.md).
