---
name: claude_light
description: Lets an agent find, change and test claude_light, an interactive CLI that answers questions about and edits a codebase with Claude, using prompt caching and retrieval.
---

You are working on `claude_light`, a Python CLI. It sends a cached project skeleton and the code chunks each question retrieves to Claude, and applies the SEARCH/REPLACE edits Claude returns. Change the package with the tests as the guard, and measure any change that affects tokens before you keep it.

- `claude_light/`: the package, one module per responsibility. Module roles and the data flow are in [docs/architecture/components.md](docs/architecture/components.md).
- `tests/`: unit, integration and utility suites, regression fixtures and benchmark scripts. See [TESTING.md](TESTING.md).
- `docs/`: architecture, configuration, benchmarks and CI reference. See [docs/DOCS.md](docs/DOCS.md).
- `installers/`: install scripts for Linux, macOS and Windows.
- `scripts/`: SBOM generation and a startup benchmark.
- `.github/workflows/`: CI. See [docs/WORKFLOW.md](docs/WORKFLOW.md).

Repository rules, invariants and commands are in [CLAUDE.md](CLAUDE.md).
