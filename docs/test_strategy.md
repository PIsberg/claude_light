# Claude Light - Test Strategy

This document describes the testing strategy, architecture, and infrastructure for Claude Light. The test suite ensures code quality, prevents regressions, and validates the token optimization claims through automated regression checks.

## Table of Contents

1. [Test Suite Structure](#test-suite-structure)
2. [Test Execution Flow](#test-execution-flow)
3. [Mock Architecture](#mock-architecture)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Test Categories](#test-categories)
6. [Running Tests](#running-tests)

---

## Test Suite Structure

The test suite is organized into multiple directories, each serving a specific purpose.

![Test Suite Structure](plantuml/Test_Suite_Structure.png)

### Directory Organization

| Directory | Purpose | Test Count |
|-----------|---------|------------|
| `tests/unit/` | Unit tests for individual modules | 17 files |
| `tests/integration/` | Integration tests for component interactions | 3 files |
| `tests/utilities/` | Test utilities and mock implementations | 2 files |
| `tests/linting/` | Regression check scripts | 3 files |
| `tests/benchmarks/` | Benchmark and regression test runners | 4 files |
| `tests/fixtures/` | Test data and fixtures | JSON files |

### Key Test Files

| File | Tests |
|------|-------|
| `test_claude_light.py` | Core functionality (6,600+ lines) |
| `test_llm.py` | Query routing, API calls, history compression |
| `test_retrieval.py` | RAG retrieval, deduplication, filtering |
| `test_editor.py` | Edit parsing, SEARCH/REPLACE, apply edits |
| `test_indexer.py` | File indexing, chunking, caching |
| `test_mocks.py` | MockManager for synthetic testing |

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Test_Suite_Structure
title Claude Light - Test Suite Structure

skinparam backgroundColor #FEFEFE
skinparam rectangle {
  BackgroundColor<<unit>> #E3F2FD
  BackgroundColor<<integration>> #E8F5E9
  BackgroundColor<<utility>> #FFF3E0
  BackgroundColor<<fixture>> #F3E5F5
}

package "tests/" {
  rectangle "conftest.py\n(Shared Fixtures)" as Conftest
  
  package "unit/" <<unit>> {
    rectangle "test_claude_light.py\n(Core Tests)" as TestCore
    rectangle "test_llm.py\n(Routing & API)" as TestLLM
    rectangle "test_retrieval.py\n(RAG)" as TestRetrieval
    rectangle "test_editor.py\n(Edit Parsing)" as TestEditor
    rectangle "test_indexer.py\n(Indexing)" as TestIndexer
    rectangle "test_skeleton.py\n(Tree Building)" as TestSkeleton
    rectangle "test_linter.py\n(Syntax Check)" as TestLinter
    rectangle "test_streaming.py\n(Streaming)" as TestStreaming
    rectangle "test_retry.py\n(Backoff)" as TestRetry
    rectangle "test_ui.py\n(UI/Colors)" as TestUI
    rectangle "test_executor.py\n(Commands)" as TestExecutor
    rectangle "test_parsing.py\n(Chunking)" as TestParsing
    rectangle "test_watchdog.py\n(File Watch)" as TestWatchdog
    rectangle "test_thread_safety.py\n(Concurrency)" as TestThread
    rectangle "test_config.py\n(Config)" as TestConfig
  }
  
  package "integration/" <<integration>> {
    rectangle "test_integration.py\n(Git Auto-Commit)" as TestIntegration
    rectangle "test_git_import.py\n(Git Module)" as TestGit
    rectangle "test_benchmarks.py\n(Benchmark Integration)" as TestBenchmarks
  }
  
  package "utilities/" <<utility>> {
    rectangle "test_mocks.py\n(Mock Manager)" as TestMocks
    rectangle "test_download_progress.py" as TestDownload
  }
  
  package "linting/" <<utility>> {
    rectangle "check_regression.py" as CheckRegression
    rectangle "check_syntax.py" as CheckSyntax
    rectangle "validate_syntax.py" as ValidateSyntax
  }
  
  package "benchmarks/" <<utility>> {
    rectangle "benchmark.py\n(Token Costs)" as BenchmarkTokens
    rectangle "benchmark_retrieval.py" as BenchmarkRetrieval
    rectangle "benchmark_claude_code.py" as BenchmarkClaudeCode
    rectangle "benchmark_cost.py" as BenchmarkCost
  }
  
  package "fixtures/" <<fixture>> {
    rectangle "retrieval_cases.json\n(Test Instances)" as RetrievalFixtures
    rectangle "*.py\n(Sample Code)" as SampleCode
  }
}

Conftest -[hidden]--> TestCore : provides fixtures
TestMocks -[hidden]--> TestCore : used by --test-mode
CheckRegression -[hidden]--> BenchmarkTokens : validates results
RetrievalFixtures -[hidden]--> BenchmarkRetrieval : test data

note top of TestCore
  6,600+ lines of unit tests
  Tests core functionality:
  - Query routing
  - Edit parsing
  - Chunk labeling
  - Cost calculation
  - History compression
end note

note right of TestMocks
  MockManager class provides:
  - Synthetic codebase generation
  - Mock Path class
  - Mock Anthropic API
  - Mock embeddings
  Used for --test-mode flag
end note

note bottom of CheckRegression
  Regression checks ensure:
  - Token costs don't increase >1%
  - Retrieval Hit@10 doesn't drop >2%
  - Retrieval Recall@10 doesn't drop >2%
end note

@enduml
```

</details>

---

## Test Execution Flow

The test execution follows a structured flow from discovery through regression checks.

![Test Execution Flow](plantuml/Test_Execution_Flow.png)

### Execution Phases

1. **Discovery** - pytest scans `tests/` directory for `test_*.py` files
2. **Fixture Setup** - Load `conftest.py`, set environment variables
3. **Test Execution** - Run unit and integration tests in parallel
4. **Test Mode** - Optional synthetic codebase testing via `--test-mode`
5. **Regression Checks** - Validate token costs and retrieval quality

### pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -p no:doctest
    --ignore=tests/fixtures
    --ignore=tests/benchmarks
    --ignore=tests/linting
```

### Excluded Directories

| Directory | Reason |
|-----------|--------|
| `tests/fixtures/` | Contains test data, not test code |
| `tests/benchmarks/` | Run separately as regression checks |
| `tests/linting/` | Contains check scripts, not pytest tests |

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Test_Execution_Flow
title Claude Light - Test Execution Flow

start

:pytest invoked;
:Read pytest.ini config;
:Scan tests/ directory;
:Collect test files;

note right
  Excludes:
  - tests/fixtures/
  - tests/benchmarks/
  - tests/linting/
end note

:Load conftest.py fixtures;
:Set ANTHROPIC_API_KEY env;

fork
  :Run unit tests;
fork again
  :Run integration tests;
end fork

if (Test mode enabled?) then (yes)
  :Load MockManager;
  :Generate synthetic codebase;
  :Patch Path, API, Embedder;
else (no)
  :Run standard tests;
endif

:Run benchmark.py;
:Generate token stats JSON;
:Run check_regression.py tokens;

if (Cost OK?) then (yes)
  :PASS;
else (no)
  :FAIL;
  stop
endif

:Run benchmark_retrieval.py;
:Run check_regression.py retrieval;

if (Retrieval OK?) then (yes)
  :PASS;
else (no)
  :FAIL;
  stop
endif

stop

@enduml
```

</details>

---

## Mock Architecture

The test suite includes a comprehensive mocking system for testing without API costs or external dependencies.

![Mock Architecture](plantuml/Test_Mock_Architecture.png)

### MockManager Components

| Component | Purpose |
|-----------|---------|
| `_mock_path_class` | Mocks file system operations |
| `_mock_create_message` | Mocks Anthropic API calls |
| `_mock_embedder_class` | Mocks sentence-transformers |
| `_mock_print_stats` | Mocks statistics output |

### Synthetic Codebase Presets

| Preset | Files | Methods | Approx. Tokens |
|--------|-------|---------|----------------|
| `small` | 5 | 10 | ~5,000 |
| `medium` | 50 | 15 | ~50,000 |
| `large` | 200 | 20 | ~200,000 |
| `extra-large` | 1,000 | 20 | ~1,000,000 |

### Usage

```bash
# Run with synthetic codebase
python -m claude_light --test-mode small "List all public classes"
python -m claude_light --test-mode medium "Find the authentication module"
python -m claude_light --test-mode large "Explain the request handling flow"
```

### Mocked Behaviors

**MockPath:**
- `rglob()` returns synthetic files
- `read_text()` returns generated code
- `exists()` returns True for test paths
- `mkdir()`/`write_text()` are no-ops

**MockAPI:**
- Returns MockUsage with token counts
- Simulates retrieved context
- Calculates savings percentage
- Never makes real API calls

**MockEmbedder:**
- Returns random numpy arrays
- Supports batch encoding
- Dimension based on model name
- No actual ML computation

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Test_Mock_Architecture
title Claude Light - Test Mock Architecture

skinparam backgroundColor #FEFEFE

package "Production Code" {
  [claude_light.llm] as LLM
  [claude_light.indexer] as Indexer
  [claude_light.skeleton] as Skeleton
  [claude_light.editor] as Editor
  [claude_light.config] as Config
}

package "Test Mocks (test_mocks.py)" {
  component "MockManager" as MockManager {
    component "_mock_path_class" as MockPath
    component "_mock_create_message" as MockAPI
    component "_mock_embedder_class" as MockEmbedder
    component "_mock_print_stats" as MockStats
  }
}

package "External Dependencies" {
  database "Anthropic API" as Anthropic
  component "SentenceTransformer" as Embedder
  component "Path (pathlib)" as Path
}

MockManager -[hidden]--> MockPath
MockManager -[hidden]--> MockAPI
MockManager -[hidden]--> MockEmbedder
MockManager -[hidden]--> MockStats

MockPath ..> LLM : patches Path
MockPath ..> Indexer : patches Path
MockPath ..> Skeleton : patches Path
MockPath ..> Editor : patches Path
MockPath ..> Config : patches Path

MockAPI ..> LLM : patches client.messages.create
MockEmbedder ..> Indexer : patches SentenceTransformer
MockStats ..> LLM : patches print_stats

note right of MockPath
  MockPath behavior:
  - rglob() returns synthetic files
  - read_text() returns generated code
  - exists() returns True for test paths
  - mkdir/write_text are no-ops
end note

note right of MockAPI
  MockAPI behavior:
  - Returns MockUsage with token counts
  - Simulates retrieved context
  - Calculates savings percentage
  - Never makes real API calls
end note

note right of MockEmbedder
  MockEmbedder behavior:
  - Returns random numpy arrays
  - Supports batch encoding
  - Dimension based on model name
  - No actual ML computation
end note

rectangle "Synthetic Codebase" as Synthetic {
  rectangle "Service0.java\n(20 methods)" as S0
  rectangle "Service1.java\n(20 methods)" as S1
  rectangle "...\n..." as Dots
  rectangle "Service999.java\n(20 methods)" as S999
}

MockManager --> Synthetic : generates based on preset

rectangle "Presets" as Presets {
  rectangle "small: 5 files, 10 methods" as Small
  rectangle "medium: 50 files, 15 methods" as Medium
  rectangle "large: 200 files, 20 methods" as Large
  rectangle "extra-large: 1000 files, 20 methods" as XL
}

Synthetic -[hidden]--> Presets

note bottom
  Usage: python -m claude_light --test-mode small "query"
  
  The MockManager intercepts all file I/O and API calls,
  allowing full integration testing without:
  - Real API costs
  - Network dependencies
  - File system modifications
end note

@enduml
```

</details>

---

## CI/CD Pipeline

GitHub Actions runs the full test suite on every push and pull request.

![CI/CD Pipeline](plantuml/CI_CD_Pipeline.png)

### Pipeline Stages

| Stage | Command | Purpose |
|-------|---------|---------|
| Setup | `actions/setup-python@v5` | Set up Python 3.11 |
| Install | `pip install ...` | Install all dependencies |
| Unit Tests | `pytest -q` | Run 200+ unit/integration tests |
| Token Regression | `benchmark.py` + `check_regression.py` | Validate token costs |
| Retrieval Regression | `benchmark_retrieval.py` + `check_regression.py` | Validate retrieval quality |

### Regression Thresholds

| Metric | Threshold | Failure Condition |
|--------|-----------|-------------------|
| Token Costs | Max 1% increase | `current > baseline * 1.01` |
| Hit@10 | Max 2% decrease | `current < baseline - 0.02` |
| Recall@10 | Max 2% decrease | `current < baseline - 0.02` |

### Baseline Files

| File | Content | Update Method |
|------|---------|---------------|
| `tests/baseline_token_stats.json` | Token cost benchmarks | Run `benchmark.py --json` |
| `tests/baseline_retrieval_stats.json` | Retrieval quality metrics | Run `benchmark_retrieval.py` |

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install pytest sentence-transformers tree-sitter-* ...
    - name: Run unit tests
      run: python -m pytest -q
    - name: Token regression check
      run: |
        python tests/benchmark.py --json > tests/baseline_token_stats_new.json
        python tests/check_regression.py tokens tests/baseline_token_stats.json tests/baseline_token_stats_new.json
    - name: Retrieval regression check
      run: |
        python tests/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/baseline_retrieval_stats_new.json
        python tests/check_regression.py retrieval tests/baseline_retrieval_stats.json tests/baseline_retrieval_stats_new.json
```

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml CI_CD_Pipeline
title Claude Light - CI/CD Pipeline

skinparam backgroundColor #FEFEFE

start

:GitHub Event push or PR;
:GitHub Actions starts runner;
:Set up Python 3.11;
:Install dependencies;

note right
  Dependencies:
  - pytest
  - sentence-transformers
  - tree-sitter
  - anthropic
  - watchdog
end note

:Run pytest -q;
:Execute 200+ tests;

if (Tests Pass?) then (yes)
  :Run benchmark.py;
  :Generate token stats JSON;
  :Run check_regression.py tokens;
  
  if (Token Cost OK?) then (yes)
    :PASS Token check;
  else (no)
    :FAIL Token regression;
    stop
  endif
  
  :Run benchmark_retrieval.py;
  :Generate retrieval stats JSON;
  :Run check_regression.py retrieval;
  
  if (Retrieval OK?) then (yes)
    :PASS Retrieval check;
  else (no)
    :FAIL Retrieval regression;
    stop
  endif
  
  :All checks passed;
  :Mark GitHub check passed;
else (no)
  :Tests failed;
  :Mark GitHub check failed;
  stop
endif

stop

@enduml
```

</details>

---

## Test Categories

### Unit Tests

Unit tests verify individual module functionality in isolation.

**Coverage:**
- Query routing logic
- Edit block parsing
- Chunk labeling
- Cost calculation
- History compression
- Skeleton building
- Linting functions
- Retry backoff
- UI formatting

**Example Test:**
```python
def test_route_query():
    # Low effort (simple lookups)
    model, effort, tokens = route_query("list all files")
    assert effort == "low"
    
    # Max effort (complex architectural reasoning)
    model, effort, tokens = route_query(
        "evaluate the scalability trade-offs deeply"
    )
    assert effort == "max"
```

### Integration Tests

Integration tests verify component interactions.

**Coverage:**
- Git auto-commit feature
- Git module imports
- Benchmark integration

**Example Test:**
```python
# test_integration.py - Git Auto-Commit
functions = [
    ('is_git_repo', callable),
    ('get_git_root', callable),
    ('auto_commit', callable),
    ('undo_last_commit', callable),
]

for func_name, _ in functions:
    assert hasattr(git_manager, func_name)
    assert callable(getattr(git_manager, func_name))
```

### Regression Tests

Regression tests ensure performance doesn't degrade.

**Token Regression:**
- Analytical cost model
- Compares warm session costs
- Validates RAG savings claims

**Retrieval Regression:**
- Uses SWE-bench fixtures
- Measures Hit@K and Recall@K
- Validates retrieval quality

---

## Running Tests

### Basic Test Run

```bash
# Run all tests
python -m pytest -q

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/unit/test_llm.py -v

# Run specific test class
python -m pytest tests/unit/test_llm.py::TestRouteQuery -v

# Run specific test function
python -m pytest tests/unit/test_llm.py::TestRouteQuery::test_low_effort_simple_lookup -v
```

### Test Mode (Synthetic Codebase)

```bash
# Small synthetic codebase
python -m claude_light --test-mode small "List all public classes"

# Medium synthetic codebase
python -m claude_light --test-mode medium "Find authentication module"

# Large synthetic codebase
python -m claude_light --test-mode large "Explain request handling"

# Extra-large synthetic codebase
python -m claude_light --test-mode extra-large "Analyze architecture"
```

### Benchmark and Regression

```bash
# Generate token stats
python tests/benchmark.py --json > tests/baseline_token_stats_new.json

# Check token regression
python tests/check_regression.py tokens tests/baseline_token_stats.json tests/baseline_token_stats_new.json

# Generate retrieval stats
python tests/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/baseline_retrieval_stats_new.json

# Check retrieval regression
python tests/check_regression.py retrieval tests/baseline_retrieval_stats.json tests/baseline_retrieval_stats_new.json
```

### Coverage Report

```bash
# Run with coverage
python -m pytest --cov=claude_light --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## Related Documentation

- [README.md](../README.md) - Getting started and usage
- [Architecture](architecture.md) - System architecture diagrams
- [CLAUDE.md](../CLAUDE.md) - Project guidelines
