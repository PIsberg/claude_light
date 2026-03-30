# Claude Light: Auto-Warmer & Hybrid RAG Chat CLI

`claude_light.py` is an interactive CLI chat tool for querying and editing a codebase using Claude. By combining Anthropic's prompt caching with a highly optimized hybrid RAG (Retrieval-Augmented Generation) pipeline, it drastically reduces API costs while maintaining full context of your project.

**License:** Free for personal and hobby use ([PolyForm Noncommercial](LICENSE)). Commercial use requires a license — contact [Peter Isberg](mailto:isberg.peter+cl@gmail.com).

![claude_light](https://github.com/user-attachments/assets/15dcb11d-c68a-4991-a5d8-adb1679e4e3b)


## 📊 Real-World Benchmark Results

Measured against 4 popular Python open-source libraries, running the same 10 queries per repo with each tool (live API runs, 2026-03-23).

**Claude Code** = the Anthropic CLI (`claude --print`), each query in a fresh isolated session using its built-in tool-calling to fetch files on demand.
**claude_light** = this project — hybrid RAG + prompt caching with automatic model routing.

### Cost per 10 queries — Claude Code vs claude_light

| Repository | Repo size | Claude Code | claude_light | Savings |
| :--- | ---: | ---: | ---: | ---: |
| `psf/requests` v2.31.0 | 108K tokens | $0.91 | $0.23 | **75%** |
| `pallets/flask` 3.0.0 | 144K tokens | $1.62 | $0.17 | **89%** |
| `encode/httpx` 0.25.2 | 191K tokens | $1.75 | $0.45 | **74%** |
| `bottlepy/bottle` 0.13.2 | 92K tokens | $1.52 | $0.30 | **80%** |
| **Total (≈40 queries)** | | **$5.81** | **$1.16** | **80% / 5× cheaper** |

### Why Claude Code costs more

Claude Code is a powerful general-purpose agentic tool and the cost difference reflects that:

- Its built-in system prompt and tool definitions consume ~35–40K tokens per session (cached, but cache-reads still cost $0.30/M).
- It always uses Sonnet, even for simple lookups where Haiku would suffice.
- Multi-turn tool calls (2–14 turns per query) accumulate cached context with each round-trip.
- It rediscovers the codebase via tools on every session — no pre-indexing.

claude_light makes different trade-offs: pre-indexes the codebase offline, routes simple queries to Haiku, and injects only ~2–3K targeted tokens per query.

> **How to reproduce:** `python benchmark_claude_code.py` — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full methodology.

---

## 📉 How It Optimizes Token Usage

This tool is aggressively designed to prevent you from paying full price for tokens. It tackles context bloat and cache expiration through several core strategies:

*   **Hybrid RAG + Prompt Caching:** The tool uses a two-tier caching strategy. The project's directory tree source files and all `.md` files act as a cached "skeleton" system prompt.
*   **Three-Tier Prompt Caching:** The tool places three `ephemeral` cache breakpoints in every request: (1) after the skeleton system prompt — stable for the whole session; (2) after the conversation history — stable for all but the newest turn; (3) after the retrieved RAG chunks in the current user message — stable across consecutive questions about the same code area. If you ask follow-up questions about the same module, the chunk block hits the cache and only your new question text is billed at full price ($3.00/M); everything above it costs $0.30/M.
*   **The "Heartbeat" Auto-Warmer:** A background daemon checks the session every 30 seconds. If you are idle for more than 4 minutes, it sends a tiny background ping to keep your ephemeral cache alive.
*   **Method-Level Chunking:** Instead of stuffing entire 1,500-line files into the context window, a brace-depth scanner splits source files into one chunk per method or constructor. Each chunk retains its package, imports, and class header to remain self-contained. This provides a massive 5–10× reduction in retrieved tokens.
*   **Sliding Window History:** To prevent conversation history from growing unboundedly and costing you on every turn, the tool caps memory at the last 6 turns via the `MAX_HISTORY_TURNS` setting.
*   **Strict Scoring Threshold:** The script drops irrelevant chunks below a cosine similarity score of 0.45 (`MIN_SCORE`) before sending them to Claude.
*   **Compressed Skeleton Tree:** The directory tree sent in the cached system prompt is compacted in two ways: single-child directory chains are collapsed (`main/java/com/example/` on one line), and sibling files sharing an extension are brace-grouped (`{OrderService,UserService,PaymentService}.java`). This saves 30–50 % of skeleton tokens on typical Java/Go/Python projects.
*   **Retrieved-Chunk Deduplication:** When multiple methods from the same file rank highly, the shared preamble (package, imports, class header) is emitted only once, with all retrieved methods listed underneath. This saves 5–20 % of retrieved-context tokens on class-heavy queries.

For a deeper dive into the implementation, see [docs/architecture.md](docs/architecture.md).

## 🚀 Getting Started

`claude_light` supports two ways to authenticate. If you are a Claude Pro subscriber, it will **automatically detect your CLI session**. For the best experience on Windows, we recommend a **one-time setup** of a long-lived **Automation Token** (`claude setup-token`) which provides a permanent, headless-friendly connection.

> **Note:** `sentence-transformers` pulls in PyTorch, which is approximately **1.5 GB** on the first install.

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Authenticate (Pick One)**:
   *   **Claude Pro**: Run `claude auth login` OR `claude setup-token` (recommended for Windows persistence).
   *   **API Key**: Set `export ANTHROPIC_API_KEY=sk-ant-...` in your shell.
3. **Run**:
   ```bash
   python claude_light.py
   ```

---

### macOS (Automated)

```bash
# Clone or download the repo, then:
bash install_macos.sh
```

The macOS installer handles the details specific to macOS:
- Detects Apple Silicon (`/opt/homebrew`) vs Intel (`/usr/local`) Homebrew paths
- Installs Python via Homebrew if no 3.9+ interpreter is found
- Creates a `.venv` virtual environment to avoid the PEP 668 "externally managed" error on macOS 13+
- Generates a `run.sh` wrapper so you don't need to activate the venv manually
- Reminds you to add the key to `~/.zshrc` (macOS default shell)

```bash
# Set API key (zsh — macOS default since Catalina)
echo 'export ANTHROPIC_API_KEY=sk-ant-your-key-here' >> ~/.zshrc && source ~/.zshrc

# Run from your project root
cd /path/to/your/project
/path/to/claude_light/run.sh
```

### Linux

```bash
# Clone or download the repo, then:
bash install.sh
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

The script detects your Python 3.9+ interpreter and installs all required and optional packages via `pip`.

### Windows (PowerShell)

```powershell
# Allow scripts for this session, then run the installer:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass;
.\install.ps1
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-ant-your-key-here","User")
```

### Manual installation (Verified Versions)

If you prefer to install packages yourself, use the [requirements.txt](requirements.txt) file which contains verified, secure versions:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the core packages manually:

```bash
# Required
pip install "sentence-transformers>=5.3.0" "numpy>=2.4.3" "watchdog>=6.0.0" "anthropic>=0.86.0" "prompt_toolkit>=3.0.52"

# Optional — strongly recommended
pip install "tree-sitter>=0.25.2" "rich>=14.3.3" "einops>=0.8.2"
```

Without tree-sitter, chunking falls back to whole-file mode. Without `rich`, output degrades gracefully to plain text formatting.

### 🔑 API key

### 🔑 Authentication

`claude_light` is designed to be flexible with how you pay for tokens. It resolves your identity in this order:

1.  **Environment Variable**: `ANTHROPIC_API_KEY` (Standard API Key)
2.  **Local Dotfiles**: `.anthropic` (home dir) or `.env` (project dir)
3.  **Claude CLI Config (OAuth)**: Automatically reads your Claude Pro session from `~/.claude/.credentials.json`.

#### Option A: Claude Pro Subscription (OAuth)
If you have a **Claude Pro** or **Team** subscription and have authorized the official [Claude CLI](https://github.com/anthropics/claude-code), `claude_light` will automatically use your flat-rate subscription. 
*   **Cost**: Included in your $20/mo subscription.
*   **Setup**: Just run `claude auth login` once.

#### Option B: Anthropic API Key
If you prefer to pay per-token or don't have a Pro subscription:
*   **Cost**: Usage-based (Pay-as-you-go via the Anthropic Console).
*   **Setup**: Set the environment variable:
    ```bash
    # Linux / macOS
    export ANTHROPIC_API_KEY=sk-ant-your-key-here

    # Windows PowerShell (persistent)
    [System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-ant-your-key-here","User")
    ```

Run the script from the root of your project — it will immediately build the skeleton, chunk your files, and auto-tune the embedding model.

---

## 🚀 Usage

### 1. Interactive Loop (Default)
`python3 claude_light.py`

The script runs three concurrent threads to keep your workflow seamless:

* **Main Thread:** Handles the input loop, retrieves relevant chunks per query, and manages the conversation history.
* **Watchdog:** Monitors your directory for file saves. If a source file changes, it automatically re-chunks and re-embeds it; if a `.md` file changes, it rebuilds the skeleton cache.
* **Heartbeat Daemon:** Keeps the Anthropic prompt cache warm while you step away.

### Auto-Tuned Embedding Models
The script automatically selects the most efficient embedding model based on the size of your repository:

* **< 50 files:** `all-MiniLM-L6-v2` (22 MB) for fast startup.
* **50–199 files:** `all-mpnet-base-v2` (420 MB) for better semantic depth.
* **200+ files:** `nomic-ai/nomic-embed-text-v1.5` for optimal recall on large codebases.

### 🛡️ Security & SBOM

To ensure supply chain transparency and prevent dependency confusion, `claude_light` uses:

*   **Pinned Versions:** All dependencies in `requirements.txt` and the installer scripts are pinned to verified, secure minimum versions.
*   **SBOM (Software Bill of Materials):** The project maintains an industry-standard SBOM in [CycloneDX](https://cyclonedx.org/) format.

#### Continuous Inventory
A GitHub Action automatically generates a fresh `sbom.json` on every push to `main` and uploads it as a build artifact.

#### Manual Generation
You can generate the SBOM manually using the provided scripts:

```bash
# Linux / macOS
bash scripts/generate_sbom.sh

# Windows
.\scripts\generate_sbom.ps1
```


### Tests

Run the full unit test suite:

```bash
python -m pytest -q
```

Run a quick CLI smoke test with the synthetic mocked codebase:

```bash
python -m claude_light --test-mode small "List all public classes or modules in this codebase"
```

Larger synthetic presets are also available for local stress testing:

```bash
python -m claude_light --test-mode medium
python -m claude_light --test-mode large
python -m claude_light --test-mode extra-large
```

### CI Regression Checks

The GitHub Actions workflow currently runs three checks:

1. Unit tests

```bash
python -m pytest -q
```

2. Analytical token-cost regression

```bash
python tests/benchmark.py --json > tests/baseline_token_stats_new.json
python tests/check_regression.py tokens tests/baseline_token_stats.json tests/baseline_token_stats_new.json
```

3. Offline retrieval regression on the committed local fixture

```bash
python tests/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/baseline_retrieval_stats_new.json
python tests/check_regression.py retrieval tests/baseline_retrieval_stats.json tests/baseline_retrieval_stats_new.json
```

If retrieval behavior changes intentionally, refresh the baseline with the same fixture command so CI remains deterministic:

```bash
python tests/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/baseline_retrieval_stats.json
```

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/PIsberg/claude_light/badge)](https://scorecard.dev/viewer/?uri=github.com/PIsberg/claude_light/)