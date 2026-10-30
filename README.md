# Claude Light: Auto-Warmer & Hybrid RAG Chat CLI

`claude_light.py` is an interactive CLI chat tool for querying and editing a codebase using Claude. By combining Anthropic's prompt caching with a highly optimized hybrid RAG (Retrieval-Augmented Generation) pipeline, it drastically reduces API costs while maintaining full context of your project.

**License:** Free for personal and hobby use ([PolyForm Noncommercial](LICENSE)). Commercial use requires a license — contact [Peter Isberg](mailto:isberg.peter+cl@gmail.com).

![claude_light](https://github.com/user-attachments/assets/15dcb11d-c68a-4991-a5d8-adb1679e4e3b)

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/PIsberg/claude_light/badge)](https://scorecard.dev/viewer/?uri=github.com/PIsberg/claude_light/)


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

> **How to reproduce:** `python tests/benchmarks/benchmark_cost.py --json > cl_results.json`, then `python tests/benchmarks/benchmark_claude_code.py --claude-light-results cl_results.json`. Both make live API calls, and the second needs Claude Code installed. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) section 4 for the methodology.

---

## 📉 How It Optimizes Token Usage

This tool is aggressively designed to prevent you from paying full price for tokens. It tackles context bloat and cache expiration through several core strategies:

*   **Hybrid RAG + Prompt Caching:** The tool uses a two-tier caching strategy. The project's directory tree source files and all `.md` files act as a cached "skeleton" system prompt.
*   **Three-Tier Prompt Caching:** The tool places three `ephemeral` cache breakpoints in every request: (1) after the skeleton system prompt — stable for the whole session; (2) after the conversation history — stable for all but the newest turn; (3) after the retrieved RAG chunks in the current user message — stable across consecutive questions about the same code area. If you ask follow-up questions about the same module, the chunk block hits the cache and only your new question text is billed at full price ($2.00/M on Sonnet 5.5); everything above it costs the cache-read rate ($0.10/M).
*   **The "Heartbeat" Auto-Warmer:** In API-key mode, a background daemon checks the session every 30 seconds and sends a tiny ping if you are idle for more than 4 minutes, keeping your ephemeral cache alive. The skeleton cache uses a 1-hour extended TTL so this rarely needs to fire. (OAuth/Pro-subscription mode skips the heartbeat: each turn already spawns a fresh `claude` CLI subprocess, so there is no persistent process to warm — the Anthropic backend's content-hashed prompt cache still hits across subprocess invocations on its own.)
*   **Method-Level Chunking:** Instead of stuffing entire 1,500-line files into the context window, a brace-depth scanner splits source files into one chunk per method or constructor. Each chunk retains its package, imports, and class header to remain self-contained. This provides a massive 5–10× reduction in retrieved tokens.
*   **Sliding Window History:** To prevent conversation history from growing unboundedly and costing you on every turn, the tool caps memory at the last 6 turns via the `MAX_HISTORY_TURNS` setting.
*   **Strict Scoring Threshold:** Two filters drop low-quality chunks before sending them to Claude. `MIN_SCORE` (0.35) is an absolute cosine-similarity floor (natural-language questions score lower against code than keyword queries), and `RELATIVE_SCORE_FLOOR` (0.60) drops any chunk whose score is below 60% of the top-ranked chunk for that query. Together they prevent a single weak query from dragging in noise alongside one strong hit.
*   **Compressed Skeleton Tree:** The directory tree sent in the cached system prompt is compacted in two ways: single-child directory chains are collapsed (`main/java/com/example/` on one line), and sibling files sharing an extension are brace-grouped (`{OrderService,UserService,PaymentService}.java`). This saves 30–50 % of skeleton tokens on typical Java/Go/Python projects.
*   **Retrieved-Chunk Deduplication:** When multiple methods from the same file rank highly, the shared preamble (package, imports, class header) is emitted only once, with all retrieved methods listed underneath. This saves 5–20 % of retrieved-context tokens on class-heavy queries.

For a deeper dive into the implementation, see [docs/architecture/](docs/architecture/ARCHITECTURE.md).

## 🚀 Getting Started

`claude_light` supports two ways to authenticate. If you are a Claude Pro subscriber, it will **automatically detect your CLI session**. For the best experience on Windows, we recommend a **one-time setup** of a long-lived **Automation Token** (`claude setup-token`) which provides a permanent, headless-friendly connection.

> **Note:** `sentence-transformers` pulls in PyTorch, which is approximately **1.5 GB** on the first install.

### Install as a tool (recommended)

If you have [`pipx`](https://pipx.pypa.io/) or [`uv`](https://docs.astral.sh/uv/) installed, the fastest path is to install `claude_light` as a globally-available CLI in its own isolated environment:

```bash
# pipx
pipx install git+https://github.com/PIsberg/claude_light.git

# uv
uv tool install git+https://github.com/PIsberg/claude_light.git
```

Then run from any project root:

```bash
claude-light                 # interactive
claude-light "what does OrderService do?"   # one-shot
```

Optional extras:

```bash
pipx install "git+https://github.com/PIsberg/claude_light.git#egg=claude-light[llmlingua]"
```

LLMLingua compresses retrieved context only when it is at least `CLAUDE_LIGHT_LLMLINGUA_MIN_TOKENS` tokens long (default `800`). The retrieval budgets are 1 500 to 9 000 tokens, so a full retrieval always qualifies; lower the value only if small retrievals should be compressed too. The first compression loads a model (about 280 MB) on CPU, which takes around 10 seconds.

A PyPI release (so the `git+...` URL becomes plain `pipx install claude-light`) is planned — the existing methods below remain fully supported in the meantime.

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
bash installers/install_macos.sh
```

The macOS installer handles the details specific to macOS:
- Detects Apple Silicon (`/opt/homebrew`) vs Intel (`/usr/local`) Homebrew paths
- Installs Python 3.12 via Homebrew if no 3.12+ interpreter is found
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
bash installers/install.sh
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

The script finds a Python 3.12+ interpreter and installs everything in [requirements.txt](requirements.txt) via `pip`.

### Windows (PowerShell)

```powershell
# Allow scripts for this session, then run the installer:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass;
.\installers\install.ps1
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-ant-your-key-here","User")
```

### Manual installation

Python 3.12 or newer. Install everything in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

Without tree-sitter, chunking falls back to whole-file mode. Without `rich`, output degrades gracefully to plain text formatting.

### 🔑 API key

### 🔑 Authentication

`claude_light` is designed to be flexible with how you pay for tokens. It resolves your identity in this order:

1.  **Environment Variable**: `ANTHROPIC_API_KEY` (Standard API Key)
2.  **Local Dotfiles**: `.anthropic` (home dir) or `.env` (project dir)
3.  **Claude CLI Config (OAuth)**: Automatically reads your Claude Pro session from `~/.claude/.credentials.json`.

#### Other providers: Gemini and Codex
`python3 claude_light.py --provider gemini` or `--provider codex` sends the same skeleton and retrieved chunks to that tool's CLI instead of Claude. Without `--provider` nothing changes.
*   **Gemini**: install `gemini` (`npm install -g @google/gemini-cli`) and sign in, or set `GEMINI_API_KEY`.
*   **Codex**: install `codex` (`npm install -g @openai/codex`) and run `codex login`.
*   **Unavailable there**: prompt caching, effort routing, the top-tier prompt, `/budget`, token and cost figures, and shell access. Each says so when you use it. Each tool runs in an empty temporary directory, so it answers from what claude_light sends and does not read or run anything in your repo.
*   **Sessions**: saved per provider; `--continue` with another provider's file starts fresh and names the provider to resume.

#### Option A: Claude Pro Subscription (OAuth)
If you have a **Claude Pro** or **Team** subscription and have authorized the official [Claude CLI](https://github.com/anthropics/claude-code), `claude_light` will automatically use your flat-rate subscription. 
*   **Cost**: Included in your $20/mo subscription.
*   **Setup**: Just run `claude auth login` once.
*   **Instructions it sees**: the repo's own `CLAUDE.md` files are loaded; your personal `~/.claude/CLAUDE.md` is not, even when the repo lives under your home directory, so instructions written for your own Claude Code setup do not leak into claude_light's agent.
*   **Running commands**: the agent edits files without asking but runs no shell commands by default. To let it run some, such as your tests so it can check its own changes, allow command prefixes (read-only commands like `git status` then work too): `claude_light.py --allow-command "python -m pytest"` (repeatable) or `CLAUDE_LIGHT_ALLOW_COMMANDS="python -m pytest; npm test"`. A PowerShell command takes a `powershell:` tag, e.g. `--allow-command "powershell: Invoke-Pester"`, and runs under the PowerShell tool; untagged entries (and `bash:`) run under Bash. An entry allows any command that starts with it. A command that chains, pipes, redirects or substitutes (any of `;`, `&`, `|`, `<`, `>`, a backtick, `$` or a newline) is refused, so `npm test && rm -rf x` does not run the `rm`. PowerShell commands also refuse `(`, `)`, `{`, `}` and `@`, because PowerShell evaluates an argument such as `(New-Item x)`; the agent runs one simple command per call. With an allowlist the agent edits files with its own tools so it can run the tests on its change before it finishes; claude_light then shows the diff and commits it (`/undo` reverts). Without one, edits come back as SEARCH/REPLACE blocks that claude_light lint-checks, previews and applies. Anything else is refused, and claude_light lists the refused commands after the turn. Permission rules, hooks and MCP servers configured inside the repo you are working on are ignored, so a cloned repo cannot grant itself commands.
*   **Long tasks**: each turn runs the CLI's own agent loop (it can read and edit files itself), so a multi-file change can take several minutes. Each file it reads, searches or edits is shown as it happens (`⏺ Read claude_light/llm.py`). Press Ctrl+C to cancel a turn: the agent process is stopped and you are back at the prompt (files it already changed stay in the working tree, uncommitted). The turn is also cancelled if the CLI produces no output for 5 minutes, or after 1 hour in total. On the very first run the index is still being built (several minutes on a CPU); you can ask questions immediately and they go to the agent without retrieved context until the index is ready. One-shot runs (`claude_light.py "question"`) do the same: the answer comes first, then the index finishes building before the process exits, so the next run has it.

#### Option B: Anthropic API Key
If you prefer to pay per-token or don't have a Pro subscription:
*   **Cost**: Usage-based (Pay-as-you-go via the Anthropic Console).
*   **Files beyond the retrieved context**: the model can call three read-only tools, `read_file`, `grep` and `list_files`, confined to the project directory, when retrieval missed the code it needs (up to 8 rounds per turn). It cannot run commands; edits still come back as SEARCH/REPLACE blocks that claude_light lint-checks, previews and commits.
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

The prompt appears in about half a second: the index builds in the background, and the toolbar under the prompt says `indexing…` until it is ready, then shows the session's tokens, cache hit rate and cost. Replies stream in as rendered markdown (tables, lists, syntax-highlighted code), with a spinner and elapsed time while Claude thinks or runs a tool, and one line per file it reads, searches or edits. In the reply each proposed edit is a single `✎ edit to path` line; the change itself is shown as a diff, with a menu underneath to approve them (arrow keys and Enter, or press `1`/`2`/`3`; Esc says no). Keys typed before the menu appears are ignored, so a stray Enter cannot answer it. "Yes, and don't ask again" stops that one question for the rest of the session (applying edits and committing the agent's edits are remembered separately); `-y` approves both from the start.

`python3 claude_light.py --continue` (or `-c`) picks up this project's previous conversation where you left it, as `claude --continue` does. The conversation is saved after every turn to `~/.claude_light_sessions/` (one file per project, outside the repo so a cloned repo cannot supply one), along with the session's model. Cost counters start from zero. Retrieval runs fresh for each new question, but earlier answers may describe code that has changed since. `/clear` deletes the saved conversation. `--continue` works only for the interactive loop: a one-shot question sends no history, so combining the two is an error.

Each question is routed to Haiku 5.5, Sonnet 5.5 or Opus 5.5, and the session keeps the first question's model (`CLAUDE_LIGHT_ROUTE_ONCE=0` routes every question). A question that routes to Opus asks first: Opus 5.5, Fable 5.1 or Sonnet 5.5, each with its estimated cost for the turn. Esc, or running without a terminal, picks Sonnet 5.5.

| Command / key | What it does |
|---|---|
| `/help` | Commands and keys |
| `/clear` | Start a new conversation and delete the one saved for `--continue`; the next question also picks the model again, which the session then keeps (`CLAUDE_LIGHT_ROUTE_ONCE=0` picks it per question) |
| `/compact` | Replace the conversation so far with a summary, so later turns send fewer tokens |
| `/cost` | Token usage and cost for this session, and lifetime savings |
| `/run <cmd>` | Run a shell command and ask Claude about its output |
| `/undo` | Undo the last `Claude:` commit. Keeps uncommitted work; refuses if the last commit is yours or its files have local edits |
| `/why` | How the last query was routed and what was retrieved |
| `exit` / `quit` | Quit; prints a one-line session summary |
| Typing `/` | Menu of commands with descriptions |
| `alt+enter` | New line in the prompt (`enter` sends) |
| `ctrl+c` | Cancel the running turn; at the prompt, quit |

One-shot runs write only the answer to stdout, as plain text; progress and tool lines go to stderr, so `claude_light.py "question" > answer.md` holds just the answer.

The script runs three concurrent threads to keep your workflow seamless:

* **Main Thread:** Handles the input loop, retrieves relevant chunks per query, and manages the conversation history.
* **Watchdog:** Monitors your directory for file saves. If a source file changes, it automatically re-chunks and re-embeds it; if a `.md` file changes, it rebuilds the skeleton cache. Files under `.git`, `.venv`, `node_modules` and other skipped directories are ignored. It reports in the toolbar (`re-indexed calc.py (2 chunks)`), never in the middle of a reply.
* **Heartbeat Daemon (API-key mode only):** Keeps the Anthropic prompt cache warm while you step away. Disabled in OAuth/Pro-subscription mode, where each turn spawns a fresh `claude` CLI subprocess and there is no persistent process to keep warm.

### Embedding Models (hybrid retrieval)
Code is indexed twice, by two models that are good at different questions:

* **`static-retrieval-mrl-en-v1`** (static embeddings, no transformer): indexes this repo's 1,400 chunks in about 2 s on a CPU, so retrieval works from the first question. Best of six models measured at questions that name code ("where is `retry_with_backoff` used?").
* **`all-mpnet-base-v2`**: about 380 s for the same chunks, so it runs in the background while you work, and its vectors are cached. Best at questions that describe behaviour without naming it ("where do we wait longer between attempts?").

Once both are ready, each chunk's similarity to the question is z-scored per model and averaged. On 35 questions about this repo, the file that answers the question reached the model's context for 23, against 15 with mpnet alone (the setup before this change), and the 6 off-topic questions still got no context. Measure on your own changes with `python tests/benchmarks/benchmark_models.py`. `CLAUDE_LIGHT_HYBRID=0` switches back to a single model picked by repository size (`all-MiniLM-L6-v2` under 50 files, `all-mpnet-base-v2` under 200, `nomic-ai/nomic-embed-text-v1.5` above).

### 🛡️ Security & SBOM

To ensure supply chain transparency and prevent dependency confusion, `claude_light` uses:

*   **Version floors:** `requirements.txt` and `pyproject.toml` set each dependency's minimum to the newest release that installs on Python 3.12. The installers read `requirements.txt`, so there is one list to keep current. The `Installers` workflow runs all three on clean Linux, macOS and Windows runners whenever an installer or `requirements.txt` changes, and weekly.
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

The GitHub Actions workflow (`.github/workflows/test.yml`, Python 3.12) runs three steps:

1. Unit tests

```bash
python -m pytest -q
```

2. Analytical token-cost regression

```bash
python tests/benchmarks/benchmark.py --json > tests/baseline_token_stats_new.json
python tests/linting/check_regression.py tokens tests/fixtures/baseline_token_stats.json tests/baseline_token_stats_new.json
```

3. Offline retrieval regression on the committed local fixture

```bash
python tests/benchmarks/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/baseline_retrieval_stats_new.json
python tests/linting/check_regression.py retrieval tests/fixtures/baseline_retrieval_stats.json tests/baseline_retrieval_stats_new.json
```

If retrieval behavior changes intentionally, refresh the baseline with the same fixture command so CI remains deterministic:

```bash
python tests/benchmarks/benchmark_retrieval.py --fixture tests/fixtures/retrieval_cases.json --output tests/fixtures/baseline_retrieval_stats.json
```