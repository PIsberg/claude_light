# Claude Light: Auto-Warmer & Hybrid RAG Chat CLI

`claude_light.py` is an interactive CLI chat tool for querying and editing a codebase using Claude. By combining Anthropic's prompt caching with a highly optimized hybrid RAG (Retrieval-Augmented Generation) pipeline, it drastically reduces API costs while maintaining full context of your project.

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

## 🚀 Getting Started

### Dependencies

Install the required Python packages and system tools:
```bash
pip install sentence-transformers numpy
pip install python3-watchdog python3-anthropic
pip install prompt_toolkit
pip install einops

```

# Running the Tool

> **Note:** `sentence-transformers` will pull in PyTorch, which is approximately **1.5 GB** on the first install.

Run the script from the root of your project. It will immediately build the skeleton, chunk your files, and auto-tune the embedding model.

### 🔑 Prerequisites
You must set your Anthropic API key before running:

`export ANTHROPIC_API_KEY=sk-ant-your-key-here`

---

## 🚀 Usage

### 1. Interactive Loop (Default)
`python3 claude_light.py`

### 2. One-Shot Mode
`python3 claude_light.py "What does OrderService do?"`

### 3. Piped Input
`echo "List all REST endpoints" | python3 claude_light.py`

---

## 🔀 Dynamic Model & Effort Routing

Before every query, a lightweight router classifies the prompt and selects the most cost-efficient model and effort level automatically:

| Effort | Model | Typical use case |
| :--- | :--- | :--- |
| `low` | Claude Haiku | Simple lookups, listings, "where is X", "how many Y" |
| `medium` | Claude Sonnet | Explanations, summaries, moderate analysis |
| `high` | Claude Sonnet | Code generation, multi-step changes, refactoring |
| `max` | Claude Opus (+ extended thinking) | Architecture, deep cross-cutting analysis, trade-offs |

The router prints its decision before each call:

```
[Router] effort=high  model=sonnet
```

Routing is heuristic-based (keyword signals + prompt length). The router is deliberately conservative — it defaults to `high/sonnet` when uncertain, favouring quality over micro-savings. `max/opus` is reserved for prompts that contain at least two strong architectural signals (e.g. "evaluate the scalability trade-offs of…").

---

## ⌨️ Interactive Commands
While in the interactive chat loop, you can use the following commands:

| Command | Description |
| :--- | :--- |
| `<prompt>` | Instructs Claude to return complete files with your requested changes. The script will automatically generate a colored diff and ask for your confirmation before writing the files to disk. |
| `/clear` | Resets the conversation history. |
| `/cost` | Shows the total session spend so far. |
| `/help` | Displays the help menu. |
| `exit` / `quit` | Exits the application. |

---

## 🧠 Architecture & Auto-Tuning
The script runs three concurrent threads to keep your workflow seamless:

* **Main Thread:** Handles the input loop, retrieves relevant chunks per query, and manages the conversation history.
* **Watchdog:** Monitors your directory for file saves. If a source file changes, it automatically re-chunks and re-embeds it; if a `.md` file changes, it rebuilds the skeleton cache.
* **Heartbeat Daemon:** Keeps the Anthropic prompt cache warm while you step away.

### Auto-Tuned Embedding Models
The script automatically selects the most efficient embedding model based on the size of your repository:

* **< 50 files:** `all-MiniLM-L6-v2` (22 MB) for fast startup.
* **50–199 files:** `all-mpnet-base-v2` (420 MB) for better semantic depth.
* **200+ files:** `nomic-ai/nomic-embed-text-v1.5` for optimal recall on large codebases..


### Tests
python claude_light.py --test-mode small

python claude_light.py --test-mode large