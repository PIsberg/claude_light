# Claude Light: Auto-Warmer & Hybrid RAG Chat CLI

`claude_light.py` is an interactive CLI chat tool for querying and editing a Java codebase using Claude. By combining Anthropic's prompt caching with a highly optimized hybrid RAG (Retrieval-Augmented Generation) pipeline, it drastically reduces API costs while maintaining full context of your project.

## 📉 How It Optimizes Token Usage

This tool is aggressively designed to prevent you from paying full price for tokens. It tackles context bloat and cache expiration through several core strategies:

*   **Hybrid RAG + Prompt Caching:** The tool uses a two-tier caching strategy. The project's directory tree and all `.md` files act as a cached "skeleton" system prompt.
*   **Double Caching for RAG:** The top-K retrieved Java source chunks are injected as a *second* cached system block. If you ask repeated questions about the same code, you pay the cache-read price ($0.30/M) instead of the full input price ($3.00/M).
*   **The "Heartbeat" Auto-Warmer:** A background daemon checks the session every 30 seconds. If you are idle for more than 4 minutes, it sends a tiny background ping to keep your ephemeral cache alive.
*   **Method-Level Chunking:** Instead of stuffing entire 1,500-line files into the context window, a brace-depth scanner splits `.java` files into one chunk per method or constructor. Each chunk retains its package, imports, and class header to remain self-contained. This provides a massive 5–10× reduction in retrieved tokens.
*   **Sliding Window History:** To prevent conversation history from growing unboundedly and costing you on every turn, the tool caps memory at the last 6 turns via the `MAX_HISTORY_TURNS` setting.
*   **Strict Scoring Threshold:** The