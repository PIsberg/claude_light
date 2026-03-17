# Claude Light: Token Usage Architecture

This document outlines the primary mechanisms and architectural decisions implemented in `claude_light.py` to massively reduce LLM token consumption. The strategies are ordered by their relative significance and impact on token cost.

---

## 1. RAG Method-Level Chunking & Retrieval
**Significance:** Absolute highest impact. Reduces the baseline context payload from "entire source system" to "only relevant methods".

### Mechanism
Rather than feeding Claude an entire codebase for every question, `claude_light` uses **Tree-sitter** to parse ASTs of source files. It splits the code strictly into method-level chunks while retaining the file's imports and class headers. When the user asks a question, only the highest-scoring chunks mathematically similar to the query are retrieved and injected as context.

### Estimated Token Savings
- **90% – 99%+ per query**
- **Example:** In an "extra-large" project of 1,000 files yielding ~900,000 tokens, injecting the top 15 relevant functional chunks takes the query context down to ~1,300 tokens. This alone prevents the query cost from exceeding $2.75 *per turn*, dropping it to $0.0004.

---

## 2. Dynamic Prefix Caching Re-Architecture
**Significance:** Extreme compounding impact over long, multi-turn conversations.

### Mechanism
Anthropic's Prompt Caching reads tokens sequentially from the start of the `messages` chain and breaks the cache the moment it hits a modified sequence. Previously, `claude_light` appended the `retrieved_ctx` (the RAG output) into the `system` block prefix. Because `retrieved_ctx` inevitably changes based on the user's current question, everything following it in the array—including the entire rolling conversation history—failed to cache. 

The architecture remedies this by loading the `skeleton` (the unchanged directory tree and core system rules) into the `system` block, and prepending the volatile `retrieved_ctx` directly into the *current* user query at the very end of the `messages` array. An `ephemeral` cache breakpoint is injected explicitly at the end of the `skeleton` and at the end of the `conversation_history`.

### Estimated Token Savings
- **50% - 95% discount per turn for historical conversation tokens**
- **Example:** Re-sending 10,000 tokens of conversational history previously cost full input pricing ($3.00 / 1M). With a stabilized cache prefix hitting 100% of the conversational history and skeleton, those tokens are billed at the cached rate ($0.30 / 1M)—a 90% cost drop.

---

## 3. Scrubbing Assistant generated Code-Blocks from History
**Significance:** Critical for any session employing `/edit` commands.

### Mechanism
When the user executes an `/edit` instruction, the assistant replies with full-file code blocks (e.g. 500 lines of Java) for the script to intercept and write to disk. Natively, appending that raw assistant reply into `conversation_history` means that 500 lines of source code become permanently tacked onto the system memory, artificially bloating every subsequent turn.

The architecture employs a regex pattern to strip fenced ````[lang]:path/to/file```` code blocks out of the assistant's reply entirely *before* committing it to `conversation_history`. The block is replaced with a lean `[File updated: path/to/file]` marker, preserving the conversational flow but jettisoning the heavy file payload.

### Estimated Token Savings
- **1,000 – 10,000+ tokens saved sequentially per edit turn**
- **Example:** Editing a 2,000 token file 5 times would normally compound into passing 10,000 useless tokens to the LLM on the 6th turn. This prevents exponential bloat during refactoring loops.

---

## 4. Bounded Markdown Ingestion (`build_skeleton`)
**Significance:** Highly effective for repositories utilizing expansive documentation bases or node frameworks with giant dependencies.

### Mechanism
The `build_skeleton()` system function recursively explores the source tree, cataloging directories and instantly reading `.md` contents (like `README.md`) to provide Claude with high-level architectural constraints. However, giant vendor manuals, release changelogs, or sprawling internal documentation could easily exceed 50,000 tokens of uncontrolled noise appended to the `skeleton`.

The system implements a strict 5,000-character truncation ceiling for any `.md` file traversed during the skeleton build. If a `.md` string exceeds this limit, it is cleanly truncated with a `[TRUNCATED due to length]` tag to signify the limit. An explicit safety bypass is provisioned for `CLAUDE.md` and `AGENTS.md` (which hold the core operational rules of the AI agents and must never be trimmed).

### Estimated Token Savings
- **Variable (10,000 – 200,000+ tokens) depending strictly on repository hygiene**
- **Example:** A project pulling down an extensive React framework might have 100,000 tokens worth of `.md` documentation within un-skipped subdirectories. This bounds that potential leak instantly to ~5,000 chars (approx. 1,000 tokens max per file).
