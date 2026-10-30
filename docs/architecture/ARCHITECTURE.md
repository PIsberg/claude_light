---
description: Implementation walkthrough of claude_light, one topic per file. Read the file for the part you are changing.
---

# Architecture

claude_light is a CLI that caches a project skeleton in Claude's prompt cache and retrieves only the code chunks each question needs. Read the topic that matches your change:

- [components.md](components.md) — modules, authentication modes, providers (Claude, Gemini, Codex), external dependencies. Start here for "where does this live".
- [query-flow.md](query-flow.md) — what happens to one question, from input to rendered answer, and the background processes.
- [caching.md](caching.md) — the cache breakpoints, the skeleton and the retrieved block, and the per-model pricing table.
- [rag-pipeline.md](rag-pipeline.md) — retrieval, effort levels and per-effort token budgets.
- [routing.md](routing.md) — `route_query()`: how a question gets its model and effort tier.
- [compression.md](compression.md) — optional LLMLingua-2 compression of the retrieved block.
- [indexing.md](indexing.md) — chunking, the hybrid index, the disk cache and incremental re-indexing.

Diagrams are PlantUML renders in [../plantuml/](../plantuml/). The design note for compression is [../llmlingua_plan.md](../llmlingua_plan.md).
