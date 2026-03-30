# Claude Light Architecture

This document provides a visual overview of the Claude Light architecture using PlantUML diagrams. Claude Light is an interactive CLI chat tool that combines Anthropic's prompt caching with a hybrid RAG (Retrieval-Augmented Generation) pipeline to drastically reduce API costs while maintaining full context of your codebase.

## Table of Contents

1. [Component Architecture](#component-architecture)
2. [Query Processing Sequence](#query-processing-sequence)
3. [Three-Tier Caching Strategy](#three-tier-caching-strategy)
4. [RAG Pipeline](#rag-pipeline)
5. [File Indexing Flow](#file-indexing-flow)

---

## Component Architecture

This diagram shows the main components and their relationships within Claude Light.

![Component Diagram](plantuml/Claude_Light_Component_Diagram.png)

### Key Components

| Component | Responsibility |
|-----------|----------------|
| `main.py` | Entry point, orchestrates the chat loop and handles user commands |
| `llm.py` | LLM client, handles API calls, routing, and response processing |
| `retrieval.py` | RAG engine, performs similarity search and chunk selection |
| `editor.py` | Code editor, parses and applies SEARCH/REPLACE blocks |
| `indexer.py` | File watcher, manages incremental indexing and cache |
| `skeleton.py` | Builds compressed directory tree and markdown documentation |
| `state.py` | Shared state module for thread-safe access to session data |
| `config.py` | Configuration constants and API key resolution |

### External Dependencies

- **Anthropic API** - Claude models (Haiku, Sonnet, Opus)
- **sentence-transformers** - Text embeddings for RAG
- **tree-sitter** - AST parsing for method-level chunking
- **Git** - Version control for auto-commit feature

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Claude_Light_Component_Diagram
title Claude Light - Component Architecture

skinparam componentStyle rectangle
skinparam backgroundColor #FEFEFE
skinparam componentBackgroundColor #FFFFFF
skinparam arrowColor #555555

package "User Interface Layer" {
  [main.py\n(Entry Point)] as Main
  [ui.py\n(Terminal UI)] as UI
  [state.py\n(Shared State)] as State
}

package "Core Processing" {
  [llm.py\n(LLM Client)] as LLM
  [retrieval.py\n(RAG Engine)] as Retrieval
  [editor.py\n(Code Editor)] as Editor
  [executor.py\n(Command Runner)] as Executor
}

package "Indexing & Caching" {
  [indexer.py\n(File Watcher)] as Indexer
  [skeleton.py\n(Skeleton Builder)] as Skeleton
  [parsing.py\n(Code Parser)] as Parsing
  [config.py\n(Configuration)] as Config
}

package "External Services" {
  database "Anthropic API\n(Claude)" as Anthropic
  [sentence-transformers\n(Embeddings)] as Embeddings
  [tree-sitter\n(AST Parsing)] as TreeSitter
  [Git Repository] as Git
}

package "Support Modules" {
  [linter.py\n(Syntax Check)] as Linter
  [retry.py\n(Backoff)] as Retry
  [streaming.py\n(Stream Handler)] as Streaming
  [git_manager.py\n(Git Ops)] as GitManager
}

' User interactions
Main --> UI : displays status
Main --> State : reads/writes session data

' Main orchestration
Main --> LLM : chat/one_shot
Main --> Indexer : file watch events

' LLM processing flow
LLM --> Retrieval : retrieve relevant chunks
LLM --> Skeleton : get cached skeleton
LLM --> Anthropic : API calls
LLM --> Streaming : stream responses
LLM --> Retry : error handling
LLM --> Editor : parse & apply edits
LLM --> State : update tokens/cost

' Retrieval pipeline
Retrieval --> Embeddings : encode queries/chunks
Retrieval --> State : access chunk_store
Retrieval --> Parsing : get code snippets

' Indexing pipeline
Indexer --> TreeSitter : parse AST
Indexer --> Parsing : chunk files
Indexer --> Embeddings : generate embeddings
Indexer --> Skeleton : update on file changes
Indexer --> State : update chunk_store

' Editor flow
Editor --> Linter : validate syntax
Editor --> GitManager : auto-commit changes
Editor --> UI : show diffs

' Git integration
GitManager --> Git : git operations
Main --> GitManager : /undo command

' Configuration
Config -[hidden]--> LLM : API keys, models
Config -[hidden]--> Retrieval : thresholds
Config -[hidden]--> Indexer : extensions

note right of State
  Shared state includes:
  - chunk_store (embeddings)
  - conversation_history
  - session_tokens
  - skeleton_context
  - file_hashes
end note

note bottom of Anthropic
  3-tier caching:
  1. Skeleton (session)
  2. History (turns)
  3. Chunks (queries)
end note

@enduml
```

</details>

---

## Query Processing Sequence

This sequence diagram illustrates the complete flow from user query to response, including background processes.

![Sequence Diagram](plantuml/Claude_Light_Sequence_Diagram.png)

### Key Phases

1. **Session Initialization** - Build skeleton, index files, warm cache
2. **Query Routing** - Weighted scoring of intent, complexity, and context to select the most efficient model.
3. **Chunk Retrieval** - Find relevant code via embedding similarity
4. **Context Assembly** - Build message with cache breakpoints
5. **API Call** - Stream response from Claude
6. **Response Processing** - Parse edits, lint, apply changes

### Background Processes

- **Heartbeat Daemon** - Keeps cache warm every 30 seconds
- **File Watcher** - Re-indexes changed files automatically

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/compact` or `/clear` | Reset conversation history |
| `/cost` | Show session spend so far |
| `/run <cmd>` | Run shell command and feed output to Claude |
| `/undo` | Undo the last git commit (revert AI changes) |
| `/help` | Display help menu |
| `exit` / `quit` | Quit the application |

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Claude_Light_Sequence_Diagram
title Claude Light - Query Processing Sequence

actor User
participant "main.py\n(Entry)" as Main
participant "llm.py\n(LLM Client)" as LLM
participant "retrieval.py\n(RAG)" as Retrieval
participant "state.py\n(Chunk Store)" as ChunkStore
participant "skeleton.py\n(Skeleton)" as Skeleton
participant "Anthropic API" as API
participant "editor.py\n(Editor)" as Editor
participant "linter.py\n(Linter)" as Linter

== Session Initialization ==
User -> Main : Start application
Main -> Skeleton : build_skeleton()
Skeleton --> Main : directory tree + .md files
Main -> LLM : warm_cache()
LLM -> API : minimal request with skeleton
API --> LLM : cache skeleton (ephemeral)
Main -> Main : Start heartbeat daemon
Main -> Main : Start file watcher

== Query Processing ==
User -> Main : Enter query
Main -> LLM : chat(query)

activate LLM
LLM -> LLM : route_query()
note right : Classifies effort level\n(low/medium/high/max)\nSelects model & budget

LLM -> Retrieval : retrieve(query, budget)
activate Retrieval
Retrieval -> ChunkStore : Get all embeddings
ChunkStore --> Retrieval : embedding matrix
Retrieval -> Retrieval : Encode query
Retrieval -> Retrieval : Matrix multiplication\nscores = embs @ q_emb
Retrieval -> Retrieval : Filter by MIN_SCORE\nand relative floor
Retrieval --> LLM : ranked chunks
deactivate Retrieval

LLM -> LLM : _maybe_compress_history()
LLM -> Skeleton : Get cached skeleton
Skeleton --> LLM : skeleton_context

LLM -> API : messages.create()\n- system (skeleton cached)\n- history (last turn cached)\n- chunks (cached if repeated)\n- new query
activate API
API --> LLM : Stream response\n(thinking + text)
deactivate API

LLM -> Editor : parse_edit_blocks(reply)
activate Editor
alt Has edits
  Editor --> LLM : list of edits
  LLM -> Linter : _lint_content()
  activate Linter
  alt Has errors
    Linter --> LLM : lint errors
    LLM -> API : Request correction (retry)
    API --> LLM : corrected code
  else No errors
    Linter --> LLM : OK
    LLM -> Editor : apply_edits()
    Editor -> User : Show diffs
    User --> Editor : Confirm y/n
    Editor --> Editor : Write files
    Editor --> LLM : Auto-commit if git
  end
  deactivate Linter
else No edits
  Editor --> LLM : empty list
  LLM -> User : Display answer
end
deactivate Editor

LLM -> LLM : Update conversation_history\n(strip code blocks)
LLM -> LLM : Update session_tokens & cost
LLM --> Main : Response complete
deactivate LLM

== Background Processes ==
note over Main : Heartbeat daemon (every 30s)
loop Every HEARTBEAT_SECS
  Main -> Main : Check idle time
  alt idle > CACHE_TTL_SECS (4 min)
    Main -> API : warm_cache()
    note right : Keep ephemeral cache alive\nPrevents 10x cold-start cost
  end
end

note over Main : File watcher (Watchdog)
loop On file change
  Main -> Main : Detect .md or source file
  alt .md file changed
    Main -> Skeleton : _refresh_single_md()
    Skeleton -> Skeleton : Rebuild skeleton
    Main -> API : warm_cache()
  else source file changed
    Main -> Main : reindex_file()
    note right : Re-chunk with tree-sitter\nRe-embed with sentence-transformers
  end
end

@enduml
```

</details>

---

## Three-Tier Caching Strategy

Claude Light uses Anthropic's ephemeral caching with three strategic breakpoints to minimize costs.

![Caching Diagram](plantuml/Claude_Light_Caching_Diagram.png)

### Cache Breakpoints

| Breakpoint | Location | Stable Across | Savings |
|------------|----------|---------------|---------|
| 1 | End of skeleton | Entire session | 90% off skeleton re-reads |
| 2 | End of history | All but newest turn | 90% off conversation history |
| 3 | End of chunks | Consecutive same-topic queries | 90% off retrieved context |

### Pricing (USD per 1M tokens)

| Type | Price |
|------|-------|
| Full Input | $3.00 |
| Cache Read | $0.30 (90% off) |
| Cache Write | $3.75 |
| Output | $15.00 |

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Claude_Light_Caching_Diagram
title Claude Light - Three-Tier Prompt Caching Strategy

skinparam backgroundColor #FEFEFE

rectangle "1. SYSTEM PROMPT" as SystemPrompt
rectangle "2. SKELETON (cached)" as Skeleton
rectangle "3. HISTORY (cached)" as History
rectangle "4. CHUNKS (cached)" as Chunks
rectangle "5. QUERY" as Query

SystemPrompt --> Skeleton
Skeleton --> History
History --> Chunks
Chunks --> Query

note right of Skeleton
  Breakpoint 1: ephemeral cache
  Stable across entire session
end note

note right of History
  Breakpoint 2: cache_control
  Stable for all but newest turn
end note

note right of Chunks
  Breakpoint 3: cache_control
  Stable for same-code questions
end note

rectangle "Full: $3.00" as FullPrice
rectangle "Cache Read: $0.30" as CacheRead

FullPrice ..> SystemPrompt
FullPrice ..> Query
CacheRead ..> Skeleton
CacheRead ..> History
CacheRead ..> Chunks

@enduml
```

</details>

---

## RAG Pipeline

The Retrieval-Augmented Generation pipeline finds and injects only relevant code into the context.

![RAG Pipeline](plantuml/Claude_Light_RAG_Pipeline.png)

### Key Optimizations

1. **Smarter Query Routing** - Weighted scoring (Arch vs Logic vs Infra) + context awareness
2. **Effort-Based Budget** - Low effort = 1.5K tokens, Max effort = 9K tokens
3. **Two-Stage Filtering** - Absolute floor (0.45) + Relative floor (60% of top score)
4. **Deduplication** - Shared preamble emitted once for multi-method results
5. **Code Block Stripping** - Edit blocks removed from history to prevent bloat

### Effort Levels

| Effort | Model | Max Tokens | Retrieval Budget | Thinking |
|--------|-------|------------|------------------|----------|
| low | Haiku | 2,048 | 1,500 | off |
| medium | Sonnet | 4,096 | 3,000 | off |
| high | Sonnet | 8,192 | 6,000 | off |
| max | Opus | 16,000 | 9,000 | on |

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Claude_Light_RAG_Pipeline
title Claude Light - Hybrid RAG Pipeline

start

:User enters query;
partition "1. Query Routing" {
  :Analyze query keywords (weighted);
  :Detect technical intent (regex);
  :Check conversation depth (context);
  if (Score >= 12.0?\n(High Arch/Logic + History)) then (yes)
    :effort = max\nmodel = Opus\nmax_tokens = 16,000;
  elseif (Score >= 5.0?\n(Edit intent + Path)) then (yes)
    :effort = high\nmodel = Sonnet\nmax_tokens = 8,192;
  elseif (Score >= 1.5 or WordCount > 15?) then (yes)
    :effort = medium\nmodel = Sonnet\nmax_tokens = 4,096;
  else (low/greeting)
    :effort = low\nmodel = Haiku\nmax_tokens = 2,048;
  endif
  :Set retrieval budget:\nlow=1.5K, med=3K, high=6K, max=9K tokens;
}

partition "2. Chunk Retrieval" {
  :Load embedding matrix from chunk_store;
  :Encode query with prefix\n(e.g., "search_query: ");
  :Compute similarity scores\nscores = embeddings @ query_emb;
  
  fork
    :Filter 1: MIN_SCORE >= 0.45\n(absolute floor);
  fork again
    :Filter 2: score >= top_score * 0.60\n(relative floor);
  end fork
  
  :Select top-k chunks\nwithin token budget;
  if (Multiple chunks from\nsame file?) then (yes)
    :Deduplicate: emit preamble once,\nlist methods below;
  endif
  :Render context with\nfile summaries + details;
}

partition "3. Context Assembly" {
  :Get skeleton from cache\n(directory tree + .md);
  :Compress conversation history\n(summarize old turns);
  :Build message array with\n3 cache breakpoints;
}

partition "4. API Call" {
  if (effort == max) then (yes)
    :Enable thinking mode\nbudget_tokens = 10,000;
  endif
  :Call Anthropic API\nwith streaming;
  :Stream response to terminal;
}

partition "5. Response Processing" {
  :Parse SEARCH/REPLACE blocks\nor new file blocks;
  if (Has edits?) then (yes)
    :Lint edited content;
    if (Syntax errors?) then (yes)
      :Request correction from Claude\n(retry up to 3x);
    else (no errors)
      :Show diffs to user;
      :Apply edits after confirmation;
      :Auto-commit if git repo;
    endif
  else (no edits)
    :Display answer only;
  endif
  :Strip code blocks from history\n(store "[Files edited: ...]" instead);
  :Update session tokens & cost;
}

stop

note right
  <b>Key Optimizations:</b>
  1. Dynamic model routing saves 8x on simple queries
  2. Effort-based retrieval budget prevents over-fetching
  3. Two-stage filtering (absolute + relative) removes noise
  4. Deduplication saves 5-20% on multi-method queries
  5. Code block stripping prevents history bloat
end note

@enduml
```

</details>

---

## File Indexing Flow

The indexing system builds and maintains an incremental cache of code embeddings.

![Indexing Flow](plantuml/Claude_Light_Indexing_Flow.png)

### Auto-Tuned Models

| Repo Size | Model | Size | Use Case |
|-----------|-------|------|----------|
| < 50 files | all-MiniLM-L6-v2 | 22 MB | Fast startup |
| 50-199 files | all-mpnet-base-v2 | 420 MB | Better depth |
| 200+ files | nomic-embed-text-v1.5 | 275 MB | Optimal recall |

### Method-Level Chunking

Instead of whole files, tree-sitter parses the AST and extracts individual methods:

```
chunk_id: filepath::methodName
text: // filepath
      package com.example;
      import java.util.*;
      
      public class Service {
          // ...
      }
      
      public ReturnType methodName(Args) {
          // method body
      }
```

**Benefits:**
- Self-contained: each chunk has full context
- Precise retrieval: fetch only relevant methods
- Deduplication: shared preamble emitted once
- Token efficiency: 5-10x reduction vs whole files

### Incremental Indexing

- Only changed files are re-chunked and re-embedded
- Cache is loaded on startup, avoiding redundant work
- File watcher triggers re-index on save
- Debounced at 1.5s to avoid rapid re-indexing

<details>
<summary>View PlantUML Source</summary>

```plantuml
@startuml Claude_Light_Indexing_Flow
title Claude Light - File Indexing & Auto-Tuning

start

:Scan project directory;
:Filter by INDEXABLE_EXTENSIONS;
:Skip directories (.git, build, etc.);

partition "Auto-Tune Model" {
  if (Files < 50) then (small)
    :all-MiniLM-L6-v2 (22 MB);
  elseif (Files < 200) then (medium)
    :all-mpnet-base-v2 (420 MB);
  else (large)
    :nomic-embed-text-v1.5;
  endif
}

partition "Cache Lookup" {
  :Load manifest.json;
  if (model matches?) then (yes)
    :Compare file hashes;
    if (hash matches) then (hit)
      :Load from index.pkl;
    else (miss)
      :Mark for re-indexing;
    endif
  else (changed)
    :Full re-index;
  endif
}

partition "Chunking" {
  :Process each file;
  if (tree-sitter?) then (yes)
    :Parse AST;
    :Extract methods;
    :Create chunks with preamble;
  else (no)
    :Whole file chunk;
  endif
}

partition "Embedding" {
  :Add document prefix;
  :Batch encode chunks;
  :Store in chunk_store;
}

partition "Save Cache" {
  :Save index.pkl;
  :Save manifest.json;
}

stop

note right
  Incremental Indexing:
  - Only changed files re-indexed
  - File watcher on save
  - Debounced at 1.5s
end note

@enduml
```

</details>

---

## Related Documentation

- [README.md](../README.md) - Getting started and usage guide
- [Test Strategy](test_strategy.md) - Test suite architecture and CI/CD pipeline
- [CLAUDE.md](../CLAUDE.md) - Project guidelines and commands
