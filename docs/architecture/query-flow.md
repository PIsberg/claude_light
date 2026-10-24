# Query Processing Sequence

> Part of the [architecture index](ARCHITECTURE.md). Source: the original `docs/architecture.md`, split by topic.


This sequence diagram illustrates the complete flow from user query to response, including background processes.

![Sequence Diagram](../plantuml/Claude_Light_Sequence_Diagram.png)

### Key Phases

1. **Session Initialization** - Build skeleton, index files, warm cache
2. **Query Routing** - Weighted scoring of intent, complexity, and context to select the most efficient model.
3. **Chunk Retrieval** - Find relevant code via embedding similarity
4. **Context Assembly** - Build message with cache breakpoints
5. **API Call** - Stream response from Claude. `ui.ResponseView` renders the turn: a spinner with elapsed time until text arrives, then the reply as markdown. A block is printed once it is finished (a blank line outside a code fence, or a closed fence); only the block still arriving is redrawn, cropped to the screen height, so a reply taller than the terminal leaves no duplicated frames. Tool calls print on their own line. File-edit blocks (```` ```lang:path ````) are shown as one line (`✎ edit to path`), since the diff after the turn shows the change. With stdout redirected, the reply is written as plain text and all progress goes to stderr.
6. **Response Processing** - Parse edits, lint, apply changes

### Background Processes

- **Heartbeat Daemon** - Keeps cache warm every 30 seconds
- **File Watcher** - Re-indexes changed files automatically

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/clear` | Reset conversation history |
| `/compact` | Replace the conversation with a summary of it (`llm.compact_history()`; Haiku, through the CLI in OAuth mode) |
| `/cost` | Show session spend and global lifetime savings |
| `/why` | Explain the last query's routing and retrieval |
| `/run <cmd>` | Run shell command and feed output to Claude |
| `/undo` | Undo the last `Claude:` commit. Keeps uncommitted work; refuses if the last commit is yours or its files have local edits |
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
LLM -> LLM : Persist global_stats to HOME
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


## Confirmation prompts

Every yes/no question (applying edits, committing the CLI agent's edits, `/undo`, token setup) goes through `ui.choose()`. It draws an inline prompt_toolkit menu under the diff and falls back to typed input without a console. "Don't ask again" sets `state.auto_approve_edits` (SEARCH/REPLACE writes) or `state.auto_commit_agent_edits` (agent commits), never both. `choose()` discards type-ahead before drawing and only numbers pick at once, so keys typed during a turn cannot answer it. Paths in diff titles are markup-escaped, because they come from the model.
