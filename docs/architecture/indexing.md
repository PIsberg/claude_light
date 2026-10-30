# File Indexing Flow

> Part of the [architecture index](ARCHITECTURE.md). Source: the original `docs/architecture.md`, split by topic.


The indexing system builds and maintains an incremental cache of code embeddings.

![Indexing Flow](../plantuml/Claude_Light_Indexing_Flow.png)

### Hybrid index (default)

Every chunk gets two vectors (#61). `indexer._index_files_hybrid()` first embeds stale chunks with the static model `static-retrieval-mrl-en-v1` (no transformer; about 2 s for this repo), saves the cache and returns, so retrieval works from the first question. A background thread (`_complete_dense_index()`) then adds `all-mpnet-base-v2` vectors a batch at a time (about 380 s cold on a CPU) and sets `state.dense_ready`. One-shot runs pass `dense="skip"` and use only dense vectors already cached. Until the dense pass finishes, retrieval uses the static scores alone; after it, each model's scores are z-scored and averaged (see [rag-pipeline.md](rag-pipeline.md)). `CLAUDE_LIGHT_HYBRID=0` switches to the single auto-tuned model below.

### Auto-Tuned Models (`CLAUDE_LIGHT_HYBRID=0`)

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
title Claude Light - File Indexing (hybrid index)

start

:Scan project directory;
:Filter by INDEXABLE_EXTENSIONS;
:Skip directories (.git, build, etc.);

if (HYBRID_RETRIEVAL?\n(default on; CLAUDE_LIGHT_HYBRID=0 turns it off)) then (yes)

  partition "Cache Lookup" {
    :Load manifest.json + index.npz\n(allow_pickle=False);
    :Compare file hashes;
    :Unchanged files: reuse chunks\nand cached vectors;
    :Changed or new files: stale;
  }

  partition "Chunking (stale files)" {
    :Process each file in a thread pool;
    if (tree-sitter?) then (yes)
      :Parse AST;
      :Extract methods;
      :Create chunks with preamble;
    else (no)
      :Whole file chunk;
    endif
  }

  partition "Static pass (blocking, ~2 s)" {
    :static-retrieval-mrl-en-v1\nencodes chunks with no emb_fast;
    :Store emb_fast in chunk_store;
    :Save index.npz + manifest.json;
    :Index ready: retrieval works now\nwith static vectors alone;
  }

  partition "Dense pass (background thread)" {
    if (Chunks still missing dense vectors?) then (yes)
      if (dense mode) then (background)
        :all-mpnet-base-v2 encodes\na batch at a time (~380 s cold);
        :Store emb, save cache;
      elseif (sync) then
        :Same, but wait for it;
      else (skip: one-shot run)
        :Use only cached dense vectors;
      endif
    else (no)
      :Load the dense model\nfor query encoding only;
    endif
    :Set dense_ready;
  }

else (no: single model)

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
        :Load from index.npz;
      else (miss)
        :Mark for re-indexing;
      endif
    else (changed)
      :Full re-index;
    endif
  }

  partition "Chunking + Embedding" {
    :Chunk stale files (tree-sitter);
    :Add document prefix;
    :Batch encode chunks;
    :Store in chunk_store;
    :Save index.npz + manifest.json;
  }

endif

stop

note right
  Incremental Indexing:
  - Only changed files re-indexed
  - File watcher on save
  - Debounced at 1.5s
  - Never loads index.pkl
    (deleted if found)
end note

@enduml
```

</details>

---


## Disk cache

Embeddings persist to `.claude_light_cache/index.npz` between runs. Ids and texts are stored as JSON and vectors as float32 arrays. The file is loaded with `allow_pickle=False`: the directory may come from a cloned repo, and the old `index.pkl` ran whatever a repo shipped in it. An old `index.pkl` is deleted, never read. A `manifest.json` tracks per-file MD5 hashes and the embedding model used, so only changed or new files are re-embedded on restart. Command history (prompt_toolkit) is stored here too. The conversation is not: `session.py` saves it for `--continue` under `~/.claude_light_sessions/`, outside the project, so a cloned repo cannot supply one.
