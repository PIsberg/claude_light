import threading
import time
import json
from claude_light.config import GLOBAL_STATS_FILE

chunk_store = {}
conversation_history = []
session_cost = 0.0
session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "compress_pre": 0, "compress_post": 0}

# Global statistics accumulated across all sessions
global_stats = {
    "total_tokens_full": 0,   # Tokens that were billed at 100%
    "total_tokens_saved": 0,  # Tokens that were billed at 10% (cache hits)
    "total_dollars_saved": 0.0,
    "total_sessions": 0,
    # LLMLingua-2 compression stats (additive with the cache savings above)
    "total_tokens_pre_compress": 0,
    "total_tokens_post_compress": 0,
    "total_dollars_saved_llmlingua": 0.0,
}

def load_global_stats():
    global global_stats
    if GLOBAL_STATS_FILE.exists():
        try:
            data = json.loads(GLOBAL_STATS_FILE.read_text(encoding="utf-8"))
            global_stats.update(data)
        except Exception:
            pass

def save_global_stats():
    with lock:
        try:
            GLOBAL_STATS_FILE.write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
        except Exception:
            pass

# Initialize global stats on import
load_global_stats()
global_stats["total_sessions"] += 1

cli_session_id: str | None = None   # OAuth/Pro: reuse CLI session across turns
last_interaction = time.time()
lock = threading.Lock()
stop_event = threading.Event()
_source_files = []
_file_hashes = {}
_file_stats = {}            # {path: [mtime, size]} — fast-path to skip re-hashing unchanged files
skeleton_context = ""
_skeleton_tree = ""
_skeleton_md_hashes = {}
_skeleton_md_stats = {}     # same fast-path cache for .md files
_skeleton_md_parts = {}

# Diagnostics for the most recent query, surfaced by the /why command so the
# RAG black box (model/effort routing + which chunks were retrieved at what
# score) is inspectable. Set by llm.chat(); None until the first query.
last_retrieval = None

EMBED_MODEL = None
TOP_K = None
embedder = None
device: str | None = None   # "cuda" | "mps" | "cpu" — detected on first model load

# Set once the embedder is actually loaded into state.embedder. When we can
# defer model load (full cache hit), index_files() starts a background thread
# and retrieve() waits on this before using state.embedder.
embedder_ready = threading.Event()
embedder_load_error: Exception | None = None

# Serializes calls into state.embedder.encode(). Several watchdog debounce
# timers can fire concurrently (one per changed file), and PyTorch CPU
# inference is not guaranteed safe under concurrent forward passes.
embedder_lock = threading.Lock()

# Serializes on-disk cache writes (index.pkl + manifest.json). Without it two
# concurrent reindex_file() timers can interleave writes and corrupt the cache.
cache_io_lock = threading.Lock()
