import threading
import time

chunk_store = {}
conversation_history = []
session_cost = 0.0
session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
last_interaction = time.time()
lock = threading.Lock()
stop_event = threading.Event()
_source_files = []
_file_hashes = {}
skeleton_context = ""
_skeleton_tree = ""
_skeleton_md_hashes = {}
_skeleton_md_parts = {}

EMBED_MODEL = None
TOP_K = None
embedder = None
