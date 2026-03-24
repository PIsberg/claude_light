import hashlib
from pathlib import Path

def _file_hash(path: Path) -> str:
    """MD5 of file bytes — fast change detection, not cryptographic."""
    return hashlib.md5(path.read_bytes()).hexdigest()
