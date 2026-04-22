"""Module entrypoint for ``python -m claude_light``."""

import sys

# Reconfigure stdout/stderr to UTF-8 BEFORE any other imports so that
# ui.py's _UNICODE detection (computed at import time) sees UTF-8 and
# uses the full symbol set rather than the ASCII fallback.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from .main import main

if __name__ == "__main__":
    main()
