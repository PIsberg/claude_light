"""Module entrypoint for ``python -m claude_light``."""

import sys

from .main import main


for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
