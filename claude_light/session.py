"""Save the conversation after each turn, so --continue can resume it.

One file per project, named by a hash of the project's absolute path, under
config.SESSION_DIR in the user's home rather than the project's
.claude_light_cache: a cloned repo could otherwise ship a crafted
conversation that --continue would hand to the model as the user's own.
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import claude_light.config as config
import claude_light.state as state

_ROLES = {"user", "assistant"}


def session_file(root: Path | None = None) -> Path:
    project = os.path.normcase(str(Path(root or Path.cwd()).resolve()))
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:16]
    return Path(config.SESSION_DIR) / f"{digest}.json"


def save() -> bool:
    """Write the conversation, or remove the file when there is none.
    Returns False, after a warning, if the write failed."""
    path = session_file()
    with state.lock:
        history = list(state.conversation_history)
        data = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "project": str(Path.cwd().resolve()),
            "turn_count": state.turn_count,
            "session_model": state.session_model,
            "conversation_history": history,
        }
    try:
        if not history:
            path.unlink(missing_ok=True)
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)  # never leave a half-written file to resume
        return True
    except (OSError, TypeError, ValueError) as e:
        print(f"  Could not save the conversation for --continue: {e}", file=sys.stderr)
        return False


def clear() -> None:
    try:
        session_file().unlink(missing_ok=True)
    except OSError:
        pass


def _valid(data) -> bool:
    if not isinstance(data, dict):
        return False
    history = data.get("conversation_history")
    if not isinstance(history, list):
        return False
    for message in history:
        if not (isinstance(message, dict) and message.get("role") in _ROLES
                and isinstance(message.get("content"), (str, list))):
            return False
    return (isinstance(data.get("turn_count", 0), int)
            and isinstance(data.get("session_model"), (str, type(None))))


def restore() -> str:
    """Load the saved conversation into state. Returns a line for the user."""
    path = session_file()
    if not path.is_file():
        return "No previous conversation for this project; starting a new one."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"Could not read the saved conversation ({e}); starting a new one."
    if not _valid(data):
        return "Could not read the saved conversation (unexpected format); starting a new one."
    with state.lock:
        # In place: the package's _wrap() re-binds state.conversation_history
        # to this same list before each call, so a new list would be dropped.
        state.conversation_history[:] = data["conversation_history"]
        state.turn_count = data.get("turn_count", len(data["conversation_history"]) // 2)
        state.session_model = data.get("session_model")
    turns = state.turn_count
    when = str(data.get("saved_at", "")).replace("T", " ")[:16]
    return (f"Continuing the conversation from {when or 'an earlier session'} "
            f"({turns} turn{'' if turns == 1 else 's'}).")
