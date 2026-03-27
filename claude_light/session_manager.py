"""
Session persistence for Claude Light.

Saves conversation history and state to disk so users can resume sessions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from claude_light.config import CACHE_DIR
import claude_light.state as state


SESSION_FILE = CACHE_DIR / "session.json"


def save_session() -> bool:
    """
    Save current conversation session to disk.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "turns": len(state.conversation_history) // 2,
            "conversation_history": state.conversation_history,
            "session_tokens": state.session_tokens.copy(),
            "session_cost": state.session_cost,
        }
        
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"[Warning] Failed to save session: {e}", file=sys.stderr)
        return False


def load_session() -> Optional[Dict[str, Any]]:
    """
    Load previous session from disk.
    
    Returns:
        Session dict if file exists and is valid, None otherwise
    """
    if not SESSION_FILE.exists():
        return None
    
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        # Validate structure
        required_keys = {"conversation_history", "session_tokens", "session_cost", "timestamp"}
        if not required_keys.issubset(session_data.keys()):
            return None
        
        return session_data
    except Exception as e:
        print(f"[Warning] Failed to load session: {e}", file=sys.stderr)
        return None


def restore_session(session_data: Dict[str, Any]) -> bool:
    """
    Restore session state from loaded data.
    
    Args:
        session_data: Session dict from load_session()
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with state.lock:
            state.conversation_history = session_data.get("conversation_history", [])
            state.session_tokens = session_data.get("session_tokens", 
                                                     {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
            state.session_cost = session_data.get("session_cost", 0.0)
        return True
    except Exception as e:
        print(f"[Warning] Failed to restore session: {e}", file=sys.stderr)
        return False


def format_session_info(session_data: Dict[str, Any]) -> str:
    """
    Format session info for display to user.
    
    Returns:
        Formatted string with session details
    """
    timestamp = session_data.get("timestamp", "Unknown")
    turns = session_data.get("turns", 0)
    cost = session_data.get("session_cost", 0.0)
    tokens = session_data.get("session_tokens", {})
    total_tokens = tokens.get("input", 0) + tokens.get("output", 0)
    
    # Parse timestamp for friendly display
    try:
        dt = datetime.fromisoformat(timestamp)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        time_str = timestamp
    
    return (f"Session from {time_str}: {turns} turns, {total_tokens:,} tokens, ${cost:.4f}")


def delete_session() -> bool:
    """Delete saved session file."""
    try:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
        return True
    except Exception as e:
        print(f"[Warning] Failed to delete session: {e}", file=sys.stderr)
        return False


def prompt_resume_session() -> bool:
    """
    Prompt user if they want to resume previous session.
    
    Returns:
        True if user wants to resume, False otherwise
    """
    session_data = load_session()
    if session_data is None:
        return False
    
    session_info = format_session_info(session_data)
    print(f"\n[Session] Found previous session: {session_info}")
    print("[Session] Resume? [y/n] ", end="", flush=True)
    
    try:
        response = input().strip().lower()
        if response in ("y", "yes"):
            if restore_session(session_data):
                turns = session_data.get("turns", 0)
                print(f"[Session] Restored {turns} turns.\n")
                return True
    except (KeyboardInterrupt, EOFError):
        pass
    
    return False
