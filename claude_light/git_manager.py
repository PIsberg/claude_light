"""
Git repository management for claude_light.

Handles automatic committing of AI-generated changes with user-friendly undo.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, List, Tuple
from claude_light.ui import _T_EDIT, _T_ERR, _T_SYS, _ANSI_GREEN, _ANSI_RED, _ANSI_RESET, _ANSI_DIM, _ANSI_BOLD


def _run_git(*args: str) -> Tuple[str, int]:
    """
    Run a git command and return (stdout, returncode).
    
    Args:
        *args: Git subcommand and arguments.
        
    Returns:
        Tuple of (command output, return code).
    """
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip(), result.returncode
    except FileNotFoundError:
        return "", 127  # git not found


def is_git_repo() -> bool:
    """
    Check if the current directory is a git repository.
    
    Returns:
        True if we're in a git repo, False otherwise.
    """
    _, code = _run_git("rev-parse", "--git-dir")
    return code == 0


def get_git_root() -> Optional[Path]:
    """
    Get the root directory of the git repository.
    
    Returns:
        Path to git root, or None if not in a git repo.
    """
    stdout, code = _run_git("rev-parse", "--show-toplevel")
    if code == 0:
        return Path(stdout)
    return None


def get_modified_files() -> List[str]:
    """
    Get list of modified/staged files in the working directory.
    
    Returns:
        List of file paths that have been modified or staged.
    """
    stdout, code = _run_git("status", "--porcelain")
    if code != 0:
        return []
    
    files = []
    for line in stdout.splitlines():
        if not line:
            continue
        # Parse git status output: "M  path", "A  path", etc.
        # First two chars are status, followed by space and file path
        if len(line) > 3:
            files.append(line[3:])
    return files


def get_last_commit_message() -> Optional[str]:
    """
    Get the message of the last commit.
    
    Returns:
        Last commit message, or None if no commits exist.
    """
    stdout, code = _run_git("log", "-1", "--pretty=%B")
    if code == 0 and stdout:
        return stdout
    return None


def auto_commit(
    edited_files: List[str],
    explanation: str = "",
    auto: bool = True
) -> bool:
    """
    Automatically commit file changes with a descriptive message.
    
    Args:
        edited_files: List of file paths that were edited.
        explanation: Brief explanation of changes (used to generate commit message).
        auto: If True, commit without confirmation. If False, ask user first.
        
    Returns:
        True if commit succeeded, False otherwise.
    """
    if not is_git_repo():
        return False
    
    # Stage files
    if edited_files:
        for file_path in edited_files:
            _, code = _run_git("add", file_path)
            if code != 0:
                print(f"{_T_ERR} Failed to stage {file_path}")
                return False
    else:
        # Stage all changes if no specific files provided
        _, code = _run_git("add", "-A")
        if code != 0:
            print(f"{_T_ERR} Failed to stage changes")
            return False
    
    # Check if there are staged changes
    stdout, code = _run_git("diff", "--cached", "--quiet")
    if code == 0:
        # No changes to commit
        return False
    
    # Generate commit message
    files_summary = ""
    if edited_files and len(edited_files) <= 3:
        files_summary = f": {', '.join(Path(f).name for f in edited_files)}"
    elif edited_files:
        files_summary = f": {len(edited_files)} files"
    
    explanation_clean = explanation.strip().split("\n")[0][:60]  # First line, max 60 chars
    if explanation_clean:
        commit_msg = f"Claude: {explanation_clean}{files_summary}"
    else:
        commit_msg = f"Claude: Applied AI-generated changes{files_summary}"
    
    # Create commit
    _, code = _run_git("commit", "-m", commit_msg)
    if code != 0:
        print(f"{_T_ERR} Failed to create commit")
        return False
    
    print(f"{_T_EDIT} {_ANSI_GREEN}Committed{_ANSI_RESET}: {commit_msg}")
    return True


def undo_last_commit() -> bool:
    """
    Undo the last commit by resetting to HEAD~1.
    
    Returns:
        True if undo succeeded, False otherwise.
    """
    if not is_git_repo():
        print(f"{_T_ERR} Not in a git repository")
        return False
    
    last_msg = get_last_commit_message()
    
    # Get commit hash before reset
    stdout, code = _run_git("rev-parse", "HEAD")
    if code != 0:
        print(f"{_T_ERR} Failed to get commit hash")
        return False
    commit_hash = stdout[:7] if stdout else "??????"
    
    # Reset to previous commit
    _, code = _run_git("reset", "--hard", "HEAD~1")
    if code != 0:
        print(f"{_T_ERR} Failed to undo commit")
        return False
    
    msg_display = f" ({last_msg})" if last_msg else ""
    print(f"{_T_EDIT} {_ANSI_GREEN}Undone{_ANSI_RESET}: {commit_hash}{msg_display}")
    return True


def get_commit_history(n: int = 5) -> List[str]:
    """
    Get recent commit messages for reference.
    
    Args:
        n: Number of commits to retrieve.
        
    Returns:
        List of recent commit messages.
    """
    if not is_git_repo():
        return []
    
    stdout, code = _run_git("log", f"-{n}", "--oneline")
    if code == 0:
        return stdout.splitlines()
    return []
