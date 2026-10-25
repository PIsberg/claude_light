"""
Run only the tests an edit can affect.

After claude_light applies and commits an edit, find the test files that
mention the edited modules and run just those, using a test command the user
already allowed (--allow-command "python -m pytest"). If they fail, offer to
undo the commit. With no allowlisted pytest command this does nothing.
"""

import re
from pathlib import Path

from claude_light.config import SKIP_DIRS

_MAX_TEST_FILES = 20
_MAX_SCAN_BYTES = 500_000
_SKIP = set(SKIP_DIRS) | {".venv", "venv", "dist", ".claude_light_cache"}


def is_test_file(path):
    """A file pytest collects tests from. conftest.py and helpers or fixtures
    under tests/ are not: passed to pytest alone they collect nothing and
    pytest exits 5, which would read as a failure."""
    p = Path(path)
    return p.suffix == ".py" and (p.name.startswith("test_") or p.name.endswith("_test.py"))


def _all_test_files(root):
    found = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in _SKIP or part.startswith(".") for part in rel.parts):
            continue
        if is_test_file(rel):
            found.append(p)
    return sorted(found)


def find_affected_tests(edited, root=None):
    """Relative posix paths of the test files affected by editing `edited`:
    an edited test file itself, plus every test file that mentions an edited
    module's name. Empty when nothing maps."""
    root = Path(root or ".").resolve()
    stems = set()
    direct = []
    for f in edited:
        p = Path(f)
        if p.suffix != ".py":
            continue
        if is_test_file(p):
            try:
                direct.append(((root / p).resolve()).relative_to(root).as_posix())
            except ValueError:
                pass
            continue
        if p.stem != "__init__":
            stems.add(p.stem)
        elif p.parent.name:
            stems.add(p.parent.name)
    hits = set(direct)
    if stems:
        rx = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in sorted(stems)) + r")\b")
        for t in _all_test_files(root):
            try:
                if t.stat().st_size > _MAX_SCAN_BYTES:
                    continue
                text = t.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if rx.search(text):
                hits.add(t.relative_to(root).as_posix())
    return sorted(hits)[:_MAX_TEST_FILES]


def build_test_command(files, allowed=None):
    """An allowlisted pytest command extended with `files`, or None."""
    from claude_light import config
    if not files:
        return None
    for entry in (config.ALLOWED_COMMANDS if allowed is None else allowed):
        _tool, prefix = config.command_rule(entry)
        if "pytest" in prefix:
            return f"{prefix} {' '.join(files)} -q"
    return None


def check_after_edit(written_files, root=None):
    """Run the affected tests for `written_files`. Returns None when nothing
    ran, else True (passed) or False (failed). Never raises."""
    try:
        from claude_light import config
        if not config.ALLOWED_COMMANDS:
            return None
        files = find_affected_tests(written_files, root)
        cmd = build_test_command(files)
        if not cmd:
            return None
        from claude_light import tools
        from claude_light.ui import _T_SYS, _T_ERR
        print(f"  {_T_SYS}  Running {len(files)} affected test file{'s' if len(files) != 1 else ''}: {cmd}")
        out, failed = tools.run_tool("shell", {"command": cmd}, root=root)
        if not failed:
            print(f"  {_T_SYS}  Affected tests passed.")
            return True
        print(out)
        print(f"  {_T_ERR}  Affected tests failed.")
        import sys
        from claude_light.ui import choose
        from claude_light import git_manager
        if sys.stdin.isatty() and git_manager.is_git_repo() and choose(
                "Revert the last commit?",
                [("no", "No, keep the edit"), ("yes", "Yes, undo it")]) == "yes":
            git_manager.undo_last_commit()
        return False
    except Exception:
        return None
