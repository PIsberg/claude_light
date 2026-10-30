"""PreToolUse hook for the OAuth CLI agent's Bash and PowerShell tools.

An --allow-command prefix becomes a Bash(<prefix>:*) rule, and the CLI matches
that rule against the first statement only: under Bash(echo:*),
`echo step1 && touch marker` ran the touch as well (issue #70, verified with
claude 2.1.314). This hook refuses any command containing a chaining,
redirection or substitution character, the same rule the API-mode `shell`
tool applies, so an allowed prefix runs that one command and nothing after it.

The CLI runs it as `python -I <this file>` with the hook JSON on stdin. It is
stdlib-only because -I keeps the project directory and PYTHONPATH off
sys.path. Exit 2 blocks the call and shows stderr to the model; any other
non-zero exit lets the command run, so every failure here exits 2.
"""

import json
import sys

SHELL_FORBIDDEN = frozenset(";&|<>`$\n\r")


def check(payload: str) -> str | None:
    """The reason to block the call described by `payload`, or None to let it run."""
    try:
        command = json.loads(payload)["tool_input"]["command"]
    except (ValueError, TypeError, KeyError):
        return "claude_light could not read the command; it was not run"
    if not isinstance(command, str):
        return "claude_light could not read the command; it was not run"
    bad = sorted(set(command) & SHELL_FORBIDDEN)
    if bad:
        return ("command may not contain " + " ".join(repr(c) for c in bad)
                + "; run one simple command per call")
    return None


def main() -> int:
    try:
        reason = check(sys.stdin.read())
    except Exception:  # fail closed: exit 1 would let the command run
        reason = "claude_light's command check failed; the command was not run"
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
