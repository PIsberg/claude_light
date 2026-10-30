"""Codex provider: the `codex` CLI as a subprocess."""
import json
import os
import shutil
import sys
from pathlib import Path

from claude_light.providers.base import Provider, ProviderError, transcript
from claude_light.providers.runner import empty_temp_dir, run_cli

BINARY = "codex"
_LABEL = "CLI default model"
_ARGV_TAIL = [
    "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
    "--sandbox", "read-only",
    "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "apps", "-",
]


def _agent_reply(stdout: str) -> str:
    """The text of the last agent_message event in codex's JSONL output."""
    reply = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            reply = item.get("text")
    if not isinstance(reply, str) or not reply.strip():
        raise ProviderError("codex output could not be parsed: no agent_message in the output")
    return reply.strip()


class CodexProvider(Provider):
    name = "codex"
    capabilities = frozenset()

    def model_label(self, routed_model: str) -> str:
        return _LABEL

    def header(self):
        return (_LABEL, "codex CLI")

    def require_auth(self) -> None:
        from claude_light.ui import _T_ERR
        if not shutil.which(BINARY):
            print(f"\n  {_T_ERR}  codex CLI not found. Install it: "
                  f"npm install -g @openai/codex", file=sys.stderr)
            sys.exit(1)
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        if (home / "auth.json").exists():
            return
        print(f"\n  {_T_ERR}  codex is not logged in. Run: codex login", file=sys.stderr)
        sys.exit(1)

    def _run(self, stdin_text: str) -> str:
        binary = shutil.which(BINARY)
        if not binary:
            raise ProviderError("codex CLI not found. Install it: "
                                "npm install -g @openai/codex")
        with empty_temp_dir() as cwd:
            out = run_cli(self.name, [binary] + _ARGV_TAIL, stdin_text, cwd=cwd)
        return _agent_reply(out)

    def complete(self, request: dict):
        from claude_light.ui import _Spinner
        system_text, prompt = transcript(request)
        with _Spinner("Waiting for codex"):
            reply = self._run(system_text + "\n\n" + prompt)
        return reply, None, False

    def summarize(self, instructions: str):
        from claude_light.ui import _Spinner
        with _Spinner("Summarizing"):
            text = self._run("You summarize conversations.\n\n" + instructions)
        return text, None


PROVIDER = CodexProvider()
