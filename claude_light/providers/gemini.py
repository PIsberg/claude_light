"""Gemini provider: Google's Antigravity CLI (`agy`) as a subprocess."""
import json
import shutil
import sys

from claude_light.providers.base import Provider, ProviderError, json_events, transcript
from claude_light.providers.runner import empty_temp_dir, run_cli

BINARY = "agy"
_LABEL = "CLI default model"
_MISSING = "agy not found. Install the Antigravity CLI: https://antigravity.google"
# Deny shell, MCP and file tools: agy otherwise runs the user's globally allowed MCP shell.
SETTINGS = json.dumps({
    "permissions": {"allow": [], "deny": ["command", "mcp", "file"]},
    "allowNonWorkspaceAccess": False,
})
# The empty attached `-p=` is required: a bare `-p` swallows the next flag as its prompt.
_ARGV_TAIL = [
    "--input-format", "stream-json", "--output-format", "stream-json",
    "--mode", "plan", "--settings", SETTINGS, "-p=",
]


def _stdin_line(text: str) -> str:
    return json.dumps({"event": "user", "message": {"role": "user", "content": text}}) + "\n"


def _result_reply(stdout: str) -> str:
    """The response of the last `result` event; ProviderError if it is missing or failed."""
    result = None
    for event in json_events(stdout):
        if event.get("event") == "result" and isinstance(event.get("result"), dict):
            result = event["result"]
    if result is None:
        raise ProviderError("agy output could not be parsed: no result event in the output")
    if result.get("status") != "SUCCESS":
        raise ProviderError(f"agy failed: {result.get('error') or result.get('status')}. "
                            f"If you are not signed in, run agy once and sign in.")
    reply = result.get("response")
    if not isinstance(reply, str) or not reply.strip():
        raise ProviderError("agy output could not be parsed: empty response")
    return reply.strip()


class GeminiProvider(Provider):
    name = "gemini"
    capabilities = frozenset()

    def model_label(self, routed_model: str) -> str:
        return _LABEL

    def header(self):
        return (_LABEL, "agy CLI")

    def require_auth(self) -> None:
        from claude_light.ui import _T_ERR
        if not shutil.which(BINARY):
            print(f"\n  {_T_ERR}  {_MISSING}", file=sys.stderr)
            sys.exit(1)

    def _run(self, stdin_text: str) -> str:
        binary = shutil.which(BINARY)
        if not binary:
            raise ProviderError(_MISSING)
        with empty_temp_dir() as cwd:
            out = run_cli(self.name, [binary] + _ARGV_TAIL, _stdin_line(stdin_text), cwd=cwd)
        return _result_reply(out)

    def complete(self, request: dict):
        from claude_light.ui import _Spinner
        system_text, prompt = transcript(request)
        with _Spinner("Waiting for gemini"):
            reply = self._run(system_text + "\n\n" + prompt)
        return reply, None, False

    def summarize(self, instructions: str):
        from claude_light.ui import _Spinner
        with _Spinner("Summarizing"):
            text = self._run("You summarize conversations.\n\n" + instructions)
        return text, None


PROVIDER = GeminiProvider()
