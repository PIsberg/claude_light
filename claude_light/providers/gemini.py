"""Gemini provider: the `gemini` CLI as a subprocess."""
import os
import shutil
import sys
from pathlib import Path

from claude_light.providers.base import Provider, ProviderError, transcript
from claude_light.providers.runner import empty_temp_dir, run_cli

BINARY = "gemini"
_LABEL = "CLI default model"
_AUTH_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")
_ARGV_TAIL = [
    "-p", "Answer the request given on standard input.",
    "-o", "text", "--approval-mode", "plan",
    "--allowed-mcp-server-names", "__claude_light_none__",
]


class GeminiProvider(Provider):
    name = "gemini"
    capabilities = frozenset()

    def model_label(self, routed_model: str) -> str:
        return _LABEL

    def header(self):
        return (_LABEL, "gemini CLI")

    def require_auth(self) -> None:
        from claude_light.ui import _T_ERR
        if not shutil.which(BINARY):
            print(f"\n  {_T_ERR}  gemini CLI not found. Install it: "
                  f"npm install -g @google/gemini-cli", file=sys.stderr)
            sys.exit(1)
        if (any(os.environ.get(v) for v in _AUTH_ENV)
                or (Path.home() / ".gemini" / "oauth_creds.json").exists()):
            return
        print(f"\n  {_T_ERR}  gemini is not logged in. Run gemini once and sign in, "
              f"or set GEMINI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    def _run(self, stdin_text: str) -> str:
        binary = shutil.which(BINARY)
        if not binary:
            raise ProviderError("gemini CLI not found. Install it: "
                                "npm install -g @google/gemini-cli")
        with empty_temp_dir() as cwd:
            out = run_cli(self.name, [binary] + _ARGV_TAIL, stdin_text, cwd=cwd).strip()
        if not out:
            raise ProviderError("gemini output could not be parsed: empty output")
        return out

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
