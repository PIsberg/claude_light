"""Codex provider: the `codex` CLI as a subprocess (stub)."""
import shutil  # noqa: F401  (tests patch shutil.which here)

from claude_light.providers.base import Provider
from claude_light.providers.runner import run_cli  # noqa: F401

BINARY = "codex"


class CodexProvider(Provider):
    name = "codex"
    capabilities = frozenset()

    def model_label(self, routed_model: str) -> str:
        raise NotImplementedError("CodexProvider.model_label")

    def require_auth(self) -> None:
        raise NotImplementedError("CodexProvider.require_auth")

    def complete(self, request: dict):
        raise NotImplementedError("CodexProvider.complete")

    def summarize(self, instructions: str):
        raise NotImplementedError("CodexProvider.summarize")


PROVIDER = CodexProvider()
