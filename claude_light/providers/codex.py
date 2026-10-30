"""Codex provider: the `codex` CLI as a subprocess (stub)."""
from claude_light.providers.base import Provider

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
