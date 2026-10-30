"""Gemini provider: the `gemini` CLI as a subprocess (stub)."""
from claude_light.providers.base import Provider

BINARY = "gemini"


class GeminiProvider(Provider):
    name = "gemini"
    capabilities = frozenset()

    def model_label(self, routed_model: str) -> str:
        raise NotImplementedError("GeminiProvider.model_label")

    def require_auth(self) -> None:
        raise NotImplementedError("GeminiProvider.require_auth")

    def complete(self, request: dict):
        raise NotImplementedError("GeminiProvider.complete")

    def summarize(self, instructions: str):
        raise NotImplementedError("GeminiProvider.summarize")


PROVIDER = GeminiProvider()
