"""Claude provider."""
from claude_light.providers import base
from claude_light.providers.base import Provider


class ClaudeProvider(Provider):
    name = "claude"
    capabilities = frozenset({base.CACHING, base.EFFORT, base.TOP_TIER,
                              base.USD_BUDGET, base.USAGE, base.AGENT, base.SHELL})

    def model_label(self, routed_model: str) -> str:
        raise NotImplementedError("ClaudeProvider.model_label")

    def header(self):
        raise NotImplementedError("ClaudeProvider.header")

    def require_auth(self) -> None:
        raise NotImplementedError("ClaudeProvider.require_auth")

    def complete(self, request: dict):
        raise NotImplementedError("ClaudeProvider.complete")

    def summarize(self, instructions: str):
        raise NotImplementedError("ClaudeProvider.summarize")

    def offer_login(self, retry) -> None:
        raise NotImplementedError("ClaudeProvider.offer_login")

    def warm(self, quiet: bool = False) -> None:
        raise NotImplementedError("ClaudeProvider.warm")

    def has(self, capability: str) -> bool:
        raise NotImplementedError("ClaudeProvider.has")


PROVIDER = ClaudeProvider()
