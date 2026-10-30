"""Shared provider interface."""

CACHING, EFFORT, TOP_TIER, USD_BUDGET, USAGE, AGENT, SHELL = (
    "caching", "effort", "top_tier", "usd_budget", "usage", "agent", "shell")


class ProviderError(Exception):
    """str(e) names the provider and the remedy."""


class NotLoggedIn(ProviderError):
    """The tool needs a login."""


class Provider:
    name: str = ""
    capabilities: frozenset = frozenset()

    def has(self, capability: str) -> bool:
        raise NotImplementedError("Provider.has")

    def model_label(self, routed_model: str) -> str:
        raise NotImplementedError("Provider.model_label")

    def header(self):
        raise NotImplementedError("Provider.header")

    def require_auth(self) -> None:
        raise NotImplementedError("Provider.require_auth")

    def complete(self, request: dict):
        raise NotImplementedError("Provider.complete")

    def summarize(self, instructions: str):
        raise NotImplementedError("Provider.summarize")

    def offer_login(self, retry) -> None:
        raise NotImplementedError("Provider.offer_login")

    def warm(self, quiet: bool = False) -> None:
        raise NotImplementedError("Provider.warm")


def transcript(request: dict) -> tuple:
    raise NotImplementedError("transcript")
