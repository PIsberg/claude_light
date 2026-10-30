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
        return capability in self.capabilities

    def model_label(self, routed_model: str) -> str:
        raise NotImplementedError("Provider.model_label")

    def header(self):
        return None

    def require_auth(self) -> None:
        raise NotImplementedError("Provider.require_auth")

    def complete(self, request: dict):
        raise NotImplementedError("Provider.complete")

    def summarize(self, instructions: str):
        raise NotImplementedError("Provider.summarize")

    def offer_login(self, retry) -> None:
        return None

    def warm(self, quiet: bool = False) -> None:
        return None


def _text_of(content) -> str:
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    return content or ""


def transcript(request: dict) -> tuple:
    """Flatten a request into (system text, prompt) for a CLI that takes text."""
    system_text = "\n".join(
        b.get("text", "") for b in request.get("system", []) if b.get("text"))
    messages = request.get("messages", [])
    history_parts = []
    for msg in messages[:-1]:
        role = msg["role"].capitalize()
        history_parts.append(f"--- {role} ---\n{_text_of(msg.get('content'))}")
    history_text = "\n".join(history_parts)
    last_user_text = _text_of(messages[-1]["content"]) if messages else ""
    prompt = (
        (f"CONVERSATION HISTORY:\n{history_text}\n\n" if history_text else "")
        + f"USER: {last_user_text}"
    )
    return system_text, prompt
