"""Provider registry: one module per provider, imported on first use."""

NAMES: tuple[str, ...] = ("claude", "gemini", "codex")
DEFAULT: str = "claude"


def get(name: str):
    raise NotImplementedError("providers.get")


def select(name: str):
    raise NotImplementedError("providers.select")


def active():
    raise NotImplementedError("providers.active")
