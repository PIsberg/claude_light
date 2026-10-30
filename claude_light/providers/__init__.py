"""Provider registry: one module per provider, imported on first use."""
import importlib

NAMES: tuple[str, ...] = ("claude", "gemini", "codex")
DEFAULT: str = "claude"


def get(name: str):
    """The provider called `name`; its module is imported on first use."""
    if name not in NAMES:
        raise ValueError(f"unknown provider {name!r}; choose one of: {', '.join(NAMES)}")
    return importlib.import_module(f"claude_light.providers.{name}").PROVIDER


def select(name: str):
    """Make `name` the session's provider and return it."""
    from claude_light import state
    provider = get(name)
    with state.lock:
        state.provider = provider
    return provider


def active():
    """The selected provider, or the default when none was selected."""
    from claude_light import state
    return state.provider or get(DEFAULT)
