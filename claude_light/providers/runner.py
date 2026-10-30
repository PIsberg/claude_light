"""CLI subprocess runner shared by the CLI providers."""

TIMEOUT_SECS = 600


def run_cli(provider: str, argv: list, stdin_text: str, timeout: float = TIMEOUT_SECS) -> str:
    raise NotImplementedError("run_cli")


def kill_process_tree(proc) -> None:
    raise NotImplementedError("kill_process_tree")
