"""CLI subprocess runner shared by the CLI providers."""
import os
import subprocess

from claude_light.providers.base import ProviderError

TIMEOUT_SECS = 600


def run_cli(provider: str, argv: list, stdin_text: str, timeout: float = TIMEOUT_SECS) -> str:
    """Run a provider CLI, feed it stdin_text and return its stdout.

    A non-zero exit, a timeout or a missing binary raise ProviderError naming
    the provider. KeyboardInterrupt stops the whole process tree and is re-raised.
    """
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise ProviderError(f"{provider} CLI not found. Install it and log in, "
                            f"or choose another --provider.")
    try:
        stdout, stderr = proc.communicate(stdin_text, timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        raise ProviderError(f"{provider} timed out after {timeout:g} s. "
                            f"Run {provider} on its own to see why.")
    except KeyboardInterrupt:
        kill_process_tree(proc)
        raise
    if proc.returncode != 0:
        lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
        last = lines[-1] if lines else "no error output"
        raise ProviderError(f"{provider} failed (exit {proc.returncode}): {last}. "
                            f"Run {provider} on its own to see the full error.")
    return stdout


def kill_process_tree(proc) -> None:
    """Stop the CLI and everything it started.

    On Windows the CLI runs via cmd.exe (shell=True), so proc.kill() only
    ends the shell: the CLI keeps running, keeps editing files, and keeps
    the stdout pipe open so the read loop never finishes.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, stdin=subprocess.DEVNULL,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    # Reap it, so it does not linger as a zombie on POSIX.
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
