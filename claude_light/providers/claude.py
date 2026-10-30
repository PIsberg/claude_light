"""Claude provider: the anthropic SDK (API_KEY) or the claude CLI (OAUTH)."""
import sys

from claude_light import config
from claude_light.providers import base
from claude_light.providers.base import Provider


class ClaudeProvider(Provider):
    name = "claude"
    capabilities = frozenset({base.CACHING, base.EFFORT, base.TOP_TIER,
                              base.USD_BUDGET, base.USAGE, base.AGENT, base.SHELL})

    def model_label(self, routed_model: str) -> str:
        return routed_model

    def require_auth(self) -> None:
        """Exit if no credential was found. main() calls it once the arguments
        are parsed, not at import, so --help works on a fresh install (#101)."""
        from claude_light.ui import _T_ERR, _T_SYS, _ANSI_CYAN, _ANSI_RESET, _ANSI_BOLD
        if not config.AUTH_MODE or (config.AUTH_MODE == "OAUTH"
                                    and not config.AUTH_TOKEN and not config.API_KEY):
            print(f"\n  {_T_ERR}  No authentication found.", file=sys.stderr)
            print(f"  {_T_SYS}  To use your Claude Pro subscription:", file=sys.stderr)
            print(f"          {_ANSI_CYAN}claude auth login{_ANSI_RESET}\n", file=sys.stderr)
            print(f"  {_T_SYS}  Or set {_ANSI_BOLD}ANTHROPIC_API_KEY{_ANSI_RESET} in your environment.", file=sys.stderr)
            sys.exit(1)

    def complete(self, request: dict):
        from claude_light import llm
        return llm._make_streaming_api_call(**request)

    def summarize(self, instructions: str):
        from claude_light import llm
        from claude_light.retry import retry_with_backoff
        if llm.AUTH_MODE == "OAUTH":
            # No API key in this mode: the summary goes through the CLI too.
            text, usage, _ = llm._make_cli_subprocess_call(
                instructions, model=llm.SUMMARY_MODEL,
                extra_flags=["--system-prompt", "You summarize conversations."],
                render=False,
                tools=False,  # the turns may quote repo text; nothing may act on it
            )
            return text.strip(), usage

        @retry_with_backoff
        def _call_api():
            return llm.client.messages.create(
                model=llm.SUMMARY_MODEL,
                # Thinking is always on and shares this cap with the summary.
                max_tokens=4_000,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": instructions}],
            )

        response = _call_api()
        return response.content[0].text.strip(), response.usage


PROVIDER = ClaudeProvider()
