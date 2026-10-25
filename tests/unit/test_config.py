"""
Tests for claude_light.config — _resolve_api_key, _load_languages, and constants.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is importable
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.config import _resolve_api_key, _WANTED_LANGS


class TestResolveApiKey(unittest.TestCase):
    """Tests for _resolve_api_key — env var and dotfile paths."""

    def test_returns_env_key(self):
        print("\n  ▶ TestResolveApiKey.test_returns_env_key")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}):
            api_key, auth_mode, source, auth_token = _resolve_api_key()
        self.assertEqual(api_key, "sk-ant-from-env")
        self.assertEqual(auth_mode, "API_KEY")
        self.assertEqual(source, "Environment")

    def test_returns_empty_string_when_no_key(self):
        print("\n  ▶ TestResolveApiKey.test_returns_empty_string_when_no_key")
        orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            import claude_light as cl
            with patch.object(cl, "is_test_mode", False):
                with patch("builtins.open", side_effect=OSError):
                    api_key, auth_mode, source, auth_token = _resolve_api_key()
            self.assertIsInstance(api_key, str)
        finally:
            if orig is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig


class TestLoadLanguages(unittest.TestCase):
    """Tests for language config and indexable extensions."""

    def test_lang_config_populated(self):
        print("\n  ▶ TestLoadLanguages.test_lang_config_populated")
        import claude_light as cl
        self.assertIn(".java", cl._LANG_CONFIG)
        self.assertIn(".py", cl._LANG_CONFIG)

    def test_indexable_extensions_not_empty(self):
        print("\n  ▶ TestLoadLanguages.test_indexable_extensions_not_empty")
        import claude_light as cl
        self.assertGreater(len(cl.INDEXABLE_EXTENSIONS), 0)

    def test_load_languages_no_treesitter(self):
        print("\n  ▶ TestLoadLanguages.test_load_languages_no_treesitter")
        from claude_light import _load_languages
        import claude_light as cl
        orig_config = dict(cl._LANG_CONFIG)
        orig_ext = set(cl.INDEXABLE_EXTENSIONS)
        try:
            cl._LANG_CONFIG.clear()
            cl.INDEXABLE_EXTENSIONS.clear()
            with patch.object(cl, "_TREESITTER_AVAILABLE", False):
                _load_languages()
            for ext in _WANTED_LANGS:
                self.assertIn(ext, cl._LANG_CONFIG)
                self.assertIsNone(cl._LANG_CONFIG[ext])
        finally:
            cl._LANG_CONFIG.clear()
            cl._LANG_CONFIG.update(orig_config)
            cl.INDEXABLE_EXTENSIONS.clear()
            cl.INDEXABLE_EXTENSIONS.update(orig_ext)


if __name__ == "__main__":
    unittest.main()


class TestGlobalStatsFile(unittest.TestCase):
    """The suite once wrote its fake token usage into the developer's real
    ~/.claude_light_stats.json, inflating the lifetime savings it reports."""

    def test_the_test_suite_does_not_use_the_real_stats_file(self):
        print("\n  ▶ TestGlobalStatsFile.test_the_test_suite_does_not_use_the_real_stats_file")
        from claude_light import config, state
        real = Path.home() / ".claude_light_stats.json"
        self.assertNotEqual(config.GLOBAL_STATS_FILE.resolve(), real.resolve())
        self.assertNotEqual(state.GLOBAL_STATS_FILE.resolve(), real.resolve())

    def test_env_var_moves_the_stats_file(self):
        print("\n  ▶ TestGlobalStatsFile.test_env_var_moves_the_stats_file")
        import subprocess
        import tempfile
        target = Path(tempfile.mkdtemp()) / "stats.json"
        env = dict(os.environ, CLAUDE_LIGHT_STATS_FILE=str(target))
        out = subprocess.run(
            [sys.executable, "-c", "import claude_light.config as c; print(c.GLOBAL_STATS_FILE)"],
            env=env, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]),
        )
        self.assertEqual(out.stdout.strip(), str(target), out.stderr)


class TestMeasuredRetrievalDefaults(unittest.TestCase):
    """Both defaults come from tests/benchmarks/benchmark_models.py runs recorded
    in TOKEN-SAVINGS.md: call-graph expansion sent 20-28% more tokens for no
    extra right-file hits, and a working-tree boost of 1.5 or more lost
    answers when an unrelated file was the modified one."""

    def test_defaults_without_env_overrides(self):
        print("\n  ▶ TestMeasuredRetrievalDefaults.test_defaults_without_env_overrides")
        import subprocess
        env = {k: v for k, v in os.environ.items()
               if k not in ("CLAUDE_LIGHT_CALLGRAPH", "CLAUDE_LIGHT_WORKTREE_BOOST")}
        out = subprocess.run(
            [sys.executable, "-c",
             "import claude_light.config as c; print(c.CALL_GRAPH, c.WORKTREE_BOOST)"],
            env=env, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]),
        )
        self.assertEqual(out.stdout.strip(), "False 1.0", out.stderr)

    def test_call_graph_is_opt_in(self):
        print("\n  ▶ TestMeasuredRetrievalDefaults.test_call_graph_is_opt_in")
        import subprocess
        env = dict(os.environ, CLAUDE_LIGHT_CALLGRAPH="1")
        out = subprocess.run(
            [sys.executable, "-c", "import claude_light.config as c; print(c.CALL_GRAPH)"],
            env=env, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]),
        )
        self.assertEqual(out.stdout.strip(), "True", out.stderr)


class TestRouteOnceDefault(unittest.TestCase):
    """The first question's model is kept by default; CLAUDE_LIGHT_ROUTE_ONCE=0
    restores per-turn model routing."""

    def _route_once(self, value=None):
        import subprocess
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_LIGHT_ROUTE_ONCE"}
        if value is not None:
            env["CLAUDE_LIGHT_ROUTE_ONCE"] = value
        out = subprocess.run(
            [sys.executable, "-c", "import claude_light.config as c; print(c.ROUTE_ONCE)"],
            env=env, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]),
        )
        return out.stdout.strip() or out.stderr

    def test_on_without_env_override(self):
        self.assertEqual(self._route_once(), "True")

    def test_zero_turns_it_off(self):
        self.assertEqual(self._route_once("0"), "False")
