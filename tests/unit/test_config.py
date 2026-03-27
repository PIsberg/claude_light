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
            result = _resolve_api_key()
        self.assertEqual(result, "sk-ant-from-env")

    def test_returns_empty_string_when_no_key(self):
        print("\n  ▶ TestResolveApiKey.test_returns_empty_string_when_no_key")
        orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            import claude_light as cl
            with patch.object(cl, "is_test_mode", False):
                with patch("builtins.open", side_effect=OSError):
                    result = _resolve_api_key()
            self.assertIsInstance(result, str)
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
