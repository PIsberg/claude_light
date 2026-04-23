"""Tests for claude_light/compressor.py (LLMLingua-2 integration)."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from claude_light import compressor


def _reset_singleton():
    compressor._compressor = None
    compressor._load_error = None
    compressor._load_attempted = False
    compressor._load_thread = None
    compressor._load_done.clear()


def _mark_loaded():
    """Mark the background loader as complete for tests that exercise the
    post-load path without actually loading a model."""
    compressor._load_done.set()


class TestCompressContext(unittest.TestCase):
    def setUp(self):
        _reset_singleton()

    def test_empty_input_is_noop(self):
        out, info = compressor.compress_context("")
        self.assertEqual(out, "")
        self.assertTrue(info["skipped"])
        self.assertEqual(info["reason"], "empty")

    def test_disabled_by_config(self):
        text = "hello " * 1000
        with patch.object(compressor.config, "LLMLINGUA_ENABLED", False):
            out, info = compressor.compress_context(text)
        self.assertEqual(out, text)
        self.assertTrue(info["skipped"])
        self.assertEqual(info["reason"], "disabled")

    def test_short_input_below_min_tokens_is_skipped(self):
        # 100 chars ≈ 25 tokens, well below LLMLINGUA_MIN_TOKENS (800)
        with patch.object(compressor.config, "LLMLINGUA_ENABLED", True):
            out, info = compressor.compress_context("short text " * 10)
        self.assertTrue(info["skipped"])
        self.assertEqual(info["reason"], "below_min_tokens")

    def test_fallback_when_llmlingua_missing(self):
        # Simulate ImportError on first get_compressor() call.
        long_text = "x" * 10_000  # ~2500 tokens estimated
        with patch.object(compressor.config, "LLMLINGUA_ENABLED", True), \
             patch.dict(sys.modules, {"llmlingua": None}):
            # Force the load attempt synchronously so _load_done is set.
            compressor.get_compressor()
            out, info = compressor.compress_context(long_text)
        self.assertEqual(out, long_text)
        self.assertTrue(info["skipped"])
        self.assertEqual(info["reason"], "llmlingua_not_installed")

    def test_skipped_when_loader_not_ready(self):
        # Enabled + long input + loader thread hasn't finished → should skip,
        # NOT block waiting for the model.
        long_text = "x" * 10_000
        with patch.object(compressor.config, "LLMLINGUA_ENABLED", True), \
             patch.object(compressor, "start_background_load") as mock_start:
            # Ensure _load_done is clear (default after reset).
            out, info = compressor.compress_context(long_text)
        self.assertEqual(out, long_text)
        self.assertTrue(info["skipped"])
        self.assertEqual(info["reason"], "still_loading")
        mock_start.assert_called_once()

    def test_mocked_compressor_path(self):
        long_text = "x" * 10_000
        fake_result = {
            "compressed_prompt": "xxx compressed xxx",
            "origin_tokens": 2500,
            "compressed_tokens": 1000,
        }
        fake_comp = MagicMock()
        fake_comp.compress_prompt.return_value = fake_result

        with patch.object(compressor.config, "LLMLINGUA_ENABLED", True), \
             patch.object(compressor, "get_compressor", return_value=fake_comp):
            _mark_loaded()
            out, info = compressor.compress_context(long_text, rate=0.4)

        fake_comp.compress_prompt.assert_called_once()
        self.assertEqual(out, "xxx compressed xxx")
        self.assertFalse(info["skipped"])
        self.assertEqual(info["tokens_before"], 2500)
        self.assertEqual(info["tokens_after"], 1000)
        self.assertAlmostEqual(info["ratio"], 0.4)
        self.assertGreaterEqual(info["elapsed_ms"], 0.0)

    def test_runtime_error_returns_original(self):
        long_text = "x" * 10_000
        fake_comp = MagicMock()
        fake_comp.compress_prompt.side_effect = RuntimeError("boom")

        with patch.object(compressor.config, "LLMLINGUA_ENABLED", True), \
             patch.object(compressor, "get_compressor", return_value=fake_comp):
            _mark_loaded()
            out, info = compressor.compress_context(long_text)

        self.assertEqual(out, long_text)
        self.assertTrue(info["skipped"])
        self.assertTrue(info["reason"].startswith("runtime_error:RuntimeError"))

    def test_no_gain_returns_original(self):
        long_text = "x" * 10_000
        # LLMLingua returned same-or-longer output → treat as no-op.
        fake_result = {
            "compressed_prompt": long_text + "!",
            "origin_tokens": 2500,
            "compressed_tokens": 2500,
        }
        fake_comp = MagicMock()
        fake_comp.compress_prompt.return_value = fake_result

        with patch.object(compressor.config, "LLMLINGUA_ENABLED", True), \
             patch.object(compressor, "get_compressor", return_value=fake_comp):
            _mark_loaded()
            out, info = compressor.compress_context(long_text)

        self.assertEqual(out, long_text)
        self.assertTrue(info["skipped"])
        self.assertEqual(info["reason"], "no_gain")


class TestAccumulateCompressionStats(unittest.TestCase):
    """Sanity-check the stats accumulator wired into llm.py."""

    def test_skipped_info_is_noop(self):
        from claude_light import llm, state
        before = dict(state.global_stats)
        llm._accumulate_compression_stats({"skipped": True, "reason": "disabled"})
        self.assertEqual(state.global_stats, before)

    def test_accumulates_delta_and_dollars(self):
        from claude_light import llm, state
        from claude_light.config import PRICE_WRITE
        with state.lock:
            pre_before  = state.global_stats["total_tokens_pre_compress"]
            post_before = state.global_stats["total_tokens_post_compress"]
            d_before    = state.global_stats["total_dollars_saved_llmlingua"]

        info = {"skipped": False, "tokens_before": 3000, "tokens_after": 1200}
        llm._accumulate_compression_stats(info)

        with state.lock:
            self.assertEqual(state.global_stats["total_tokens_pre_compress"],  pre_before  + 3000)
            self.assertEqual(state.global_stats["total_tokens_post_compress"], post_before + 1200)
            expected_d = d_before + (1800 / 1_000_000.0) * PRICE_WRITE
            self.assertAlmostEqual(state.global_stats["total_dollars_saved_llmlingua"], expected_d)


if __name__ == "__main__":
    unittest.main()
