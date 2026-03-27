"""
Tests for claude_light.executor — _run_command and auto_tune.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from claude_light.executor import _run_command, _RUN_HEAD_LINES, _RUN_TAIL_LINES, _RUN_MAX_CHARS, auto_tune
import claude_light.state as _state


class TestRunCommand(unittest.TestCase):
    """Tests for shell command execution."""

    def test_run_success(self):
        print("\n  ▶ TestRunCommand.test_run_success")
        result = _run_command('python -c "print(\'hello\')"')
        self.assertIn("exit 0", result)
        self.assertIn("hello", result)

    def test_run_failure_exit_code(self):
        print("\n  ▶ TestRunCommand.test_run_failure_exit_code")
        result = _run_command('python -c "raise SystemExit(1)"')
        self.assertIn("exit 1", result)

    def test_run_truncation(self):
        print("\n  ▶ TestRunCommand.test_run_truncation")
        total = _RUN_HEAD_LINES + _RUN_TAIL_LINES + 50
        result = _run_command(f'python -c "[print(i) for i in range({total})]"')
        self.assertIn("lines omitted", result)

    def test_run_stderr_merged(self):
        print("\n  ▶ TestRunCommand.test_run_stderr_merged")
        result = _run_command('python -c "import sys; sys.stderr.write(\'err_token\\n\')"')
        self.assertIn("err_token", result)

    def test_run_outputs_stderr(self):
        print("\n  ▶ TestRunCommand.test_run_outputs_stderr")
        result = _run_command('python -c "import sys; sys.stderr.write(\'STDERROUT\\n\')"')
        self.assertIn("STDERROUT", result)

    def test_run_char_truncation(self):
        print("\n  ▶ TestRunCommand.test_run_char_truncation")
        result = _run_command(f'python -c "print(\'x\' * {_RUN_MAX_CHARS + 1000})"')
        self.assertIn("truncated", result)

    def test_run_includes_exit_code_in_transcript(self):
        print("\n  ▶ TestRunCommand.test_run_includes_exit_code_in_transcript")
        result = _run_command('python -c "raise SystemExit(42)"')
        self.assertIn("42", result)


class TestAutoTune(unittest.TestCase):
    """Tests for embedding model auto-selection."""

    def _make_files(self, n):
        files = []
        for i in range(n):
            f = MagicMock()
            f.exists.return_value = True
            f.stat.return_value.st_size = 2000
            files.append(f)
        return files

    def test_small_repo_selects_minilm(self):
        print("\n  ▶ TestAutoTune.test_small_repo_selects_minilm")
        files = self._make_files(10)
        _state.EMBED_MODEL = None
        _state.embedder = None
        with patch("claude_light.executor.SentenceTransformer") as mock_st, \
             patch("claude_light.executor._Spinner"):
            mock_st.return_value = MagicMock()
            auto_tune(files, quiet=True)
        self.assertEqual(_state.EMBED_MODEL, "all-MiniLM-L6-v2")

    def test_medium_repo_selects_mpnet(self):
        print("\n  ▶ TestAutoTune.test_medium_repo_selects_mpnet")
        files = self._make_files(100)
        _state.EMBED_MODEL = None
        _state.embedder = None
        with patch("claude_light.executor.SentenceTransformer") as mock_st, \
             patch("claude_light.executor._Spinner"):
            mock_st.return_value = MagicMock()
            auto_tune(files, quiet=True)
        self.assertEqual(_state.EMBED_MODEL, "all-mpnet-base-v2")

    def test_large_repo_selects_nomic(self):
        print("\n  ▶ TestAutoTune.test_large_repo_selects_nomic")
        files = self._make_files(250)
        _state.EMBED_MODEL = None
        _state.embedder = None
        with patch("claude_light.executor.SentenceTransformer") as mock_st, \
             patch("claude_light.executor._Spinner"):
            mock_st.return_value = MagicMock()
            auto_tune(files, quiet=True)
        self.assertEqual(_state.EMBED_MODEL, "nomic-ai/nomic-embed-text-v1.5")

    def test_sets_top_k_from_chunks(self):
        print("\n  ▶ TestAutoTune.test_sets_top_k_from_chunks")
        files = self._make_files(5)
        chunks = [{"text": "a" * 400} for _ in range(5)]
        _state.EMBED_MODEL = None
        _state.embedder = None
        with patch("claude_light.executor.SentenceTransformer") as mock_st, \
             patch("claude_light.executor._Spinner"):
            mock_st.return_value = MagicMock()
            auto_tune(files, chunks=chunks, quiet=True)
        self.assertIsNotNone(_state.TOP_K)
        self.assertGreaterEqual(_state.TOP_K, 2)
        self.assertLessEqual(_state.TOP_K, 15)

    def test_reuses_embedder_if_model_unchanged(self):
        print("\n  ▶ TestAutoTune.test_reuses_embedder_if_model_unchanged")
        files = self._make_files(10)
        mock_embedder = MagicMock()
        _state.EMBED_MODEL = "all-MiniLM-L6-v2"
        _state.embedder = mock_embedder
        with patch("claude_light.executor.SentenceTransformer") as mock_st:
            auto_tune(files, quiet=True)
            mock_st.assert_not_called()
        self.assertIs(_state.embedder, mock_embedder)

    def test_top_k_with_no_chunks(self):
        print("\n  ▶ TestAutoTune.test_top_k_with_no_chunks")
        files = self._make_files(20)
        _state.EMBED_MODEL = None
        _state.embedder = None
        with patch("claude_light.executor.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            auto_tune(files, chunks=None, quiet=True)
        self.assertIsNotNone(_state.TOP_K)


if __name__ == "__main__":
    unittest.main()
