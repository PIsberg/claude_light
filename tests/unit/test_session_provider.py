"""
Tests for per-provider session files (specs/001-gemini-codex-providers: T007).

Covers FR-016 and the --continue edge case (data-model.md rules, message from
contracts/provider.md). The active provider is faked on claude_light.providers.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401
import claude_light.state as state
from claude_light import config, providers, session

_CLAUDE_HISTORY = [
    {"role": "user", "content": "claude question"},
    {"role": "assistant", "content": "claude answer"},
]
_GEMINI_HISTORY = [
    {"role": "user", "content": "gemini question"},
    {"role": "assistant", "content": "gemini answer"},
]


class _ProviderSessionTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="cl_sessions_prov_"))
        patcher = patch.object(config, "SESSION_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._reset_state()
        self.addCleanup(self._reset_state)

    def _reset_state(self):
        state.conversation_history.clear()
        state.turn_count = 0
        state.session_model = None

    def use(self, name):
        p = patch.object(providers, "active", lambda: types.SimpleNamespace(name=name))
        p.start()
        self.addCleanup(p.stop)
        return p

    def save_as(self, name, history):
        """Save `history` while `name` is the active provider, then reset state."""
        with patch.object(providers, "active", lambda: types.SimpleNamespace(name=name)):
            state.conversation_history[:] = [dict(m) for m in history]
            state.turn_count = len(history) // 2
            session.save()
            self._reset_state()

    def file_for(self, name):
        with patch.object(providers, "active", lambda: types.SimpleNamespace(name=name)):
            return session.session_file()


class TestSessionFileNames(_ProviderSessionTest):

    def test_fr016_default_provider_keeps_the_existing_file_name(self):
        claude = self.file_for(providers.DEFAULT)
        self.assertRegex(claude.name, r"^[0-9a-f]{16}\.json$")

    def test_fr016_other_providers_get_their_own_file_next_to_it(self):
        claude = self.file_for(providers.DEFAULT)
        gemini = self.file_for("gemini")
        codex = self.file_for("codex")
        self.assertEqual(gemini.name, claude.name[:-len(".json")] + ".gemini.json")
        self.assertEqual(codex.name, claude.name[:-len(".json")] + ".codex.json")
        self.assertEqual(gemini.parent, claude.parent)

    def test_fr016_explicit_provider_argument_wins_over_the_active_one(self):
        self.use("gemini")
        self.assertEqual(session.session_file(provider="codex"), self.file_for("codex"))


class TestSaveKeepsProvidersApart(_ProviderSessionTest):

    def test_fr016_saved_claude_file_is_byte_identical_after_a_gemini_restore(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        claude_file = self.file_for("claude")
        before = claude_file.read_bytes()
        mtime = os.stat(claude_file).st_mtime_ns

        self.use("gemini")
        message = session.restore()

        self.assertEqual(claude_file.read_bytes(), before)
        self.assertEqual(os.stat(claude_file).st_mtime_ns, mtime)
        self.assertEqual(state.conversation_history, [])
        self.assertEqual(state.turn_count, 0)
        self.assertIn("claude", message)

    def test_fr016_after_a_gemini_save_claude_restore_returns_its_own_history(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        self.save_as("gemini", _GEMINI_HISTORY)

        self.use("claude")
        session.restore()
        self.assertEqual(state.conversation_history, _CLAUDE_HISTORY)

    def test_fr016_gemini_restore_returns_gemini_history_when_both_exist(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        self.save_as("gemini", _GEMINI_HISTORY)

        self.assertNotEqual(self.file_for("gemini"), self.file_for("claude"))
        self.assertTrue(self.file_for("gemini").is_file())
        self.use("gemini")
        session.restore()
        self.assertEqual(state.conversation_history, _GEMINI_HISTORY)

    def test_fr016_clear_touches_only_the_active_providers_file(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        self.save_as("gemini", _GEMINI_HISTORY)
        self.use("gemini")
        session.clear()
        self.assertTrue(self.file_for("claude").is_file())
        self.assertFalse(self.file_for("gemini").is_file())


class TestMismatchWarning(_ProviderSessionTest):

    def test_fr016_message_names_saved_and_chosen_provider_and_the_remedy(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        self.use("gemini")
        message = session.restore()
        self.assertEqual(
            message,
            "The saved conversation was made with claude; starting a fresh gemini "
            "conversation. Resume it with --provider claude --continue.")

    def test_fr016_with_two_other_files_the_newer_by_mtime_is_named(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        self.save_as("codex", _GEMINI_HISTORY)
        old, new = self.file_for("claude"), self.file_for("codex")
        os.utime(old, (1_000_000_000, 1_000_000_000))
        os.utime(new, (2_000_000_000, 2_000_000_000))
        self.use("gemini")
        self.assertIn("made with codex", session.restore())

        os.utime(old, (3_000_000_000, 3_000_000_000))
        self.assertIn("made with claude", session.restore())

    def test_fr016_nothing_saved_keeps_the_existing_message(self):
        self.use("gemini")
        self.assertIn("No previous conversation", session.restore())

    def test_fr016_mismatch_opens_no_file_for_writing_and_changes_no_state(self):
        self.save_as("claude", _CLAUDE_HISTORY)
        self.use("gemini")
        state.session_model = "keep-me"
        real_open = os.open
        writes = []

        def spy(path, flags, *a, **k):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                writes.append(path)
            return real_open(path, flags, *a, **k)

        with patch("os.open", spy):
            session.restore()
        self.assertEqual(writes, [])
        self.assertEqual(state.session_model, "keep-me")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         [self.file_for("claude").name])


if __name__ == "__main__":
    unittest.main()
