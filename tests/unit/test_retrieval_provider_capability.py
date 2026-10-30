"""
Retrieval under a provider without the EFFORT capability (FR-003, US1).

Gemini and Codex run with tools off, so a low-effort summary-only context can
never be followed up with fetch_chunk. They must get the medium-effort render.
Claude (which has EFFORT) keeps the summary-only low-effort render.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import claude_light  # noqa: F401  (loads the package before the submodules)
from claude_light import retrieval, state
from claude_light.providers import base
from claude_light.providers.claude import ClaudeProvider

_BODY = "def place(self, order):\n    return self.repo.save(order_unique_marker)"
_TEXT = "// order.py\n    // ...\n" + _BODY


class _NoCapProvider(base.Provider):
    name = "nocap"
    capabilities = frozenset()

    def complete(self, request):  # pragma: no cover
        raise NotImplementedError

    def summarize(self, instructions):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(state, "chunk_store",
                        {"order.py::place": {"text": _TEXT}})
    monkeypatch.setattr(retrieval.config if hasattr(retrieval, "config") else
                        __import__("claude_light.config", fromlist=["x"]),
                        "CALL_GRAPH", False)
    return [("order.py::place", 0.9)]


def test_fr003_low_effort_without_effort_capability_includes_body(monkeypatch, store):
    monkeypatch.setattr(state, "provider", _NoCapProvider())
    out = retrieval._finish(store, 4000, "low")[0]
    assert "Detailed Code Context" in out
    assert "order_unique_marker" in out


def test_fr003_low_effort_under_claude_stays_summary_only(monkeypatch, store):
    monkeypatch.setattr(state, "provider", ClaudeProvider())
    out = retrieval._finish(store, 4000, "low")[0]
    assert "Detailed Code Context" not in out
    assert "order_unique_marker" not in out
    assert "place" in out
