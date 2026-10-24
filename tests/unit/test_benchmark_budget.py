"""Tests for the per-effort budget measurement in tests/benchmarks/benchmark_models.py.

The benchmark is not collected by pytest, so these tests load it by path and
check the two things its budget sweep reports: the tokens each question sends
and the number of chunks each budget allows.
"""
import importlib.util
from pathlib import Path

import pytest

import claude_light.config as config
import claude_light.retrieval as retrieval
import claude_light.state as state

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_models.py"


@pytest.fixture
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_models", _BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store(monkeypatch):
    chunks = {
        "a.py::one": {"text": "x" * 400},
        "b.py::two": {"text": "y" * 800},
    }
    monkeypatch.setattr(state, "chunk_store", chunks)
    monkeypatch.setattr(state, "TOP_K", 13)
    return chunks


def test_evaluate_reports_tokens_sent_per_question(bench, store):
    questions = [{"kind": "identifier", "file": "a.py", "question": "q1"}]

    def select(i, k):
        return [("a.py::one", 0.9), ("b.py::two", 0.8)]

    r = bench.evaluate("t", list(store), ["a.py", "b.py"], questions, select)

    expected = sum(retrieval._estimate_tokens(store[c]["text"]) for c in store)
    assert r["tokens_sent"] == [expected]


def test_evaluate_counts_no_tokens_for_unrelated_question(bench, store):
    questions = [{"kind": "unrelated", "file": None, "question": "q"}]

    r = bench.evaluate("t", list(store), ["a.py", "b.py"], questions, lambda i, k: [])

    assert r["tokens_sent"] == [0]
    assert r["unrelated"] == [1, 1]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_evaluate_scales_chunk_count_with_budget(bench, store, effort):
    budget = config._RETRIEVAL_BUDGET[effort]
    seen = []

    def select(i, k):
        seen.append(k)
        return []

    questions = [{"kind": "unrelated", "file": None, "question": "q"}]
    bench.evaluate("t", list(store), ["a.py", "b.py"], questions, select, budget=budget)

    assert seen == [max(2, round(13 * budget / config.TARGET_RETRIEVED_TOKENS))]
