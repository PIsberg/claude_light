"""Benchmark scripts for claude_light.

This package contains benchmark scripts that measure performance, cost,
and quality metrics. These are NOT standard pytest tests but can be
run directly for performance analysis.

Scripts:
- benchmark.py: Analytical token savings benchmark (no API calls)
- benchmark_retrieval.py: RAG retrieval quality benchmark
- benchmark_cost.py: Real-world cost benchmark
- benchmark_claude_code.py: Code quality analysis benchmark

To run benchmarks, use: python -m tests.benchmarks.<script_name>
"""
