"""
Fuzz tests for claude_light using Hypothesis.

This module provides property-based testing to find edge cases
in the core functionality of claude_light.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# Import functions to test - adjust based on actual module structure
try:
    from claude_light import (
        _build_compressed_tree,
        _dedup_retrieved_context,
        route_query,
    )
except ImportError:
    # Functions may be internal - skip if not importable
    pass


class TestFuzzCompressedTree:
    """Fuzz tests for _build_compressed_tree function."""

    @given(
        st.lists(
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs", "Cc"),
                    blacklist_characters={"/", "\\", ":", "*", "?", '"', "<", ">", "|"}
                ),
                min_size=1,
                max_size=50
            ),
            min_size=0,
            max_size=20
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_build_compressed_tree_with_random_paths(self, paths):
        """Test that _build_compressed_tree handles arbitrary path lists."""
        try:
            result = _build_compressed_tree(paths)
            # Result should be a string
            assert isinstance(result, str)
            # Result should not be excessively long
            assert len(result) < 100000
        except NameError:
            pytest.skip("Function not available")


class TestFuzzRouteQuery:
    """Fuzz tests for route_query function."""

    @given(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
            min_size=1,
            max_size=1000
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_route_query_with_random_strings(self, query):
        """Test that route_query handles arbitrary input strings."""
        try:
            result = route_query(query)
            # Result should be a tuple/dict with model info
            assert result is not None
        except NameError:
            pytest.skip("Function not available")


class TestFuzzDedupContext:
    """Fuzz tests for _dedup_retrieved_context function."""

    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=100),  # chunk_id
                st.text(min_size=0, max_size=500),  # content
                st.floats(min_value=0.0, max_value=1.0)  # score
            ),
            min_size=0,
            max_size=50
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dedup_context_with_random_chunks(self, chunks):
        """Test that _dedup_retrieved_context handles arbitrary chunk lists."""
        try:
            result = _dedup_retrieved_context(chunks)
            # Result should be a string
            assert isinstance(result, str)
        except NameError:
            pytest.skip("Function not available")


class TestFuzzChunking:
    """Fuzz tests for chunking functions."""

    @given(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
            min_size=0,
            max_size=10000
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_chunk_python_code_with_random_content(self, code):
        """Test chunking with arbitrary Python-like content."""
        # Test that chunking doesn't crash on arbitrary input
        try:
            from claude_light import chunk_python_file
            chunks = chunk_python_file("test.py", code)
            assert isinstance(chunks, list)
        except (NameError, ImportError):
            pytest.skip("Function not available")
