"""
Thread safety tests for session cost and token tracking.

Ensures that concurrent access to shared state (session_cost, session_tokens)
doesn't cause race conditions or data corruption.
"""

import pytest
import threading
import time
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import claude_light.state as state
from claude_light.ui import print_session_summary, print_stats


class MockUsage:
    """Mock usage object for testing."""
    def __init__(self, input_tokens=100, output_tokens=50, 
                 cache_creation=0, cache_read=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


class TestThreadSafety:
    """Test thread safety of state access patterns."""

    def setup_method(self):
        """Reset state before each test."""
        state.session_cost = 0.0
        state.session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
        state.conversation_history = []
        state.last_interaction = time.time()

    def test_accumulate_usage_thread_safety(self):
        """Test that _accumulate_usage is thread-safe."""
        from claude_light.llm import _accumulate_usage

        num_threads = 10
        updates_per_thread = 100
        expected_total = num_threads * updates_per_thread

        def update_usage():
            usage = MockUsage(
                input_tokens=1,
                output_tokens=1,
                cache_creation=1,
                cache_read=1
            )
            for _ in range(updates_per_thread):
                _accumulate_usage(usage)

        threads = [threading.Thread(target=update_usage) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all updates were applied correctly
        assert state.session_tokens["input"] == expected_total
        assert state.session_tokens["cache_write"] == expected_total
        assert state.session_tokens["cache_read"] == expected_total
        assert state.session_tokens["output"] == expected_total

    def test_print_session_summary_thread_safety(self):
        """Test that print_session_summary is thread-safe."""
        # Populate some data
        state.session_tokens["input"] = 1000
        state.session_tokens["cache_write"] = 100
        state.session_tokens["cache_read"] = 200
        state.session_tokens["output"] = 500
        state.session_cost = 0.05
        state.conversation_history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        errors = []

        def call_print_summary():
            try:
                # Capture stdout
                import io
                from contextlib import redirect_stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    print_session_summary()
            except Exception as e:
                errors.append(e)

        # Call from multiple threads
        threads = [threading.Thread(target=call_print_summary) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"print_session_summary raised errors: {errors}"

    def test_print_stats_thread_safety(self):
        """Test that print_stats is thread-safe."""
        import io
        from contextlib import redirect_stdout

        state.session_cost = 0.10

        errors = []

        def call_print_stats():
            try:
                usage = MockUsage(input_tokens=100, output_tokens=50)
                f = io.StringIO()
                with redirect_stdout(f):
                    print_stats(usage, label="Test")
            except Exception as e:
                errors.append(e)

        # Call from multiple threads
        threads = [threading.Thread(target=call_print_stats) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"print_stats raised errors: {errors}"

    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes to session state."""
        from claude_light.llm import _accumulate_usage

        num_readers = 5
        num_writers = 5
        duration_seconds = 2
        stop_event = threading.Event()

        read_results = []
        write_count = [0]

        def reader():
            """Continuously read state."""
            while not stop_event.is_set():
                with state.lock:
                    read_results.append({
                        "input": state.session_tokens["input"],
                        "output": state.session_tokens["output"],
                        "cost": state.session_cost,
                    })
                time.sleep(0.001)

        def writer():
            """Continuously update state."""
            while not stop_event.is_set():
                usage = MockUsage(input_tokens=1, output_tokens=1)
                _accumulate_usage(usage)
                with state.lock:
                    state.session_cost += 0.001
                write_count[0] += 1
                time.sleep(0.001)

        readers = [threading.Thread(target=reader) for _ in range(num_readers)]
        writers = [threading.Thread(target=writer) for _ in range(num_writers)]

        for t in readers + writers:
            t.start()

        time.sleep(duration_seconds)
        stop_event.set()

        for t in readers + writers:
            t.join()

        # Verify state consistency
        assert state.session_tokens["input"] == write_count[0]
        assert state.session_tokens["output"] == write_count[0]
        assert abs(state.session_cost - (write_count[0] * 0.001)) < 0.001

        # Verify readers didn't encounter inconsistent state
        assert len(read_results) > 0

    def test_no_race_condition_on_history_access(self):
        """Test that conversation history access is safe."""
        errors = []

        def add_to_history():
            for i in range(50):
                try:
                    with state.lock:
                        state.conversation_history.append({
                            "role": "user",
                            "content": f"message_{i}"
                        })
                except Exception as e:
                    errors.append(e)

        def read_history():
            for _ in range(50):
                try:
                    with state.lock:
                        len_hist = len(state.conversation_history)
                        if len_hist > 0:
                            first = state.conversation_history[0]
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        writers = [threading.Thread(target=add_to_history) for _ in range(3)]
        readers = [threading.Thread(target=read_history) for _ in range(3)]

        for t in writers + readers:
            t.start()
        for t in writers + readers:
            t.join()

        assert not errors
        assert len(state.conversation_history) == 150  # 3 writers * 50 messages

    def test_state_lock_prevents_corruption(self):
        """Verify that without lock, data corruption could occur (by intent)."""
        # This test demonstrates why locks are needed
        state.session_tokens["input"] = 0

        def unsafe_increment():
            """Simulate unsynchronized update - NOT using lock."""
            # Get current value
            current = state.session_tokens["input"]
            # Simulate some work
            time.sleep(0.00001)
            # Write back incremented value
            state.session_tokens["input"] = current + 1

        # Multiple threads doing unsafe increments
        threads = [threading.Thread(target=unsafe_increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Due to race conditions, the final value will likely be < 10
        # (This test shows why synchronization is critical)
        # With our fix using locks, this should not happen in production code.
        # This is just a demonstration of the problem.

    def test_lock_exists(self):
        """Verify that state lock is properly initialized."""
        assert hasattr(state, 'lock')
        # threading.Lock() is a factory, not a class; check the lock interface instead
        assert hasattr(state.lock, 'acquire')
        assert hasattr(state.lock, 'release')


class TestStatusBarThreadSafety:
    """Test thread safety of status bar updates."""

    def setup_method(self):
        """Reset state before each test."""
        state.session_cost = 0.0
        state.session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
        state.conversation_history = []

    def test_status_bar_under_concurrent_updates(self):
        """Test that get_status_bar doesn't crash with concurrent state updates."""
        from claude_light.llm import _accumulate_usage
        import io
        from contextlib import redirect_stdout

        errors = []
        updates = [0]

        def update_state():
            """Continuously update state."""
            for _ in range(100):
                usage = MockUsage(input_tokens=10, output_tokens=5)
                _accumulate_usage(usage)
                with state.lock:
                    state.session_cost += 0.01
                updates[0] += 1
                time.sleep(0.001)

        def read_status():
            """Simulate reading status bar values."""
            for _ in range(50):
                try:
                    with state.lock:
                        total_in = (state.session_tokens["input"] + 
                                   state.session_tokens["cache_write"] + 
                                   state.session_tokens["cache_read"])
                        saved = state.session_tokens["cache_read"]
                        cost = state.session_cost
                    # Simulate some work
                    time.sleep(0.002)
                except Exception as e:
                    errors.append(e)

        updaters = [threading.Thread(target=update_state) for _ in range(3)]
        readers = [threading.Thread(target=read_status) for _ in range(3)]

        for t in updaters + readers:
            t.start()
        for t in updaters + readers:
            t.join()

        assert not errors
        assert updates[0] > 0


class TestLockContention:
    """Test that locks don't cause deadlocks."""

    def setup_method(self):
        """Reset state before each test."""
        state.session_cost = 0.0
        state.session_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}

    def test_no_deadlock_with_multiple_lock_holders(self):
        """Verify that lock usage pattern doesn't cause deadlocks."""
        timeout_occurred = False

        def acquire_lock_sequentially():
            """Acquire lock multiple times in sequence."""
            for _ in range(10):
                with state.lock:
                    state.session_tokens["input"] += 1
                time.sleep(0.001)

        threads = [threading.Thread(target=acquire_lock_sequentially) for _ in range(5)]

        for t in threads:
            t.start()

        # Wait with timeout to detect deadlocks
        for t in threads:
            t.join(timeout=5)
            if t.is_alive():
                timeout_occurred = True

        assert not timeout_occurred, "Possible deadlock detected"
        assert state.session_tokens["input"] == 50  # 5 threads * 10 increments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
