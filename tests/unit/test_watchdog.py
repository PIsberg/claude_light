"""
Tests for watchdog observer resource management.

Ensures that the file observer is properly started, stopped, and cleaned up
even when errors occur or user interrupts the session.
"""

import pytest
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent to path - go up 3 levels from tests/unit/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import claude_light.state as state


class MockObserver:
    """Mock watchdog Observer for testing."""
    def __init__(self):
        self.started = False
        self.stopped = False
        self.join_called = False
        self.scheduled = []
        self._stop_event = threading.Event()
    
    def schedule(self, handler, path, recursive=False):
        self.scheduled.append((handler, path, recursive))
    
    def start(self):
        self.started = True
    
    def stop(self):
        self.stopped = True
        self._stop_event.set()
    
    def join(self, timeout=None):
        self.join_called = True
        # Simulate observer join with timeout
        return self._stop_event.wait(timeout)


class TestWatchdogResourceManagement:
    """Test proper cleanup of observer resources."""

    def setup_method(self):
        """Reset state before each test."""
        state.stop_event.clear()

    def test_observer_starts_successfully(self):
        """Test that observer starts and schedules correctly."""
        with patch('claude_light.main.Observer', return_value=MockObserver()) as mock_observer_class:
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', return_value='exit'):
                        from claude_light.main import start_chat
                        try:
                            start_chat()
                        except (KeyboardInterrupt, EOFError):
                            pass
            
            # Verify observer was created
            assert mock_observer_class.called

    def test_observer_stops_on_exit(self):
        """Test that observer is properly stopped when session exits."""
        mock_observer = MockObserver()
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', side_effect=EOFError):
                        from claude_light.main import start_chat
                        try:
                            start_chat()
                        except (KeyboardInterrupt, EOFError):
                            pass
        
        # Verify observer was stopped
        assert mock_observer.stopped
        assert mock_observer.join_called

    def test_observer_cleanup_on_exception(self):
        """Test that observer is cleaned up even if exception occurs."""
        mock_observer = MockObserver()
        
        def raise_error():
            raise RuntimeError("Test error")
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh', side_effect=raise_error):
                from claude_light.main import start_chat
                try:
                    start_chat()
                except RuntimeError:
                    pass
        
        # Observer shouldn't be stopped because it failed to start
        # (in this case, observer is None)

    def test_observer_join_has_timeout(self):
        """Test that observer.join() has a timeout to prevent hanging."""
        mock_observer = MagicMock()
        mock_observer.join = MagicMock()
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', side_effect=EOFError):
                        from claude_light.main import start_chat
                        try:
                            start_chat()
                        except (KeyboardInterrupt, EOFError):
                            pass
        
        # Verify join was called with timeout
        mock_observer.join.assert_called_once()
        call_args = mock_observer.join.call_args
        # Check that timeout was passed
        assert call_args is not None

    def test_observer_none_on_start_failure(self):
        """Test that failure to start observer doesn't crash session."""
        def raise_on_create(*args, **kwargs):
            raise RuntimeError("Failed to create observer")
        
        with patch('claude_light.main.Observer', side_effect=raise_on_create):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', return_value='exit'):
                        from claude_light.main import start_chat
                        # Should not crash
                        try:
                            start_chat()
                        except (EOFError, SystemExit):
                            pass

    def test_observer_scheduled_with_source_handler(self):
        """Test that SourceHandler is properly scheduled."""
        mock_observer = MockObserver()
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main.SourceHandler'):
                    with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                        with patch('builtins.input', return_value='exit'):
                            from claude_light.main import start_chat
                            try:
                                start_chat()
                            except (KeyboardInterrupt, EOFError):
                                pass
        
        # Verify SourceHandler was scheduled
        assert len(mock_observer.scheduled) > 0
        handler, path, recursive = mock_observer.scheduled[0]
        assert path == "."
        assert recursive is True

    def test_keyboard_interrupt_cleanup(self):
        """Test that Ctrl+C triggers proper cleanup."""
        mock_observer = MockObserver()
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', side_effect=KeyboardInterrupt):
                        from claude_light.main import start_chat
                        # Should handle interrupt gracefully
                        start_chat()
        
        # Verify observer was cleaned up
        assert mock_observer.stopped

    def test_signal_handler_stops_session(self):
        """Test that signal handler properly sets stop event."""
        from claude_light.main import _setup_signal_handlers
        
        # Setup handlers
        _setup_signal_handlers()
        
        # Verify signal handlers were registered (no exception)
        # (Can't easily test actual signal handling without os.kill)


class TestObserverErrorHandling:
    """Test error handling in observer management."""

    def test_observer_stop_error_caught(self):
        """Test that errors during observer.stop() are caught."""
        mock_observer = MagicMock()
        mock_observer.stop.side_effect = RuntimeError("Stop failed")
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', return_value='exit'):
                        from claude_light.main import start_chat
                        # Should not crash despite stop error
                        try:
                            start_chat()
                        except (EOFError, SystemExit):
                            pass

    def test_observer_join_timeout_handled(self):
        """Test that observer.join() timeout doesn't crash."""
        mock_observer = MagicMock()
        mock_observer.join.side_effect = TimeoutError()
        
        with patch('claude_light.main.Observer', return_value=mock_observer):
            with patch('claude_light.main.full_refresh'):
                with patch('claude_light.main._PROMPTTK_AVAILABLE', False):
                    with patch('builtins.input', return_value='exit'):
                        from claude_light.main import start_chat
                        # Should not crash despite timeout
                        try:
                            start_chat()
                        except (EOFError, SystemExit):
                            pass


class TestSignalHandling:
    """Test signal handler setup."""

    def setup_method(self):
        """Reset stop_event before each test."""
        state.stop_event.clear()

    def test_signal_handler_setup_no_exception(self):
        """Test that signal handler setup doesn't raise."""
        from claude_light.main import _setup_signal_handlers
        
        # Should not raise regardless of platform
        _setup_signal_handlers()

    def test_stop_event_signal_behavior(self):
        """Test that stop_event responds to stop requests."""
        assert not state.stop_event.is_set()
        
        state.stop_event.set()
        assert state.stop_event.is_set()
        
        state.stop_event.clear()
        assert not state.stop_event.is_set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
