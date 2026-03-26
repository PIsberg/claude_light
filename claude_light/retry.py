"""
Rate-limit handling and retry logic for Claude Light.

Implements exponential backoff for transient API errors (429, 5xx).
"""

import time
import anthropic
from typing import Callable, TypeVar, Any
from claude_light.ui import _T_SYS, _T_ERR, _ANSI_YELLOW, _ANSI_RESET, _ANSI_DIM

T = TypeVar('T')

# Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECS = 2  # Start at 2 seconds
MAX_BACKOFF_SECS = 60     # Cap at 60 seconds


def _should_retry(error: Exception) -> bool:
    """
    Determine if an error is retriable.
    
    Retriable errors:
    - RateLimitError (429 Too Many Requests)
    - APIConnectionError (transient network issues)
    - Server errors (5xx)
    
    Non-retriable:
    - AuthenticationError (401)
    - PermissionError (403)
    - NotFoundError (404)
    - InvalidRequestError (400)
    
    Args:
        error: The exception to check.
        
    Returns:
        True if the error should trigger a retry.
    """
    if isinstance(error, anthropic.RateLimitError):
        return True
    
    if isinstance(error, anthropic.APIConnectionError):
        # Transient network issue
        return True
    
    # Check for generic API errors with status codes
    if hasattr(error, 'status_code'):
        status = getattr(error, 'status_code', None)
        # 5xx errors are retriable
        if isinstance(status, int) and 500 <= status < 600:
            return True
    
    return False


def retry_with_backoff(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that retries a function with exponential backoff on transient errors.
    
    Usage:
        @retry_with_backoff
        def call_claude():
            return client.messages.create(...)
    
    Args:
        func: Function to wrap with retry logic.
        
    Returns:
        Wrapped function that implements exponential backoff.
    """
    def wrapper(*args, **kwargs) -> T:
        backoff = INITIAL_BACKOFF_SECS
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                # Check if this error is retriable
                if not _should_retry(e):
                    # Not retriable; fail immediately
                    raise
                
                # This is the last attempt; don't retry further
                if attempt >= MAX_RETRIES:
                    break
                
                # Calculate backoff time (exponential: 2, 4, 8, ...)
                wait_time = min(backoff, MAX_BACKOFF_SECS)
                backoff *= 2
                
                # Show retry message
                error_type = type(e).__name__
                print(
                    f"\n{_T_SYS} {error_type} (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
        
        # All retries exhausted
        if last_error:
            raise last_error
        raise Exception("Unexpected: all retries failed but no error recorded")
    
    return wrapper
