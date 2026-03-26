#!/bin/bash
cd /c/dev/private/claude_light
git add -A
git commit -m "Implement API rate-limit handling with exponential backoff

- Add new retry module (claude_light/retry.py) with @retry_with_backoff decorator
- Handles transient errors: 429 rate limits, 5xx server errors, connection errors
- Exponential backoff strategy: 2s -> 4s -> 8s (capped at 60s)
- Max 3 retry attempts before failure
- Non-retriable errors (401, 403, 404, 400) fail immediately
- Wraps all API calls in chat(), one_shot(), and _summarize_turns()
- Add comprehensive test suite (tests/test_retry.py) with 11+ test cases
- Update CLAUDE.md documentation

Prevents crashes under transient API failures. Guarantees eventual success or
graceful failure for all recoverable errors.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
