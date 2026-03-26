#!/usr/bin/env python3
"""Quick syntax check for test file."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try to import the test module
    import tests.test_thread_safety as test_module
    print("✓ test_thread_safety.py imports successfully")
    
    # Check that test classes are defined
    assert hasattr(test_module, 'TestThreadSafety')
    assert hasattr(test_module, 'TestStatusBarThreadSafety')
    assert hasattr(test_module, 'TestLockContention')
    print("✓ All test classes defined")
    
    # Verify state module
    import claude_light.state as state
    assert hasattr(state, 'lock')
    print("✓ state.lock exists")
    
    print("\nAll syntax and import checks passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
