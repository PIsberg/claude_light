#!/usr/bin/env python3
"""Quick test to verify git_manager module imports correctly."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Set dummy API key to avoid exit
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

try:
    from claude_light import git_manager
    print("✓ git_manager imported successfully")
    
    # Check all expected functions exist
    functions = [
        'is_git_repo',
        'get_git_root',
        'get_modified_files',
        'get_last_commit_message',
        'auto_commit',
        'undo_last_commit',
        'get_commit_history',
    ]
    
    for func_name in functions:
        if hasattr(git_manager, func_name):
            print(f"✓ Function '{func_name}' exists")
        else:
            print(f"✗ Function '{func_name}' missing!")
            sys.exit(1)
    
    print("\n✓ All git_manager functions are present and importable")
    sys.exit(0)
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
