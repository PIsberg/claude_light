#!/usr/bin/env python3
"""
Wrapper to run the modularized claude_light application.
"""
import os
# Fix for Windows: prevent Intel Fortran runtime (from NumPy/PyTorch)
# from intercepting Ctrl+C and causing a 'forrtl: error (200)' crash.
if os.name == 'nt':
    os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from claude_light.main import main

if __name__ == "__main__":
    main()
