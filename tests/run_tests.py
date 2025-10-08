#!/usr/bin/env python3
"""
Simple test runner for the normalization tests.
"""

import sys
import os
import subprocess

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_tests():
    """Run all tests in the tests directory."""
    test_dir = os.path.dirname(__file__)
    
    # Run pytest on the tests directory
    cmd = [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"]
    
    print(f"Running tests with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(test_dir))
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
