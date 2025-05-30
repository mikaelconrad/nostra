#!/usr/bin/env python3
"""
Test runner for the cryptocurrency investment app
"""

import sys
import subprocess
import os

def run_tests():
    """Run the test suite"""
    print("="*60)
    print("Running Cryptocurrency Investment App Test Suite")
    print("="*60)
    
    # Install pytest if not available
    try:
        import pytest
    except ImportError:
        print("Installing pytest and dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "pytest", "pytest-cov", "pytest-mock", "requests-mock"])
    
    # Run tests with coverage
    args = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "--cov=backend",  # Coverage for backend
        "--cov=api",  # Coverage for api
        "--cov=utils",  # Coverage for utils
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html",  # Generate HTML report
    ]
    
    # Add any command line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run tests
    result = subprocess.run(args)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed")
        print("="*60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())