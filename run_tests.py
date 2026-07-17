#!/usr/bin/env python3
"""Run the PCD-NeRV v2 unit-test suite (stdlib unittest, no extra deps)."""

import os
import sys
import unittest

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(os.path.join(HERE, 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
