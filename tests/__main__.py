# -*- coding: utf-8 -*-
"""
Run all unit tests for the grand_radiomorphing package
"""
import os
import unittest


def suite():
    test_loader = unittest.TestLoader()
    path = os.path.dirname(__file__)
    test_suite = test_loader.discover(path, pattern="test_*.py")
    return test_suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
