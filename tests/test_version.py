# -*- coding: utf-8 -*-
"""
Unit tests for the grand_radiomorphing.version module
"""

import unittest
import sys

import grand_radiomorphing
from framework import git


class VersionTest(unittest.TestCase):
    """Unit tests for the version module"""

    def test_hash(self):
        githash = git("rev-parse", "HEAD")
        self.assertEqual(githash.strip(), grand_radiomorphing.version.__githash__)

    def test_version(self):
        self.assertIsNotNone(grand_radiomorphing.version.__version__)


if __name__ == "__main__":
    unittest.main()
