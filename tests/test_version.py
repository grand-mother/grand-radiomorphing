# -*- coding: utf-8 -*-
"""
Unit tests for the grand_radiomorphing.version module
"""

import unittest
import sys

import grand_radiomorphing
from grand_pkg import git


try:
    import grand_radiomorphing.version
except:
    # Skip version tests for non release builds
    pass
else:
    class VersionTest(unittest.TestCase):
        """Unit tests for the version module"""

        def test_hash(self):
            githash = git("rev-parse", "HEAD")
            self.assertEqual(githash.strip(),
                             grand_radiomorphing.version.__git__["sha1"])

        def test_version(self):
            self.assertIsNotNone(grand_radiomorphing.version.__version__)


if __name__ == "__main__":
    unittest.main()
