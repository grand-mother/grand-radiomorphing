import unittest
import os
import shutil
from os.path import split, join, realpath
import sys

from framework import init, RunContext
root_dir = realpath(join(split(__file__)[0], ".."))
sys.path.append(join(root_dir, "examples"))
import example


class Running(unittest.TestCase):
    """Run example for the init module"""

    def test_run(self):
        print('executing test')
        #path = os.path.join(self.tmpdir, package_name)
        args = (' ')#("python3","../examples/example.py") 
        with RunContext(*args) as context:
            print('executing test')
            example.run()
            
        self.assertEqual(context.err, "")
        self.assertEqual(context.code, 0)



if __name__ == "__main__":
    unittest.main()
