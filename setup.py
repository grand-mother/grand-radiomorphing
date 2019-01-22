# -*- coding: utf-8 -*-

from framework import setup_package


# The package version
MAJOR = 0
MINOR = 0
MICRO = 0


# Extra package meta data can be added here. For a full list of available
# classifiers, see:
#     https://pypi.org/pypi?%3Aaction=list_classifiers
EXTRA_CLASSIFIERS = (
    "Development Status :: 1 - Planning",
)

def main():
    setup_package(
        # Framework arguments
        __file__, (MAJOR, MINOR, MICRO), EXTRA_CLASSIFIERS,

        # Vanilla setuptools.setup arguments can be added below,
        # e.g. `entry_points` for executables or `data_files`
        
        install_requires = (
            "numpy>=1.13.3",
            "scipy>=1.2.0",
            "autopep8>=1.4.0"
            )




if __name__ == "__main__":
    main()
