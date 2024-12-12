# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) Spyder Project Contributors
#
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)
# -----------------------------------------------------------------------------

"""Running programs utilities."""

from __future__ import print_function

import os


def is_program_installed(basename):
    """
    Return program absolute path if installed in PATH.

    Otherwise, return None
    """
    for path in os.environ['PATH'].split(os.pathsep):
        abspath = os.path.join(path, basename)
        if os.path.isfile(abspath):
            return abspath
    return None
