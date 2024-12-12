# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2020 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Tool path utilities."""

import os
import sys


def get_pyexec(prefix: str) -> str:
    """Provides the full path to python executable"""
    result: str = ''
    if sys.platform == 'win32':
        result += os.sep.join([prefix, 'python.exe'])
    else:
        result += os.sep.join([prefix, 'bin', 'python'])
    return result


def get_pyscript(prefix: str, name: str) -> str:
    """Provides the OS dependent path in bin/Scripts for python script tool"""
    result: str = ''
    if sys.platform == 'win32':
        result += os.sep.join([prefix, 'Scripts', name + '-script.py'])
    else:
        result += os.sep.join([prefix, 'bin', name])
    return result
