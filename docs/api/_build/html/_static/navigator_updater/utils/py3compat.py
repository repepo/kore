# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
Transitional module providing compatibility functions between python 2 and 3.

This module should be fully compatible with:
    * Python >=v2.7
    * Python 3
"""

from __future__ import print_function

import typing
import operator
import os
import sys


PY2 = sys.version[0] == '2'
PY3 = sys.version[0] == '3'

# =============================================================================
# Data types
# =============================================================================
TEXT_TYPES: typing.Tuple[type, ...] = (str,)
INT_TYPES: typing.Tuple[type, ...] = (int,)
NUMERIC_TYPES: typing.Tuple[type, ...] = INT_TYPES + (float, complex)


# =============================================================================
# Strings
# =============================================================================

def is_text_string(obj):
    """
    Return True if `obj` is a text string, False if it is anything else.

    Binary data (Python 3) or QString (Python 2, PyQt API #1).
    """
    return isinstance(obj, str)


def is_binary_string(obj):
    """Return True if obj is a binary string, False if it is anything else."""
    return isinstance(obj, bytes)


def is_string(obj):
    """
    Check if object is a string.

    Return True if `obj` is a text or binary Python string object False if it
    is anything else, like a QString (Python 2, PyQt API #1).
    """
    return is_text_string(obj) or is_binary_string(obj)


def is_unicode(obj):
    """Return True if `obj` is unicode."""
    return isinstance(obj, str)


def to_text_string(obj, encoding=None):
    """Convert `obj` to (unicode) text string."""
    if encoding is None:
        return str(obj)
    if isinstance(obj, str):  # In case this function is not used properly, this could happen
        return obj
    return str(obj, encoding)


def to_binary_string(obj, encoding=None):
    """Convert `obj` to binary string (bytes in Python 3, str in Python 2)."""
    return bytes(obj, 'utf-8' if encoding is None else encoding)


def u(obj):
    """Return string as it is."""
    return obj


# =============================================================================
# Misc.
# =============================================================================

getcwd = os.getcwd
str_lower = str.lower


def cmp(a, b):
    """Return negative if a<b, zero if a==b, positive if a>b."""
    return (a > b) - (a < b)


def qbytearray_to_str(qba):
    """Convert QByteArray object to str in a way compatible with Python 2/3."""
    return str(bytes(qba.toHex().data()).decode())


# =============================================================================
# Dict funcs
# =============================================================================

def iterkeys(d, **kw):
    """Return an iterator over the dictionary's keys."""
    return iter(d.keys(**kw))


def itervalues(d, **kw):
    """Return an iterator over the dictionary's values."""
    return iter(d.values(**kw))


def iteritems(d, **kw):
    """Return an iterator over the dictionary's items."""
    return iter(d.items(**kw))


def iterlists(d, **kw):
    """Return an iterator over a multi dictionary."""
    return iter(d.lists(**kw))


viewkeys = operator.methodcaller('keys')

viewvalues = operator.methodcaller('values')

viewitems = operator.methodcaller('items')
