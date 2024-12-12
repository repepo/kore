# -*- coding: utf-8 -*-

"""Utilities to work with ANSI characters."""

__all__ = ['escape_ansi']

import re
import typing


ESCAPE_ANSI_PATTERN: typing.Final[typing.Pattern[str]] = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')


def escape_ansi(string: str) -> str:
    """
    Remove the ANSI escape sequences from a string.
    :param: String to cleared from ANSI escape sequences.
    :return: Cleared string.
    """
    return ESCAPE_ANSI_PATTERN.sub('', string)
