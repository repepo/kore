# -*- coding: utf-8 -*-

"""Additional utility functions to use with detectors."""

from __future__ import annotations

__all__ = ['extract_app_from_command']

import shlex
import typing


def extract_app_from_command(value: str, value_type: int) -> typing.Iterator[str]:  # pylint: disable=unused-argument
    """Extract path to executable from shell-like command."""
    result: typing.List[str] = shlex.split(value)
    if result:
        yield result[0]
