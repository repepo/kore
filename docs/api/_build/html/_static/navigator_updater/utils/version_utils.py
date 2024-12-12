# -*- coding: utf8 -*-

"""Utilities for version comparison."""

from __future__ import annotations

__all__ = ['compare']

import re
import typing

import packaging.version


NUMBER: typing.Final[typing.Pattern[str]] = re.compile(r'([0-9]+)')


class SupportsLessThan(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Value that can be compared to another one."""

    def __lt__(self, other: typing.Any) -> bool:
        """Check if value is less than other one."""


SupportsLessThanT = typing.TypeVar('SupportsLessThanT', bound=SupportsLessThan)


def _compare(first: SupportsLessThanT, second: SupportsLessThanT) -> int:
    """
    Compare two values.

    :return: :code:`1` if :code:`first` > :code:`second`; :code:`-1` if :code:`first` < :code:`second`; :code:`0` if
             both values are equal.
    """
    if first < second:
        return -1
    if second < first:
        return 1
    return 0


def _parse(value: str) -> typing.Tuple[typing.Union[int, str], ...]:
    """Parse any version-like string into tuple of integer-string values."""
    return tuple(
        int(part) if (index % 2) else str(part)
        for index, part in enumerate(NUMBER.split(value))
    )


def compare(first: str, second: str) -> int:
    """Compare two strings with versions."""
    try:
        return _compare(packaging.version.Version(first), packaging.version.Version(second))
    except packaging.version.InvalidVersion:
        return _compare(_parse(first), _parse(second))
