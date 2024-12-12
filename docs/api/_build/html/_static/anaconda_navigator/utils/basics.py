# -*- coding: utf-8 -*-

"""Collection of basic utilities."""

from __future__ import annotations

__all__ = ['coalesce']

import typing


T = typing.TypeVar('T', bound=object)


@typing.overload
def coalesce(*args: typing.Optional[T], allow_none: typing.Literal[False] = False) -> T:
    """Find first not-:code:`None` value."""


@typing.overload
def coalesce(*args: typing.Optional[T], allow_none: typing.Literal[True]) -> typing.Optional[T]:
    """Find first not-:code:`None` value."""


def coalesce(
        *args: typing.Optional[T],
        allow_none: typing.Literal[True, False] = False,
) -> typing.Optional[T]:
    """
    Find first not-:code:`None` value.

    :param allow_none: Allow :code:`None` to be the result if all values are :code:`None`.
    """
    arg: typing.Optional[T]
    for arg in args:
        if arg is not None:
            return arg
    if allow_none:
        return None
    raise ValueError('not-null not found')
