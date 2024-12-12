# -*- coding: utf-8 -*-

"""Utility functions to use for application detection."""

from __future__ import annotations

__all__ = ['collect_str', 'join']

import os
import re
import typing
from .. import validation_utils
from . import folders

T_co = typing.TypeVar('T_co', covariant=True)
RecursiveOptionalString = typing.Union[None, str, 'RecursiveIterable[typing.Optional[str]]']


class RecursiveIterable(typing.Protocol[T_co]):  # pylint: disable=too-few-public-methods
    """Iterable that can yield another recursive iterable alongside other items."""

    def __iter__(self) -> typing.Iterator[typing.Union[T_co, RecursiveIterable[T_co]]]:
        """Iterate through a content of the object."""


def collect_str(source: RecursiveOptionalString) -> typing.Iterator[str]:
    """Collect string values from the recursive collections/iterables of :class:`~str` or :code:`None` instances."""
    stack: typing.List[RecursiveOptionalString] = [source]
    while stack:
        current: RecursiveOptionalString = stack.pop()

        if current is None:
            continue

        if isinstance(current, str):
            yield current

        else:
            offset: int = len(stack)
            child: typing.Union[None, str, typing.Iterable[typing.Any]]
            for child in current:
                stack.insert(offset, child)


@typing.overload
def join(root: str, *args: str) -> str:
    """Join path parts, if all of them are not :code:`None`."""


@typing.overload
def join(root: typing.Optional[str], *args: typing.Optional[str]) -> typing.Optional[str]:
    """Join path parts, if all of them are not :code:`None`."""


def join(root: typing.Optional[str], *args: typing.Optional[str]) -> typing.Optional[str]:
    """Join path parts, if all of them are not :code:`None`."""
    if root is None:
        return None

    if any(arg is None for arg in args):
        return None
    args = typing.cast(typing.Tuple[str, ...], args)

    return os.path.join(root, *args)


INLINE_VALUE_PATTERN: typing.Final[typing.Pattern[str]] = re.compile(r'{([^}]*)}')


def parse_and_join(value: typing.Union[str, typing.Sequence[str]]) -> typing.Optional[str]:
    """Parse path parts and join them."""
    if isinstance(value, str):
        return value

    # `str` is added here for the verbosity of the message
    validation_utils.of_type(str, typing.Sequence)(value)

    part: typing.Optional[str]
    parts: typing.List[typing.Optional[str]] = []
    for part in validation_utils.iterable_items(value):
        validation_utils.is_str(part)

        index: int
        items: typing.List[str] = INLINE_VALUE_PATTERN.split(part)
        for index in range(1, len(items), 2):
            item: 'folders.Folder' = typing.cast('folders.Folder', items[index].strip())
            validation_utils.of_options(*folders.FOLDERS.keys())(item, field_name=('inline', index // 2))
            current: typing.Optional[str] = folders.FOLDERS[item]
            if current is None:
                parts.append(None)
                break
            items[index] = current
        else:
            parts.append(''.join(items))

    validation_utils.has_items(at_least=1)(parts)

    with validation_utils.catch_exception():
        return join(*parts)  # pylint: disable=no-value-for-parameter
