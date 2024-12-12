# -*- coding: utf-8 -*-

"""Additional data collections."""

from __future__ import annotations

__all__ = ['OrderedSet']

import collections
import typing


T = typing.TypeVar('T', bound=typing.Hashable)


class OrderedSet(typing.MutableSet[T], typing.Generic[T]):
    """Set, that preserves order of items added to it."""

    __slots__ = ('__content',)

    def __init__(self, content: typing.Iterable[T] = ()) -> None:
        """Initialize new :class:`~OrderedSet` instance."""
        self.__content: typing.Final[typing.OrderedDict[T, None]] = collections.OrderedDict(
            (item, None)
            for item in content
        )

    def add(self, value: T) -> None:
        """Add new value to :class:`~OrderedSet` instance."""
        self.__content[value] = None

    def discard(self, value: T) -> None:
        """Discard a value from :class:`~OrderedSet` instance."""
        self.__content.pop(value, None)

    def update(self, values: typing.Iterable[T]) -> None:
        """Add multiple values to :class:`~OrderedSet` instance."""
        value: T
        for value in values:
            self.add(value)

    def __contains__(self, value: typing.Any) -> bool:
        """Check if `value` is in :class:`~OrderedSet` instance."""
        return value in self.__content

    def __iter__(self) -> typing.Iterator[T]:
        """Iterate through the added values to :class:`~OrderedSet` instance."""
        return iter(self.__content)

    def __len__(self) -> int:
        """Retrieve total number of added items in :class:`~OrderedSet` instance."""
        return len(self.__content)
