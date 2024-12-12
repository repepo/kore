# -*- coding: utf-8 -*-

"""Common configuration structures to use in :mod:`~anaconda_navigator.config.preferences`."""

from __future__ import annotations

__all__ = ['SidebarLink', 'SidebarSocial', 'Interval', 'Intervals']

import bisect
import typing

import attr

T = typing.TypeVar('T')
T_co = typing.TypeVar('T_co', covariant=True)


class SidebarLink(typing.NamedTuple):
    """Description of a single link in a sidebar."""

    text: str
    url: str
    utm_medium: str


class SidebarSocial(typing.NamedTuple):
    """Description of a single social button in a sidebar."""

    text: str
    url: str


@attr.s(auto_attribs=True, eq=False, frozen=True, slots=True)
class Interval(typing.Generic[T_co]):  # pylint: disable=too-few-public-methods
    """
    Value that repeats `count` times.

    :param count: how many times the value should be repeated.
    :param value: value that should be repeated.
    """

    count: int
    value: T_co


class Intervals(typing.Generic[T]):
    """
    Sequence of repeating values.

    :param args: values to store in this sequence.
    :param offset: index of the first value in the sequence. This allows creating custom sequences, which may start
                   from any custom index (e.g. create sequence with indices from -5 to 15, or from 100 to 123).
    """

    __slots__ = ('__keys', '__offset', '__values')

    def __init__(self, *args: Interval[T], offset: int = 0) -> None:
        """Initialize new :class:`~Intervals` instance."""
        if not args:
            raise TypeError(f'at least single {Interval.__name__} must be provided')

        self.__keys: typing.Final[typing.List[int]] = []
        self.__offset: typing.Final[int] = offset
        self.__values: typing.Final[typing.List[T]] = []

        offset -= 1

        arg: Interval[T]
        for arg in args:
            if arg.count <= 0:
                raise ValueError(f'each item may repeat only a positive number of times, not {arg.count}')

            offset += arg.count
            self.__keys.append(offset)
            self.__values.append(arg.value)

    @property
    def first(self) -> int:  # noqa: D401
        """Index of the first element in the sequence."""
        return self.__offset

    @property
    def last(self) -> int:  # noqa: D401
        """Index of the last element in the sequence."""
        return self.__keys[-1]

    def get(self, index: int, *, strict: bool = False) -> T:
        """
        Retrieve element at `index`.

        If `strict` is set - method may raise an :exc:`~IndexError` if invalid item is requested. Otherwise - method
        will return value closest to the requested index.
        """
        if strict:
            if index < self.__offset:
                raise IndexError(f'index must be at least {self.__offset!r}')
            if index > self.__keys[-1]:
                raise IndexError(f'index must be at most {self.__offset!r}')

        index = bisect.bisect_left(self.__keys, index, 0, len(self.__keys) - 1)
        return self.__values[index]
