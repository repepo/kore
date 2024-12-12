# -*- coding: utf-8 -*-

"""A simple base to create singleton instances."""

from __future__ import annotations

__all__ = ['Singleton', 'SingleInstanceOf']

import abc
import typing


T = typing.TypeVar('T')


__singletons__: typing.Final[typing.List[Singleton]] = []


def reset_all() -> None:
    """Reset all singletons in application at once."""
    singleton: Singleton
    for singleton in __singletons__:
        singleton.reset()


class Singleton(typing.Generic[T], metaclass=abc.ABCMeta):
    """
    Container for a singleton instance of some type.

    Supports lazy initialization as soon as the instance might be required.
    """

    __slots__ = ('__instance',)

    def __init__(self) -> None:
        """Initialize new :class:`~Singleton` instance."""
        self.__instance: typing.Optional[T] = None
        __singletons__.append(self)

    @property
    def instance(self) -> T:  # noqa: D401
        """Singleton instance."""
        if self.__instance is None:
            self.__instance = self._prepare()
        return self.__instance

    def reset(self) -> None:
        """
        Reset singleton value.

        This will trigger repeated initialization of an instance on next :attr:`~Singleton.instance` call.
        """
        if self.__instance is None:
            return

        self._release()
        self.__instance = None

    @abc.abstractmethod
    def _prepare(self) -> T:
        """Initialize singleton instance."""

    def _release(self) -> None:
        """Destroy singleton instance when :meth:`~Singleton.reset` is called."""


class SingleInstanceOf(Singleton[T], typing.Generic[T]):
    """Shortcut for singletons constructed from external functions or instance constructors."""

    __slots__ = ('__constructor',)

    def __init__(self, constructor: typing.Callable[[], T]) -> None:
        """Initialize new :class:`~SingleInstanceOf` instance."""
        super().__init__()
        self.__constructor: typing.Final[typing.Callable[[], T]] = constructor

    def _prepare(self) -> T:
        """Initialize singleton instance."""
        return self.__constructor()
