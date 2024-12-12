# -*- coding: utf8 -*-

"""Basic event pool to process events in runtime-only."""

from __future__ import annotations

__all__ = ['BasicPool']

import collections.abc
import typing

from anaconda_navigator import config
from . import core


class BasicPool(core.Pool):
    """
    Basic event pool to process events in runtime-only.

    All events that are not submitted to the telemetry services will be lost.

    Service takes :code:`provide_analytics` preference into account when it processes events.
    """

    __slots__ = ('__content',)

    def __init__(self) -> None:
        """Initialize new instance of a :class:`~BasicPool`."""
        self.__content: typing.Final[dict[str, list[core.Event]]] = {}

    def pending(self, provider: str) -> list[core.Event]:
        """Retrieve all currently pending events for :code:`provider`."""
        result: list[core.Event] = self.__content.setdefault(provider, [])
        if not config.CONF.get('main', 'provide_analytics'):
            result.clear()
        return result

    def push(self, event: core.Event) -> None:
        """Push new :code:`event` to the pool."""
        if not config.CONF.get('main', 'provide_analytics'):
            return

        value: list[core.Event]
        for value in self.__content.values():
            value.append(event)

    def register(self, providers: collections.abc.Iterable[str]) -> None:
        """Register :code:`providers` in the pool for it to be aware of them."""
        provider: str
        for provider in providers:
            self.__content.setdefault(provider, [])
