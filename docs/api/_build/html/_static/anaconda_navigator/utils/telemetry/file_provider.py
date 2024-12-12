# -*- coding: utf8 -*-

"""Simple provider to store telemetry into a file."""

from __future__ import annotations

__all__ = ['FileProvider']

import json

import collections.abc
import typing

from . import core


class FileProvider(core.Provider):
    """
    Provider that writes all telemetry to the file.

    :param file: Path to a file to write telemetry to.
    """

    __slots__ = ('__stream',)

    def __init__(self, context: core.Context, file: str, *, alias: str = 'file') -> None:
        """Initialize new instance of a :class:`~FileProvider`."""
        super().__init__(alias=alias, context=context)

        # pylint: disable=consider-using-with
        self.__stream: typing.Final[typing.TextIO] = open(file, 'at', encoding='utf8')

    def close(self) -> None:
        """Close provider."""
        self.__stream.close()

    def process(self, events: collections.abc.MutableSequence[core.Event]) -> None:
        """Process all pending events."""
        for event in events:
            json.dump(
                {
                    'key': event.key,
                    'name': event.name,
                    'properties': event.properties,
                    'timestamp': event.timestamp.isoformat(),
                    'user_properties': event.user_properties,
                },
                self.__stream,
            )
            self.__stream.write('\n')
        events.clear()
