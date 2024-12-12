# -*- coding: utf-8 -*-

"""Grouping components for detector items."""

from __future__ import annotations

__all__ = ['Group', 'OptionalGroup', 'LinuxOnly', 'OsXOnly', 'WindowsOnly']

import abc
import itertools
import typing
from anaconda_navigator import config as navigator_config
from .. import exceptions
from .. import validation_utils
from . import core


class Group(core.Source, alias='group'):  # pylint: disable=too-few-public-methods
    """
    Group of application detecting sources and filters.

    Order in which you provide sources and filters matters. Output of each source is chained with all previously
    detected applications (if such). Filters are applied to all previously detected applications.
    """

    __slots__ = ('__content',)

    def __init__(self, *args: core.Detector) -> None:
        """Initialize new :class:`~Group` instance."""
        if len(args) <= 0:
            raise TypeError('at least one source must be provided')

        if not isinstance(args[0], core.Source):
            raise TypeError(f'first argument must be a {core.Source.__qualname__!r}')

        arg: core.Detector
        for arg in args:
            if not isinstance(arg, core.Detector):
                raise TypeError(f'each argument must be a {core.Detector.__qualname__!r}, not {type(arg).__name__!r}')

        self.__content: typing.Final[typing.Tuple[core.Detector, ...]] = args

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        new_args: typing.List[core.Detector] = []
        with exceptions.ValidationError.with_field('args'):
            validation_utils.has_items(at_least=1)(args)

            arg: typing.Any
            for arg in validation_utils.iterable_items(args):
                new_args.append(core.Detector.parse_configuration(configuration=arg))

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*new_args)

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        result: typing.Iterator[core.DetectedApplication] = typing.cast(core.Source, self.__content[0])(context=context)

        arg: core.Detector
        for arg in itertools.islice(self.__content, 1, len(self.__content), 1):
            if isinstance(arg, core.Source):
                result = itertools.chain(result, arg(context=context))
            elif isinstance(arg, core.Filter):
                result = arg(result, context=context)
            else:
                raise TypeError(f'unexpected detector type: {type(arg).__name__}')

        return result


class OptionalGroup(Group, metaclass=abc.ABCMeta):  # pylint: disable=too-few-public-methods
    """Group, that might be omitted if a condition isn't met."""

    __slots__ = ()

    @abc.abstractmethod
    def _check(self) -> bool:
        """Check if group should be processed."""

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        if self._check():
            return super().__call__(context=context)
        return iter(())


class LinuxOnly(OptionalGroup, alias='linux_only'):  # pylint: disable=too-few-public-methods
    """Group, that would be processed only on Linux machines."""

    __slots__ = ()

    def _check(self) -> bool:
        """Check if group should be processed."""
        return navigator_config.LINUX


class OsXOnly(OptionalGroup, alias='osx_only'):  # pylint: disable=too-few-public-methods
    """Group, that would be processed only on OS X machines."""

    __slots__ = ()

    def _check(self) -> bool:
        """Check if group should be processed."""
        return navigator_config.MAC


class WindowsOnly(OptionalGroup, alias='windows_only'):  # pylint: disable=too-few-public-methods
    """Group, that would be processed only on Windows machines."""

    __slots__ = ()

    def _check(self) -> bool:
        """Check if group should be processed."""
        return navigator_config.WIN
