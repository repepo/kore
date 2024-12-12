# -*- coding: utf-8 -*-

"""Components to expand/modify initial application descriptions."""

from __future__ import annotations

__all__ = ['StepIntoRoot', 'AppendExecutable', 'AppendRoot']

import os
import typing
from .. import exceptions
from .. import validation_utils
from . import core
from . import utilities


class StepIntoRoot(core.Filter, alias='step_into_root'):  # pylint: disable=too-few-public-methods
    """
    Iterate through children of application root.

    Each child might be checked that it `starts_with`, `ends_with` or `equals` to one or multiple values.

    Iteration results are sorted. If you want descending order of the results - set `reverse` to :code:`True`.
    """

    __slots__ = ('__equals', '__starts_with', '__ends_with', '__reverse')

    def __init__(
            self,
            *,
            equals: typing.Union[str, typing.Iterable[str]] = (),
            starts_with: typing.Union[str, typing.Iterable[str]] = (),
            ends_with: typing.Union[str, typing.Iterable[str]] = (),
            reverse: bool = False,
    ) -> None:
        """Initialize new :class:`~StepIntoRoot` instance."""
        if isinstance(equals, str):
            equals = {equals}
        else:
            equals = set(equals)

        if isinstance(starts_with, str):
            starts_with = (starts_with,)
        else:
            starts_with = tuple(starts_with)

        if isinstance(ends_with, str):
            ends_with = (ends_with,)
        else:
            ends_with = tuple(ends_with)

        self.__equals: typing.Final[typing.Set[str]] = equals
        self.__starts_with: typing.Final[typing.Tuple[str, ...]] = starts_with
        self.__ends_with: typing.Final[typing.Tuple[str, ...]] = ends_with
        self.__reverse: typing.Final[bool] = reverse

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        field_validator: typing.Final[validation_utils.ValueChecker] = validation_utils.any_of(
            validation_utils.is_str,
            validation_utils.is_str_sequence,
        )

        validation_utils.has_items(at_most=0)(args, field_name='args')

        equals: typing.Union[str, typing.Iterable[str]] = kwargs.pop('equals', ())
        field_validator(equals, field_name='equals')

        starts_with: typing.Union[str, typing.Iterable[str]] = kwargs.pop('starts_with', ())
        field_validator(starts_with, field_name='starts_with')

        ends_with: typing.Union[str, typing.Iterable[str]] = kwargs.pop('ends_with', ())
        field_validator(ends_with, field_name='ends_with')

        reverse: bool = kwargs.pop('reverse', False)
        validation_utils.is_bool(reverse, field_name='reverse')

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(equals=equals, starts_with=starts_with, ends_with=ends_with, reverse=reverse)

    def __call__(
            self,
            parent: typing.Iterator[core.DetectedApplication],
            *,
            context: core.DetectorContext,
    ) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        application: core.DetectedApplication
        for application in parent:
            if not application.root:
                continue

            children: typing.List[str]
            try:
                children = sorted(os.listdir(application.root), reverse=self.__reverse)
            except OSError:
                continue

            child: str
            for child in children:
                if self.__equals and (child not in self.__equals):
                    continue
                if self.__starts_with and not any(child.startswith(item) for item in self.__starts_with):
                    continue
                if self.__ends_with and not any(child.endswith(item) for item in self.__ends_with):
                    continue
                yield application.replace(root=os.path.join(application.root, child))


class AppendExecutable(core.Filter, alias='append_executable'):  # pylint: disable=too-few-public-methods
    """
    Check multiple options of executables and iterate through existing ones.

    Each option of the executable must be a path relative to the application root.
    """

    __slots__ = ('__content',)

    def __init__(self, *args: typing.Union[None, str, typing.Iterable[typing.Optional[str]]]) -> None:
        """Initialize new :class:`~AppendExecutable` instance."""
        self.__content: typing.Final[typing.Tuple[str, ...]] = tuple(utilities.collect_str(args))

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        new_args: typing.List[str] = []
        with exceptions.ValidationError.with_field('args'):
            validation_utils.has_items(at_least=1)(args)
            new_args.extend(map(utilities.parse_and_join, validation_utils.iterable_items(args)))  # type: ignore

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*new_args)

    def __call__(
            self,
            parent: typing.Iterator[core.DetectedApplication],
            *,
            context: core.DetectorContext,
    ) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        application: core.DetectedApplication
        for application in parent:
            executable: str
            for executable in self.__content:
                executable = os.path.abspath(os.path.join(application.root, executable))
                if os.path.isfile(executable):
                    yield application.replace(executable=executable)


class AppendRoot(core.Filter, alias='append_root'):  # pylint: disable=too-few-public-methods
    """
    Append parent directory of an `executable` as a root.

    It is possibly to select the `level` of the parent folder, which should be used as a root directory.

    If you have  application with :code:`executable = '/a/b/c/d/e/application'`, then:

    - :code:`level=0` would result in :code:`root = '/a/b/c/d/e'`
    - :code:`level=1` would result in :code:`root = '/a/b/c/d'`
    - :code:`level=2` would result in :code:`root = '/a/b/c'`
    - and so on
    """

    __slots__ = ('__level',)

    def __init__(self, *, level: int = 0) -> None:
        """Initialize new :class:`~AppendRoot` instance."""
        self.__level: typing.Final[int] = level + 1

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        validation_utils.has_items(at_most=0)(args, field_name='args')

        level = kwargs.pop('level', 0)
        validation_utils.is_int(level, field_name='level')

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(level=level)

    def __call__(
            self,
            parent: typing.Iterator[core.DetectedApplication],
            *,
            context: core.DetectorContext,
    ) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        application: core.DetectedApplication
        for application in parent:
            root: str = application.executable
            for _ in range(self.__level):
                root = os.path.dirname(root)
            yield application.replace(root=root)
