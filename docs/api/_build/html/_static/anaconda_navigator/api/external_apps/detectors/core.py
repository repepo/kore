# -*- coding: utf-8 -*-

"""Core definitions for detectors."""

from __future__ import annotations

__all__ = ['DetectedApplication', 'DetectorContext', 'Detector', 'Source', 'Filter']

import abc
import os
import typing
from .. import exceptions
from .. import import_utils
from .. import validation_utils

if typing.TYPE_CHECKING:
    from anaconda_navigator.config import user as user_config


KEEP: typing.Any = object()


class DetectedApplication:
    """Description of the detected application."""

    __slots__ = ('__root', '__executable', '__version')

    def __init__(
            self,
            *,
            root: typing.Optional[str] = None,
            executable: typing.Optional[str] = None,
            version: typing.Optional[str] = None,
    ) -> None:
        """Initialize new :class:`~DetectedApplication`."""
        if root is not None:
            root = os.path.abspath(root)
        if executable is not None:
            executable = os.path.abspath(executable)

        self.__root: typing.Final[typing.Optional[str]] = root
        self.__executable: typing.Final[typing.Optional[str]] = executable
        self.__version: typing.Final[typing.Optional[str]] = version

    @property
    def complete(self) -> bool:  # noqa: D401
        """All required values are provided."""
        return all(
            item is not None
            for item in (self.__root, self.__executable, self.__version)
        )

    @property
    def root(self) -> str:  # noqa: D401
        """Root directory of the detected application."""
        if self.__root is None:
            raise AttributeError('Incomplete application description')
        return self.__root

    @property
    def executable(self) -> str:  # noqa: D401
        """Path to the executable of the detected application."""
        if self.__executable is None:
            raise AttributeError('Incomplete application description')
        return self.__executable

    @property
    def version(self) -> str:  # noqa: D401
        """Version of the detected application."""
        if self.__version is None:
            raise AttributeError('Incomplete application description')
        return self.__version

    def replace(
            self,
            *,
            root: typing.Optional[str] = KEEP,
            executable: typing.Optional[str] = KEEP,
            version: typing.Optional[str] = KEEP,
    ) -> 'DetectedApplication':
        """Prepare a copy of current instance with some additional replacements."""
        if root is KEEP:
            root = self.__root
        if executable is KEEP:
            executable = self.__executable
        if version is KEEP:
            version = self.__version
        return DetectedApplication(root=root, executable=executable, version=version)


class DetectorContext(typing.NamedTuple):
    """Context used for application detection."""

    app_name: str
    user_configuration: 'user_config.UserConfig'


class Detector(metaclass=abc.ABCMeta):  # pylint: disable=too-few-public-methods
    """Common interface for application detectors."""

    __slots__ = ()

    __detectors__: typing.Final[typing.MutableMapping[str, typing.Type[Detector]]] = {}

    def __init_subclass__(
            cls,  # pylint: disable=unused-argument
            alias: typing.Optional[str] = None,
            **kwargs: typing.Any,
    ) -> None:
        """
        Initialize new :class:`~Detector` subclass.

        Used to register new classes in registry.
        """
        if alias is None:
            return

        if alias in cls.__detectors__:
            raise ValueError(f'{alias!r} is already registered')

        cls.__detectors__[alias] = cls

    @classmethod
    def parse_configuration(cls, configuration: typing.Mapping[str, typing.Any]) -> Detector:
        """Generate detectors according to the `configuration`."""
        validation_utils.of_type(typing.Mapping)(configuration)

        target_cls: typing.Type[Detector]
        type_: str = validation_utils.get_mapping_item(configuration, 'type')
        with exceptions.ValidationError.with_field('type'):
            validation_utils.is_str(type_)
            if type_.startswith('import!'):
                target_cls = import_utils.import_class(type_[7:], inherited_from=Detector)  # type: ignore
            else:
                validation_utils.of_options(*cls.__detectors__)(type_, field_name='type')
                target_cls = cls.__detectors__[type_]

        args: typing.Iterable[typing.Any] = configuration.get('args', ())
        validation_utils.of_type(typing.Sequence)(args, 'args')

        return target_cls._parse_configuration(  # pylint: disable=protected-access
            *args,
            **{key: value for key, value in configuration.items() if key not in {'args', 'type'}},
        )

    @staticmethod
    @abc.abstractmethod
    def _parse_configuration(*args: typing.Any, **kwargs: typing.Any) -> Detector:
        """Parse configuration for this particular :class:`~Detector`."""


class Source(Detector, metaclass=abc.ABCMeta):
    """
    Initial source of application details.

    It is not required for application details to be complete - they still might be extended with filters.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __call__(
            self,
            *,
            context: DetectorContext,
    ) -> typing.Iterator['DetectedApplication']:
        """Iterate through detected applications."""


class Filter(Detector, metaclass=abc.ABCMeta):
    """
    Filters for application details.

    Filters are allowed to generate new application details, as well as modify existing ones before forwarding them
    further.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __call__(
            self,
            parent: typing.Iterator['DetectedApplication'],
            *,
            context: DetectorContext,
    ) -> typing.Iterator['DetectedApplication']:
        """Iterate through detected applications."""
