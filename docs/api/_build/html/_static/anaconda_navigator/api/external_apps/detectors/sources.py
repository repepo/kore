# -*- coding: utf-8 -*-

"""Components to use as a sources of initial application descriptions."""

from __future__ import annotations

__all__ = [
    'RegistryKey', 'CheckConfiguredRoots', 'CheckKnownRoots', 'CheckKnownOsXRoots', 'CheckPATH',
    'CheckExecutableInWindowsRegistry', 'CheckRootInWindowsRegistry',
]

import abc
import os
import typing
from anaconda_navigator.utils.logs import logger
from .. import exceptions
from .. import import_utils
from .. import validation_utils
from . import core
from . import folders
from . import utilities


WindowsRegistryRoot = typing.Literal[
    'HKEY_CLASSES_ROOT',
    'HKEY_CURRENT_USER',
    'HKEY_LOCAL_MACHINE',
    'HKEY_USERS',
    'HKEY_PERFORMANCE_DATA',
    'HKEY_CURRENT_CONFIG',
]


WINDOWS_REGISTRY_ROOT_KEYS: 'typing.Final[typing.Tuple[WindowsRegistryRoot, ...]]' = (
    WindowsRegistryRoot.__args__  # type: ignore
)


class RegistryKey(typing.NamedTuple):
    """
    Information about Windows registry key, from which to get value.

    :param root: Root key to start from (e.g. :code:`'HKEY_CURRENT_USER'`, :code:`'HKEY_LOCAL_MACHINE'`.
    :param key: Path to the exact key (e.g. :code:`'app\\shell\\open\\command'`.
    :param sub_key: Subkey to get value from (might be :code:`None` to retrieve value from the root of the `key`,
                    or a name of any actual key, e.g. :code:`'ExecutablePath'`.
    :param converter: Additional modification to the key value.

                      This should be a function that accepts two arguments: value (str) and it's type identifier (int),
                      and it should return a generator of extracted values.
    """

    root: 'WindowsRegistryRoot'
    key: str
    sub_key: typing.Optional[str] = None
    converter: typing.Optional[typing.Callable[[str, int], typing.Iterator[str]]] = None

    # Possible extensions:
    #   Extend format of the key to also accept sequence of values:
    #   - strings for exact key values (just as it is right now)
    #   - predicate functions for key names (to iterate through keys, when only their format is known, but not the
    #     exact value).

    @classmethod
    def _parse_configuration(cls, configuration: typing.Mapping[str, typing.Any]) -> RegistryKey:
        """Parse configuration for :class:`~RegistryKey`."""
        validation_utils.of_type(typing.Mapping)(configuration)
        mirror: typing.Dict[str, typing.Any] = dict(configuration)

        root: 'WindowsRegistryRoot' = validation_utils.pop_mapping_item(mirror, 'root')
        validation_utils.is_str(root)
        validation_utils.of_options(*WINDOWS_REGISTRY_ROOT_KEYS)(root)

        key: str = validation_utils.pop_mapping_item(mirror, 'key')
        validation_utils.is_str(key)

        sub_key: typing.Optional[str] = mirror.pop('sub_key', None)
        validation_utils.is_str_or_none(sub_key)

        converter: typing.Optional[str] = mirror.pop('converter', None)
        validation_utils.is_str_or_none(sub_key)

        converter_value: typing.Optional[typing.Callable[..., typing.Any]]
        converter_value = import_utils.import_callable(converter, allow_none=True)

        validation_utils.mapping_is_empty()(mirror)

        return cls(root=root, key=key, sub_key=sub_key, converter=converter_value)


class CheckConfiguredRoots(core.Source, alias='check_configured_roots'):  # pylint: disable=too-few-public-methods
    """
    Get application root from the configuration.

    Multiple configuration keys might be provided as `args`.

    Configuration, from which value should be retrieved, must be provided as a keyword argument.

    If configuration section is not provided - "main" will be used by default.
    """

    __slots__ = ('__content', '__section')

    def __init__(
            self,
            *args: typing.Union[None, str, typing.Iterable[typing.Optional[str]]],
            section: str = 'applications',
    ) -> None:
        """Initialize new :class:`~CheckKnownRoots` instance."""
        self.__content: typing.Final[typing.Tuple[str, ...]] = tuple(utilities.collect_str(args))
        self.__section: typing.Final[str] = section

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        with exceptions.ValidationError.with_field('args'):
            validation_utils.has_items(at_least=1)(args)
            validation_utils.each_item(validation_utils.is_str)(args)

        section: str = kwargs.pop('section', 'applications')
        validation_utils.is_str(section, field_name='section')

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*args, section=section)

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        content: typing.Tuple[str, ...] = self.__content
        if not content:
            content = (f'{context.app_name}_path',)

        setting: str
        for setting in content:
            root: str = context.user_configuration.get(self.__section, setting, '')
            if not root:
                continue

            root = os.path.abspath(root)
            if os.path.exists(root):
                yield core.DetectedApplication(root=root)


class CheckKnownRoots(core.Source, alias='check_known_roots'):  # pylint: disable=too-few-public-methods
    """
    Iterate through known roots, selecting only existing ones.

    Some root options might be :code:`None` - they will be skipped.
    """

    __slots__ = ('__content',)

    def __init__(self, *args: typing.Union[None, str, typing.Iterable[typing.Optional[str]]]) -> None:
        """Initialize new :class:`~CheckKnownRoots` instance."""
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

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        root: str
        for root in self.__content:
            root = os.path.abspath(root)
            if os.path.exists(root):
                yield core.DetectedApplication(root=root)


class CheckKnownOsXRoots(CheckKnownRoots, alias='check_known_osx_roots'):  # pylint: disable=too-few-public-methods
    """
    Shortcut for checking application roots of OS X applications.

    Unlike :class:`~CheckKnownRoots` - you need to provide application names instead of directory paths.
    """

    __slots__ = ()

    def __init__(self, *args: typing.Union[None, str, typing.Iterable[typing.Optional[str]]]) -> None:
        """Initialize new :class:`~CheckKnownMacRoots` instance."""
        super().__init__(*(
            utilities.join(root, arg)
            for root in (folders.FOLDERS['osx/user_applications'], folders.FOLDERS['osx/applications'])
            for arg in utilities.collect_str(args)
        ))

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        with exceptions.ValidationError.with_field('args'):
            validation_utils.has_items(at_least=1)(args)
            validation_utils.each_item(validation_utils.is_str)(args)

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*args)


class CheckPATH(core.Source, alias='check_path'):  # pylint: disable=too-few-public-methods
    """
    Find path to the application executable in PATH.

    Multiple application names might be provided (like: :code:`'pycharm64.exe', 'pycharm32.exe', 'pycharm.exe'`). They
    will be checked in order they are provided, and the first match will be returned.
    """

    __slots__ = ('__content',)

    def __init__(
            self, *args: typing.Union[None, str, typing.Iterable[typing.Optional[str]]]
    ) -> None:
        """Initialize new :class:`~CheckPATH` instance."""
        self.__content: typing.Final[typing.Tuple[str, ...]] = tuple(utilities.collect_str(args))

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        with exceptions.ValidationError.with_field('args'):
            validation_utils.has_items(at_least=1)(args)
            validation_utils.each_item(validation_utils.is_str)(args)

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*args)

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        path: str
        for path in filter(bool, os.environ.get('PATH', '').split(os.pathsep)):
            executable: str
            for executable in self.__content:
                executable = os.path.join(path, executable)
                if not os.path.isfile(executable):
                    continue

                depth: int = 128
                while os.path.islink(executable) and (depth > 0):
                    executable = os.path.abspath(os.path.realpath(executable))
                    depth -= 1
                if depth <= 0:
                    logger.warning('recursive file link detected in system: %s', executable)
                    continue

                yield core.DetectedApplication(executable=executable)


class CheckWindowsRegistry(core.Source, metaclass=abc.ABCMeta):  # pylint: disable=too-few-public-methods
    """
    Common base for Windows registry checkers.

    It accepts list of keys that should be checked for values in registry. Each argument should be a
    :class:`~RegistryKey` instance.
    """

    __slots__ = ('__content',)

    def __init__(self, *args: RegistryKey) -> None:
        """Initialize new :class:`~CheckWindowRegistry` instance."""
        self.__content: typing.Final[typing.Tuple[RegistryKey, ...]] = args

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> core.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        new_args: typing.List[RegistryKey] = []
        with exceptions.ValidationError.with_field('args'):
            arg: typing.Any
            for arg in validation_utils.iterable_items(args):
                new_args.append(RegistryKey._parse_configuration(arg))  # pylint: disable=protected-access

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls(*new_args)

    def _values(self) -> typing.Iterator[str]:
        """
        Iterate through values that are available in the registry.

        All missing values will be ignored.
        """
        import winreg  # pylint: disable=import-error,import-outside-toplevel

        item: RegistryKey
        for item in self.__content:
            value: str
            value_type: int
            try:
                key: winreg.HKEYType  # type: ignore
                with winreg.OpenKeyEx(getattr(winreg, item.root), item.key) as key:  # type: ignore
                    value, value_type = winreg.QueryValueEx(key, item.sub_key)  # type: ignore
            except WindowsError:  # type: ignore  # pylint: disable=undefined-variable
                continue
            if item.converter is not None:
                yield from item.converter(value, value_type)
            else:
                yield value


class CheckRootInWindowsRegistry(  # pylint: disable=too-few-public-methods
    CheckWindowsRegistry,
    alias='check_root_in_windows_registry',
):
    """Check for application roots in Windows Registry."""

    __slots__ = ()

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        value: str
        for value in self._values():
            value = os.path.abspath(value)
            if os.path.isdir(value):
                yield core.DetectedApplication(root=value)


class CheckExecutableInWindowsRegistry(  # pylint: disable=too-few-public-methods
    CheckWindowsRegistry,
    alias='check_executable_in_windows_registry',
):
    """Check for paths to executable in Windows Registry."""

    def __call__(self, *, context: core.DetectorContext) -> typing.Iterator[core.DetectedApplication]:
        """Iterate through detected applications."""
        value: str
        for value in self._values():
            value = os.path.abspath(value)
            if os.path.isfile(value):
                yield core.DetectedApplication(executable=value)
