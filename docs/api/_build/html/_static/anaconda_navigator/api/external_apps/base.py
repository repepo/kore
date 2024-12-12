# -*- coding: utf-8 -*-

"""Core structures for third-party application descriptions."""

from __future__ import annotations

__all__ = ['AppPatch', 'BaseApp', 'BaseWebApp', 'BaseInstallableApp']

import abc
import os
import re
import types
import typing
import webbrowser
import attr
from anaconda_navigator.static.images import IMAGE_PATH
from anaconda_navigator.utils import constants
from . import detectors
from . import exceptions
from . import import_utils
from . import parsing_utils
from . import validation_utils


if typing.TYPE_CHECKING:
    from anaconda_navigator.api import process
    from anaconda_navigator.api import types as api_types
    from anaconda_navigator.config import user as user_config


AppPatchField = typing.Literal['display_name', 'description', 'image_path', 'rank']


class AppPatchDelta(typing.TypedDict, total=False):
    """Change provided by :class:`~AppPatch`."""
    display_name: str
    description: str
    image_path: str
    rank: int


APP_PATCH_FIELD_KEYS: 'typing.Final[typing.Tuple[AppPatchField, ...]]' = AppPatchField.__args__  # type: ignore


@attr.s(auto_attribs=True, cache_hash=True, frozen=True, order=False, slots=True)
class AppPatch:
    """Patch for the application defined outside this component."""

    app_name: str = attr.ib(default='')
    display_name: typing.Optional[str] = attr.ib(default=None)
    description: typing.Optional[str] = attr.ib(default=None)
    image_path: typing.Optional[str] = attr.ib(default=None)
    is_available: bool = attr.ib(default=True)
    rank: typing.Optional[int] = attr.ib(default=None)

    @property
    def delta(self) -> 'AppPatchDelta':  # noqa: D401
        """Dict-like content of the patch."""
        result: 'AppPatchDelta' = {}

        # NOTE: should be replaced with py38+: fields = AppPatchField.__args__

        field: 'AppPatchField'
        for field in APP_PATCH_FIELD_KEYS:
            current: typing.Any = getattr(self, field)
            if current is not None:
                result[field] = current

        return result

    @typing.overload
    def apply_to(self, instance: AppPatch) -> AppPatch:
        """Modify other AppPatch."""

    @typing.overload
    def apply_to(self, instance: 'api_types.Application') -> 'api_types.Application':
        """Apply all values from a patch to an application description dictionary."""

    def apply_to(
            self,
            instance: typing.Union[AppPatch, 'api_types.Application'],
    ) -> typing.Union[AppPatch, 'api_types.Application']:
        """Apply patch to another instance."""
        if isinstance(instance, AppPatch):
            return attr.evolve(
                instance,
                is_available=self.is_available,
                **self.delta,
            )

        instance.update(typing.cast('api_types.Application', self.delta))
        return instance


InstallFunction = typing.Callable[['BaseInstallableApp'], None]
InstallExtensionsFunction = typing.Callable[['BaseInstallableApp'], 'process.ProcessWorker']
UpdateConfigFunction = typing.Callable[['BaseInstallableApp', str], None]


@attr.s(auto_attribs=True, cache_hash=True, frozen=True, order=False)
class BaseApp(metaclass=abc.ABCMeta):
    """
    Root description of the application.

    :param app_type: Type of the application (web application, installable, etc.).
    :param app_name: Alias of the application in a package-naming format.
    :param display_name: Name of the application to show in home tile.
    :param description: Description of the application to show in home tile.
    :param image_path: Application icon to show in home tile.
    :param config: Application configuration.
    """

    __app_types__: typing.ClassVar[typing.MutableMapping[str, typing.Type['BaseApp']]] = {}

    app_type: constants.AppType
    app_name: str
    display_name: str
    description: str
    image_path: str
    non_conda: bool
    config: 'user_config.UserConfig'
    is_available: bool
    rank: int

    def __init_subclass__(
            cls,  # pylint: disable=unused-argument
            alias: typing.Optional[constants.AppType] = None,
            **kwargs: typing.Any,
    ) -> None:
        """
        Initialize new :class:`~BaseApp` subclass.

        Used to register new classes in registry.
        """
        if alias is None:
            return

        if alias in cls.__app_types__:
            raise ValueError(f'{alias!r} is already registered')

        cls.__app_types__[alias] = cls

    @property
    def tile_definition(self) -> 'api_types.Application':  # noqa: D401
        """Definition of the application tile."""
        return {
            'app_type': self.app_type,
            'description': self.description,
            'display_name': self.display_name,
            'image_path': self.image_path,
            'name': self.app_name,
            'non_conda': self.non_conda,
            'rank': self.rank,
        }

    @classmethod
    def parse_configuration(
            cls,
            context: parsing_utils.ParsingContext,
            configuration: typing.Mapping[str, typing.Any],
            app_name: str,
    ) -> typing.Union[None, 'BaseApp', 'AppPatch']:
        """Generate applications according to the `configuration`."""
        validation_utils.of_type(typing.Mapping)(configuration)
        mirror: typing.Dict[str, typing.Any] = dict(configuration)

        app_type: str
        target_cls: typing.Optional[typing.Type[BaseApp]]
        try:
            app_type = validation_utils.pop_mapping_item(mirror, 'app_type')
        except exceptions.ValidationError:
            target_cls = None
        else:
            with exceptions.ValidationError.with_field('app_type'):
                validation_utils.is_str(app_type)
                validation_utils.of_options(*cls.__app_types__)(app_type)
            target_cls = cls.__app_types__[app_type]

        display_name: typing.Optional[str] = cls.__resolve_string_field(
            mirror,
            'display_name',
            required=target_cls is not None,
        )

        description: typing.Optional[str] = cls.__resolve_string_field(
            mirror,
            'description',
            default='',
            required=target_cls is not None,
        )

        image_path: typing.Optional[str] = cls.__resolve_string_field(
            mirror,
            'image_path',
            default='anaconda-icon-256x256.png',
            required=target_cls is not None,
        )
        if image_path is not None:
            image_path = cls.__resolve_image_path(image_path)

        rank: typing.Optional[int]
        try:
            rank = validation_utils.pop_mapping_item(mirror, 'rank')
        except exceptions.ValidationError:
            if target_cls is None:
                rank = None
            else:
                rank = 0
        else:
            validation_utils.of_type(int)(rank, field_name='rank')

        is_available: typing.Union[bool, str] = validation_utils.pop_mapping_item(mirror, 'is_available', True)
        with exceptions.ValidationError.with_field('is_available'):
            validation_utils.of_type(bool, str)(is_available)
            is_available = cls.__resolve_is_available(is_available)

        if target_cls is None:
            validation_utils.mapping_is_empty()(mirror)
            return AppPatch(
                app_name=app_name,
                display_name=display_name,
                description=description,
                image_path=image_path,
                is_available=is_available,
                rank=rank,
            )

        if not is_available:
            return None

        return target_cls._parse_configuration(  # pylint: disable=protected-access
            context=context,
            configuration=mirror,

            app_name=app_name,
            config=context.user_configuration,
            display_name=display_name,
            description=description,
            image_path=image_path,
            is_available=is_available,
            rank=rank,
        )

    @staticmethod
    @abc.abstractmethod
    def _parse_configuration(
            context: parsing_utils.ParsingContext,
            configuration: typing.MutableMapping[str, typing.Any],
            **kwargs: typing.Any,  # app_name, config, display_name, description, image_path, is_available, rank
    ) -> BaseApp:
        """Parse configuration for this particular :class:`~BaseApp`."""

    @staticmethod
    def __resolve_string_field(
            source: typing.Dict[str, typing.Any],
            field: str,
            *,
            default: typing.Optional[str] = None,
            required: bool = True,
    ) -> typing.Optional[str]:
        """Validate and resolve value of `display_name`."""
        result: typing.Optional[str] = None
        try:
            result = validation_utils.pop_mapping_item(source, field)
        except exceptions.ValidationError:
            if required:
                if default is not None:
                    result = default
                else:
                    raise
        else:
            with exceptions.ValidationError.with_field(field):
                validation_utils.is_str(result)
        return result

    @staticmethod
    def __resolve_image_path(value: str) -> str:
        """Validate and resolve value of `image_path`."""
        if os.path.isfile(value):
            return value

        result: str = os.path.join(IMAGE_PATH, value)
        if os.path.isfile(result):
            return result

        raise exceptions.ValidationError()

    @staticmethod
    def __resolve_is_available(value: typing.Union[bool, str]) -> bool:
        """Validate and resolve value of `is_available`."""
        if isinstance(value, bool):
            return value

        result: bool
        func: typing.Callable[[], bool] = import_utils.import_callable(value)
        with validation_utils.catch_exception():
            result = func()
        validation_utils.is_bool(result, 'result')
        return result


@attr.s(auto_attribs=True, cache_hash=True, collect_by_mro=True, frozen=True, order=False)
class BaseWebApp(BaseApp, alias=constants.AppType.WEB):
    """
    Common base for all external web applications.

    Most of the params reused from the :class:`~BaseApp`.

    :param url: Web page to open for the application.
    """

    app_type: constants.AppType = attr.ib(default=constants.AppType.WEB, init=False)
    non_conda: bool = attr.ib(default=True, init=False)

    url: str

    @property
    def tile_definition(self) -> 'api_types.Application':  # noqa: D401
        """Definition of the application tile."""
        result: 'api_types.Application' = super().tile_definition
        result['installed'] = True
        return result

    @classmethod
    def _parse_configuration(
            cls,
            context: parsing_utils.ParsingContext,
            configuration: typing.MutableMapping[str, typing.Any],
            **kwargs: typing.Any,  # app_name, config, display_name, description, image_path, is_available, rank
    ) -> BaseWebApp:
        """Parse configuration for this particular :class:`~BaseApp`."""
        url: str = validation_utils.pop_mapping_item(configuration, 'url')
        validation_utils.is_str(url, field_name='url')

        validation_utils.mapping_is_empty()(configuration)

        return BaseWebApp(url=url, **kwargs)


def to_string_tuple(value: typing.Iterable[typing.Any]) -> typing.Sequence[str]:
    """Convert any iterable value to a string tuple."""
    if isinstance(value, str):
        return (value,)
    return tuple(map(str, value))


@attr.s(auto_attribs=True, cache_hash=True, collect_by_mro=True, frozen=True, order=False)
class BaseInstallableApp(BaseApp, alias=constants.AppType.INSTALLABLE):
    """
    Common base for all external installable applications.

    Most of the params reused from the :class:`~BaseApp`.

    :param detector: Detector of the application location.
    :param extra_arguments: Additional arguments to pass to the application.
    :param process_api: API to spawn new processes (for extensions installation etc.).
    """

    install: typing.ClassVar[InstallFunction]

    app_type: constants.AppType = attr.ib(default=constants.AppType.INSTALLABLE, init=False)
    non_conda: bool = attr.ib(default=True, init=False)

    extra_arguments: typing.Sequence[typing.Any] = attr.ib(converter=to_string_tuple)
    _detector: 'detectors.Source'
    _process_api: process.WorkerManager
    _location: typing.Optional[detectors.DetectedApplication] = attr.ib(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        """Additional initialization of the class."""
        context: detectors.DetectorContext = detectors.DetectorContext(
            app_name=self.app_name,
            user_configuration=self.config,
        )
        location: 'typing.Optional[detectors.DetectedApplication]'
        for location in self._detector(context=context):
            if location.complete:
                break
        else:
            location = None
        object.__setattr__(self, '_location', location)

        self.config.set('applications', f'{self.app_name}_path', self.root or '')

    @property
    def tile_definition(self) -> 'api_types.Application':  # noqa: D401
        """Definition of the application tile."""
        result: 'api_types.Application' = super().tile_definition
        result['command'] = self.executable or ''
        result['extra_arguments'] = self.extra_arguments
        result['installed'] = self.is_installed
        result['versions'] = self.versions
        return result

    @property
    def executable(self) -> typing.Optional[str]:  # noqa: D401
        """Command to execute the application."""
        if self._location is None:
            return None

        from anaconda_navigator.utils import launch  # pylint: disable=import-outside-toplevel

        return launch.safe_argument(self._location.executable)

    @property
    def is_installation_enabled(self) -> bool:  # noqa: D401
        """Application can be installed."""
        return hasattr(self, 'install')

    @property
    def is_installed(self) -> bool:  # noqa: D401
        """Application is installed in local system."""
        return bool(self.executable)

    @property
    def root(self) -> typing.Optional[str]:  # noqa: D401
        """Directory, in which application is installed."""
        if self._location is None:
            return None
        return self._location.root

    @property
    def version(self) -> typing.Optional[str]:  # noqa: D401
        """Version of the installed application."""
        if self._location is None:
            return None
        return self._location.version

    @property
    def versions(self) -> typing.Sequence[str]:  # noqa: D401
        """List of available application versions."""
        if self._location is None:
            return []
        return [self._location.version]

    @classmethod
    def _parse_configuration(
            cls,
            context: parsing_utils.ParsingContext,
            configuration: typing.MutableMapping[str, typing.Any],
            **kwargs: typing.Any,  # app_name, config, display_name, description, image_path, is_available, rank
    ) -> BaseInstallableApp:
        """Parse configuration for this particular :class:`~BaseApp`."""
        raw_detector: typing.Union[str, typing.Mapping[str, typing.Any]]
        raw_detector = validation_utils.pop_mapping_item(configuration, 'detector')
        with exceptions.ValidationError.with_field('detector'):
            validation_utils.of_type(str, typing.Mapping)(raw_detector)
            detector: detectors.Source = cls.__resolve_detector(raw_detector)

        extra_arguments: typing.Union[None, str, typing.Sequence[str]]
        extra_arguments = validation_utils.pop_mapping_item(configuration, 'extra_arguments', None)
        with exceptions.ValidationError.with_field('extra_arguments'):
            validation_utils.of_type(str, typing.Sequence, allow_none=True)(extra_arguments)
            extra_arguments = cls.__resolve_extra_arguments(extra_arguments)

        result: BaseInstallableApp = BaseInstallableApp(
            detector=detector,
            process_api=context.process_api,
            extra_arguments=extra_arguments,
            **kwargs,
        )

        raw_method: typing.Optional[str] = validation_utils.pop_mapping_item(configuration, 'install', None)
        with exceptions.ValidationError.with_field('install'):
            validation_utils.is_str_or_none(raw_method)
            if raw_method is not None:
                object.__setattr__(
                    result,
                    'install',
                    types.MethodType(cls.__resolve_install(raw_method), result),
                )

        method_name: str
        for method_name in ('install_extensions', 'update_config'):
            raw_method = validation_utils.pop_mapping_item(configuration, method_name, None)
            validation_utils.is_str_or_none(raw_method)
            if raw_method is not None:
                object.__setattr__(
                    result,
                    method_name,
                    types.MethodType(import_utils.import_callable(raw_method), result),
                )

        validation_utils.mapping_is_empty()(configuration)

        return result

    @staticmethod
    def __resolve_detector(value: typing.Union[str, typing.Mapping[str, typing.Any]]) -> detectors.Source:
        """Validate and resolve value of `detector`."""
        result: detectors.Source
        if isinstance(value, str):
            resolver: typing.Callable[[], detectors.Source] = import_utils.import_callable(value)
            with validation_utils.catch_exception():
                result = resolver()
            validation_utils.of_type(detectors.Source)(result, field_name='result')
            return result

        result = typing.cast(detectors.Source, detectors.Detector.parse_configuration(value))
        validation_utils.of_type(detectors.Source)(result)
        return result

    @staticmethod
    def __resolve_extra_arguments(value: typing.Union[None, str, typing.Sequence[str]]) -> typing.Sequence[str]:
        """Validate and resolve value of `extra_arguments`."""
        if value is None:
            return ()

        if isinstance(value, str):
            result: typing.Sequence[str]
            resolver: typing.Callable[[], typing.Sequence[str]] = import_utils.import_callable(value)
            with validation_utils.catch_exception():
                result = resolver()
            validation_utils.is_str_sequence(result, field_name='result')
            return result

        validation_utils.each_item(validation_utils.is_str)(value)
        return value

    @staticmethod
    def __resolve_install(value: str) -> InstallFunction:
        """Validate and resolve value of `install`."""
        if re.match(r'^[a-z][a-z0-9.+-]*://', value):
            def install(self: 'BaseInstallableApp') -> None:  # pylint: disable=unused-argument
                """Install application"""
                webbrowser.open(value)

            return install

        return import_utils.import_callable(value)

    def install_extensions(self) -> 'process.ProcessWorker':
        """Install app extensions."""
        return self._process_api.create_process_worker(['python', '--version'])

    def update_config(self, prefix: str) -> None:
        """Update user config to use selected Python prefix interpreter."""
