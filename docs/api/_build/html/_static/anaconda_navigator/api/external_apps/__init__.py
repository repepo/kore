# -*- coding: utf-8 -*-

"""Common interface for third-party applications."""

from __future__ import annotations

import typing
from .base import *
from .constants import *
from . import config_utils
from . import parsing_utils
if typing.TYPE_CHECKING:
    from anaconda_navigator.api import process
    from anaconda_navigator.config import user as user_config


class ApplicationCollection(typing.NamedTuple):
    """Collection of collected applications."""

    installable_apps: typing.Mapping[str, BaseInstallableApp]
    web_apps: typing.Mapping[str, BaseWebApp]
    app_patches: typing.Mapping[str, AppPatch]


APPLICATION_CACHE: typing.Optional[ApplicationCollection] = None


def get_applications(
        configuration: typing.Optional['user_config.UserConfig'] = None,
        process_api: typing.Optional['process.WorkerManager'] = None,
        *,
        cached: bool = False,
) -> ApplicationCollection:
    """Initialize all :class:`~BaseInstallableApp` instances for apps listed in root config."""
    global APPLICATION_CACHE  # pylint: disable=global-statement
    if cached and (APPLICATION_CACHE is not None):
        return APPLICATION_CACHE

    if configuration is None:
        raise TypeError('`configuration` must be provided')
    if process_api is None:
        raise TypeError('`process_api` must be provided')

    result: ApplicationCollection
    context = parsing_utils.ParsingContext(
        process_api=process_api,
        user_configuration=configuration,
    )
    apps: typing.Sequence[typing.Union[BaseApp, AppPatch]] = config_utils.load_configuration(context=context)
    APPLICATION_CACHE = result = ApplicationCollection(
        installable_apps={
            item.app_name: item
            for item in apps
            if isinstance(item, BaseInstallableApp)
        },
        web_apps={
            item.app_name: item
            for item in apps
            if isinstance(item, BaseWebApp)
        },
        app_patches={
            item.app_name: item
            for item in apps
            if isinstance(item, AppPatch)
        },
    )
    return result
