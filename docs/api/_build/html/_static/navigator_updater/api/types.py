# -*- coding: utf-8 -*-

"""Collection of additional data types used by APIs."""

__all__ = ()

import typing

from navigator_updater.utils import constants


ApplicationName = str
Version = str


class RawApplication(typing.TypedDict, total=False):
    """
    Application description from external services (conda).

    Transformed into :class:`~Application` in :mod:`~navigator_updater.api.anaconda_api`
    """

    versions: typing.Sequence[Version]
    size: typing.Mapping[Version, int]
    type: typing.Mapping[Version, typing.Literal['app']]
    app_entry: typing.Mapping[Version, str]
    app_type: typing.Mapping[Version, typing.Literal[None, 'desk', 'web']]
    latest_version: Version

    name: str
    description: str
    image_path: str


class Application(typing.TypedDict, total=False):
    """
    Common description of a third-party application.

    Used for all home page tiles.
    """

    app_type: constants.AppType
    name: ApplicationName
    display_name: str
    description: str
    image_path: str

    rank: int

    versions: typing.Sequence[str]
    version: str

    non_conda: bool
    installed: bool
    command: str
    extra_arguments: typing.Sequence[typing.Any]
