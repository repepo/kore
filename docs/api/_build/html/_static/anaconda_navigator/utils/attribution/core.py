# -*- coding: utf-8 -*-

"""Common definitions of the partner attribution configuration."""

from __future__ import annotations

__all__ = ['PartnerWidgetPlacement', 'PartnerSettings', 'PartnerWidget']

import abc
import enum
import fnmatch
import typing

from qtpy import QtWidgets

from anaconda_navigator.utils import url_utils
from . import configuration


class WidgetDict(typing.TypedDict):
    """
    Common definition for different advertisement widgets.

    Custom configuration should inherit this type and extend it with additional details.

    .. note::

        To support additional built-in advertisement types: - additional `kind` field might be added.

        This property should behave like optional, i.e. be requested via `.get('kind', 'simple')` when the type will
        be detected.
    """

    placement: 'PartnerWidgetPlacement'


class SettingsDict(typing.TypedDict):
    """Dictionary of an additional configuration, that might be used by advertisements and the application."""

    url_parameters: typing.MutableMapping[str, str]


class ConfigurationDict(typing.TypedDict):
    """Complete format of a general advertisement configuration."""

    widgets: typing.MutableSequence[WidgetDict]
    settings: SettingsDict


WidgetT_co = typing.TypeVar('WidgetT_co', bound=QtWidgets.QWidget, covariant=True)


class PartnerWidgetPlacement(str, enum.Enum):
    """Location options to place advertisement widget to."""

    BOTTOM_LEFT_CORNER = 'bottom_left_corner'


class PartnerSettings:
    """Common configuration for advertisements and application."""

    __slots__ = ('__content',)

    def __init__(self, content: typing.Optional['SettingsDict'] = None) -> None:
        """Initialize new :class:`~PartnerConfiguration` instance."""
        if content is None:
            content = {
                'url_parameters': {},
            }

        self.__content: typing.Final[SettingsDict] = content

    @property
    def url_parameters(self) -> typing.Mapping[str, str]:  # noqa: D401
        """Additional URL parameters that should be injected into urls."""
        return self.__content['url_parameters']

    def inject_url_parameters(self, url: str, force: bool = False, **kwargs: str) -> str:
        """
        Extend URL with a stored parameters.

        :param url: Url to inject parameters to.
        :param force: Apply new parameters to any URL.

                       Otherwise, parameters will be applied only to URLs, which domains are listed in
                       :data:`~anaconda_navigator.widgets.attribution.configuration.APPLICABLE_HOSTS`.
        :param kwargs: Additional parameters to also include in the url

                       These values will override parameters with the same keys from
                       :attr:`~PartnerSettings.url_parameters`.
        :return: Modified URL.
        """
        if not force:
            netloc: str = url_utils.netloc(url)

            applicable_host: str
            for applicable_host in configuration.APPLICABLE_HOSTS:
                if fnmatch.fnmatch(netloc, applicable_host):
                    break
            else:
                return url

        return url_utils.inject_query_parameters(url=url, params={**self.url_parameters, **kwargs})


class PartnerWidget(typing.Generic[WidgetT_co], metaclass=abc.ABCMeta):
    """Base for all advertisement widgets."""

    __slots__ = ('__placement', '__settings')

    def __init__(self, settings: PartnerSettings, placement: PartnerWidgetPlacement) -> None:
        """Initialize new :class:`~PartnerWidget` instance."""
        self.__placement: typing.Final[PartnerWidgetPlacement] = PartnerWidgetPlacement(placement)
        self.__settings: typing.Final[PartnerSettings] = settings

    @property
    def settings(self) -> 'PartnerSettings':  # noqa: D401
        """Common configuration for this advertisement."""
        return self.__settings

    @property
    def placement(self) -> PartnerWidgetPlacement:  # noqa: D401
        """Location of the advertisement."""
        return self.__placement

    @abc.abstractmethod
    def widget(self) -> WidgetT_co:
        """Generate new widget to use in Qt application."""
