# -*- coding: utf-8 -*-

"""Storage for detected and fetched widgets."""

from __future__ import annotations

__all__ = ['PartnerPool', 'POOL', 'UPDATER']

import json
import re
import typing
from urllib import parse
import yaml
from qtpy import QtCore
from anaconda_navigator import config
from anaconda_navigator.config import preferences
from anaconda_navigator.utils import extra_collections
from anaconda_navigator.utils.logs.loggers import logger
from anaconda_navigator.utils import singletons
from anaconda_navigator.utils import workers
from . import core
from . import resources
from . import simple_advertisement

if typing.TYPE_CHECKING:
    from qtpy import QtWidgets


class PartnerConfiguration(typing.TypedDict, total=False):
    """Dictionary of an additional configuration, that might be used by advertisements and the application."""

    utm_campaign: str


class PartnerPool(typing.Mapping[core.PartnerWidgetPlacement, 'QtWidgets.QWidget']):
    """Collection of widgets for partner attribution."""

    __slots__ = ('__content', '__settings')

    def __init__(self) -> None:
        """Initialize new :class:`~PartnerPool` instance."""
        self.__settings: core.PartnerSettings = core.PartnerSettings()
        self.__content: typing.Final[typing.Dict[core.PartnerWidgetPlacement, core.PartnerWidget]] = {}

    @property
    def settings(self) -> core.PartnerSettings:  # noqa: D401
        """Common configuration that might be used by widgets and different application components."""
        return self.__settings

    def _set_settings(self, value: typing.Union[core.PartnerSettings, 'core.SettingsDict']) -> None:
        """Update value of the `configuration`."""
        if not isinstance(value, core.PartnerSettings):
            value = core.PartnerSettings(value)
        self.__settings = value

    def register(self, widget: core.PartnerWidget) -> None:
        """Register new widget in the pool."""
        self.__content[widget.placement] = widget

    def __getitem__(self, key: core.PartnerWidgetPlacement) -> typing.Optional[QtWidgets.QWidget]:
        """Retrieve widget that should be placed in `key` location."""
        value: typing.Optional[core.PartnerWidget] = self.__content.get(key, None)
        if value is None:
            return None
        return value.widget()

    def __iter__(self) -> typing.Iterator[core.PartnerWidgetPlacement]:
        """Iterate through available location options."""
        return iter(core.PartnerWidgetPlacement)

    def __len__(self) -> int:
        """Total number of """
        return len(core.PartnerWidgetPlacement)


POOL: typing.Final[PartnerPool] = PartnerPool()


class Updater(QtCore.QObject):
    """Core for updating advertisement information."""

    sig_updated = QtCore.Signal()

    def __init__(self) -> None:
        """Initialize new :class:`~Updater` instance."""
        super().__init__()

        self.__partner_configuration: typing.Optional['PartnerConfiguration'] = None

    @property
    def partner_configuration(self) -> 'PartnerConfiguration':  # noqa: D401
        """Configuration for partners."""
        path: str
        if self.__partner_configuration is None:
            for path in preferences.AD_CONFIGURATION_PATHS:
                try:
                    stream: typing.TextIO
                    with open(path, 'rt', encoding='utf8') as stream:
                        result: 'PartnerConfiguration' = yaml.safe_load(stream)

                        if not isinstance(result, typing.Mapping):
                            logger.exception('broken partner configuration file at %s', path)  # type: ignore
                            continue

                        self.__partner_configuration = result
                        break

                except FileNotFoundError:
                    continue

                except (OSError, yaml.YAMLError):
                    logger.exception('broken partner configuration file at %s', path)
                    continue

            else:
                self.__partner_configuration = {}

            # normalize utm_campaign value
            self.__partner_configuration.setdefault('utm_campaign', '')
            if not isinstance(self.__partner_configuration['utm_campaign'], str):
                logger.warning('utm_campaign in partner configuration must be a string')  # type: ignore
                self.__partner_configuration['utm_campaign'] = ''
            if len(self.__partner_configuration['utm_campaign']) > 64:
                logger.warning('utm_campaign value is too long')
                self.__partner_configuration['utm_campaign'] = ''

        return self.__partner_configuration

    @property
    def sources(self) -> extra_collections.OrderedSet[str]:  # noqa: D401
        """List of URLs to fetch ads from."""
        advertisement_url: str = config.CONF.get('main', 'advertisement_url')

        # prepare parameters for URL formatting
        source_values: typing.Mapping[str, str] = {
            'partner_identity': parse.quote(self.partner_configuration['utm_campaign']),
        }
        source_placeholders: typing.Mapping[str, str] = {key: '✫' for key in source_values}
        source_values = {key: value for key, value in source_values.items() if value}

        # format and collect URLs from preferences
        source: str
        result: extra_collections.OrderedSet[str] = extra_collections.OrderedSet()
        for source in preferences.AD_SOURCES:
            pending: bool = False
            try:
                result.add(source.format(**source_values))
            except KeyError:
                pending = True

            if re.fullmatch(re.escape(source.format(**source_placeholders)).replace('✫', '.+'), advertisement_url):
                if pending:
                    result.add(advertisement_url)
                advertisement_url = ''

        # add custom URL from user preferences
        # it would be automatically discarded in case it is already added in one of previous steps
        result.add(advertisement_url)

        # make sure empty URL is not in the queue
        result.discard('')

        return result

    @workers.Task
    def update(self) -> None:
        """Fetch and apply advertisement configuration to the :data:`~POOL`."""
        configuration: core.ConfigurationDict = self.__fetch_configuration()

        # make configuration available to everything
        POOL._set_settings(configuration['settings'])  # pylint: disable=protected-access

        # create widgets according to the configuration
        for widget in configuration['widgets']:
            instance: core.PartnerWidget

            # NOTE: to support additional built-in advertisement types:
            #
            #       if widget.get('type', 'simple') == 'simple':
            instance = simple_advertisement.CompositeAdvertisementWidget(
                definition=typing.cast('simple_advertisement.CompositeAdvertisementDict', widget),
                settings=POOL.settings,
            )

            POOL.register(instance)

        # NOTE: NAV-753 should be here
        #
        #       Parse other locations for widgets, initialize them with `POOL.settings` and then `POOL.register` them.

        self.sig_updated.emit()

    def __fetch_configuration(self) -> core.ConfigurationDict:
        """Collect configuration for advertisements."""
        result: core.ConfigurationDict = {
            'widgets': [],
            'settings': {
                'url_parameters': {},
            },
        }

        # process advertisement URLs
        for advertisement_url in self.sources:
            try:
                advertisement_content: typing.Optional[bytes] = resources.load_resource(advertisement_url)
                if not advertisement_content:
                    continue

                # Fetch configuration
                definition: 'simple_advertisement.CompositeAdvertisementDict' = {
                    'placement': core.PartnerWidgetPlacement.BOTTOM_LEFT_CORNER,
                    'advertisements': json.loads(advertisement_content),
                }
                if not definition['advertisements']:
                    continue

                # Store configuration
                result['widgets'].append(definition)

                # Extract utm parameters from the redirect_url
                item: simple_advertisement.SimpleAdvertisementDict
                for item in definition['advertisements']:
                    if 'redirect_url' not in item:
                        continue

                    result['settings']['url_parameters'].update(
                        (key, value)
                        for key, value in parse.parse_qsl(parse.urlparse(item['redirect_url']).query)
                        if key.startswith('utm_')
                    )
                    break  # NOTE: remove if we need to combine data from all URLs and not just the first valid one

                # Remember the working API url
                config.CONF.set('main', 'advertisement_url', advertisement_url)
                break

            except BaseException:  # pylint: disable=broad-except
                logger.exception('failed to initialize advertisement')

        # apply `utm_campaign` from `partner_configuration`
        if self.partner_configuration['utm_campaign']:
            result['settings']['url_parameters']['utm_campaign'] = self.partner_configuration['utm_campaign']

        # apply default values for required parameters
        result['settings']['url_parameters'].setdefault('utm_source', 'anaconda_navigator')

        return result


UPDATER: typing.Final[singletons.Singleton[Updater]] = singletons.SingleInstanceOf(Updater)
