# -*- coding: utf-8 -*-

"""Utilities to manage external application configuration files."""

from __future__ import annotations

__all__ = [
    'Action', 'retrieve_configuration_locations', 'read_configuration_file', 'load_configuration_file',
    'merge_configurations', 'apply_configuration', 'load_configuration',
]

import contextlib
import enum
import functools
import os
import re
import typing
import yaml
from anaconda_navigator import config as navigator_config
from anaconda_navigator.static import content as static_content
from anaconda_navigator.utils.logs import logger
from . import base
from . import exceptions
from . import validation_utils

if typing.TYPE_CHECKING:
    from . import parsing_utils


EMPTY_MAPPING: typing.Mapping[typing.Any, typing.Any] = {}

AppRecord = typing.MutableMapping[str, typing.Any]
AppConfiguration = typing.MutableMapping[str, AppRecord]


class Action(str, enum.Enum):
    """Possible actions on application configurations."""

    PATCH = 'patch'
    PUT = 'put'
    DISCARD = 'discard'


ActionFunc = typing.Callable[[typing.Optional[AppRecord], AppRecord], typing.Optional[AppRecord]]


def apply_discard(
        existing: typing.Optional[AppConfiguration],  # pylint: disable=unused-argument
        update: AppConfiguration,  # pylint: disable=unused-argument
) -> typing.Optional[AppConfiguration]:
    """
    Discard existing record.

    .. warning::

        Operation may update provided arguments!
    """
    return None


def apply_patch(
        existing: typing.Optional[AppConfiguration],
        update: AppConfiguration,
) -> typing.Optional[AppConfiguration]:
    """
    Apply patch to the `existing` record.

    .. warning::

        Operation may update provided arguments!
    """
    if existing is None:
        return update
    existing.update(update)
    return existing


def apply_put(
        existing: typing.Optional[AppConfiguration],  # pylint: disable=unused-argument
        update: AppConfiguration,
) -> typing.Optional[AppConfiguration]:
    """
    Put `update` instead of `existing` record.

    .. warning::

        Operation may update provided arguments!
    """
    return update


ACTIONS: typing.Final[typing.Mapping[str, ActionFunc]] = {
    Action.DISCARD: apply_discard,
    Action.PATCH: apply_patch,
    Action.PUT: apply_put,
}


@contextlib.contextmanager
def validation_to_warning(path: str) -> typing.Iterator[None]:
    """Convert Validation errors into warnings."""
    try:
        yield
    except exceptions.ValidationError as exception:
        logger.warning('%s: %s', path, exception.details)


def retrieve_configuration_locations() -> typing.Sequence[str]:
    """Retrieve list of configuration file locations."""
    result: typing.List[str] = [static_content.EXTERNAL_APPS_CONF_PATH]

    item: str
    root: str = os.path.join(navigator_config.CONF_PATH, 'applications')
    try:
        os.makedirs(root, exist_ok=True)
        for item in sorted(os.listdir(root)):
            item = os.path.join(root, item)
            if os.path.isfile(item):
                result.append(item)
    except OSError:
        pass

    return result


def read_configuration_file(path: str) -> AppConfiguration:
    """Read content of the configuration file."""
    if re.match(r'^[a-z][a-z0-9.+-]*://', path):
        raise NotImplementedError('web resources are not supported yet')

    try:
        stream: typing.TextIO
        with open(path, encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    except OSError:
        logger.warning('%s: unable to read file', path, exc_info=True)
        return {}
    except yaml.YAMLError:
        logger.warning('%s: unable to parse file', path, exc_info=True)
        return {}


def load_configuration_file(path: str) -> AppConfiguration:
    """Load and cleanup configuration file."""
    result: AppConfiguration = {}

    with validation_to_warning(path):
        raw_result: AppConfiguration = read_configuration_file(path)
        validation_utils.of_type(typing.MutableMapping)(raw_result)

        key: str
        value: AppRecord
        for key, value in raw_result.items():
            key = str(key)

            with validation_to_warning(path):
                if not re.fullmatch(r'^[a-z0-9_]+$', key):
                    raise exceptions.ValidationError(f'{key!r} is invalid key to identify application with')

                with exceptions.ValidationError.with_field(key):
                    validation_utils.of_type(typing.MutableMapping)(value)

                    action: str = validation_utils.pop_mapping_item(value, 'action', Action.PATCH)
                    validation_utils.is_str(action, field_name='action')
                    validation_utils.of_options(*ACTIONS)(action, field_name='action')

                result[key] = value

    return result


def merge_configurations(existing: AppConfiguration, update: AppConfiguration) -> AppConfiguration:
    """Update `existing` configuration with values from `update`."""
    key: str
    value: AppRecord
    for key, value in update.items():
        action: ActionFunc = ACTIONS[value.get('action', Action.PATCH)]
        current: typing.Optional[AppRecord] = action(existing.get(key, None), value)
        if current is None:
            existing.pop(key, None)
        else:
            existing[key] = current
    return existing


def apply_configuration(
        configuration: AppConfiguration,
        context: 'parsing_utils.ParsingContext',
) -> typing.Sequence[typing.Union[base.BaseApp, base.AppPatch]]:
    """Initialize application according to the configuration."""
    result: typing.List[typing.Union[base.BaseApp, base.AppPatch]] = []

    key: str
    value: AppConfiguration
    for key, value in configuration.items():
        with validation_to_warning('*collected configuration*'), exceptions.ValidationError.with_field(key):
            addition: typing.Union[None, base.BaseApp, base.AppPatch] = base.BaseApp.parse_configuration(
                context=context,
                configuration=value,
                app_name=key,
            )
            if addition is not None:
                result.append(addition)

    return result


def load_configuration(
        *paths: str,
        context: 'parsing_utils.ParsingContext',
) -> typing.Sequence[typing.Union[base.BaseApp, base.AppPatch]]:
    """Load configuration for the applications."""
    if not paths:
        paths = tuple(retrieve_configuration_locations())

    return apply_configuration(
        configuration=functools.reduce(merge_configurations, map(load_configuration_file, paths)),
        context=context,
    )
