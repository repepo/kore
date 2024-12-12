# -*- coding: utf-8 -*-

"""Extra solvers to migrate trusted_servers to domain-only format."""

from __future__ import annotations

__all__ = ()

import collections.abc
import configparser
import contextlib
import html
import typing

from anaconda_navigator.utils import notifications
from . import core


class MigrationItem(typing.NamedTuple):
    """Instructions on how a preference should be migrated."""

    section: str
    from_option: str
    to_option: str


MIGRATION: typing.Final[collections.abc.Sequence[MigrationItem]] = (
    MigrationItem(section='main', from_option='team_edition_api_url', to_option='anaconda_server_api_url'),
    MigrationItem(section='main', from_option='team_edition_token', to_option='anaconda_server_token'),
    MigrationItem(section='main', from_option='team_edition_token_id', to_option='anaconda_server_token_id'),
    MigrationItem(section='main', from_option='commercial_edition_url', to_option='anaconda_professional_url'),
)


@core.CONFIGURATION_POOL.register
def migrate_branding(context: core.ConfigurationContext) -> notifications.Notification | None:
    """Migrate options affected by branding changes."""
    item: MigrationItem
    issues: list[MigrationItem] = []
    for item in MIGRATION:
        new_value: str | None
        try:
            new_value = context.config.get(item.section, item.from_option)
        except configparser.NoOptionError:
            pass
        else:
            with contextlib.suppress(configparser.NoOptionError):
                default: typing.Any = context.config.get_default(item.section, item.to_option)
                if context.config.get(item.section, item.to_option) != default:
                    issues.append(item)
            context.config.set(item.section, item.to_option, new_value)
            context.config.remove_option(item.section, item.from_option)

    if issues:
        return notifications.Notification(
            message='<br>'.join(
                f'<b>{html.escape(issue.to_option)}</b> was overwritten by <b>{html.escape(issue.from_option)}</b>'
                for issue in issues
            ),
            caption='Conflicting preferences in navigator configuration',
            tags=('navigator', 'branding_overlap'),
        )
    return None
