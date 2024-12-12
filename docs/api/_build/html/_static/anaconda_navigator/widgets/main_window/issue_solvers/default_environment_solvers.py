# -*- coding: utf-8 -*-

"""Solvers for invalid default_env setting."""

from __future__ import annotations

__all__ = ()

import html
import os

from anaconda_navigator.utils import notifications
from . import core


@core.CONFIGURATION_POOL.register
def solve_missing_environment(context: core.ConfigurationContext) -> notifications.Notification | None:
    """Check if `default_env` is missing from file system."""
    default_env: str = context.config.get('main', 'default_env')
    if default_env and os.path.isdir(default_env):
        return None

    context.config.set('main', 'default_env', context.api.ROOT_PREFIX)

    if default_env:
        return notifications.Notification(
            message=f'Environment is no longer available:<br>- {html.escape(default_env)}',
            caption='Broken Navigator configuration',
            tags=('navigator', 'default_env'),
        )

    return None


@core.CONFLICT_POOL.register
def solve_unknown_environment(context: core.ConflictContext) -> notifications.Notification | None:
    """Check if `default_env` is unknown to conda."""
    default_env: str = context.config.get('main', 'default_env')
    if default_env in context.conda_info['processed_info']['__environments']:
        return None

    context.config.set('main', 'default_env', context.api.ROOT_PREFIX)

    if default_env:
        return notifications.Notification(
            message=f'Environment is no longer available:<br>- {html.escape(default_env)}',
            caption='Broken Navigator configuration',
            tags=('navigator', 'default_env'),
        )

    # Just in case, as it should not trigger under normal circumstances
    return notifications.Notification(
        message='Default environment is updated',
        caption='Broken Navigator configuration',
        tags=('navigator', 'default_env'),
    )
