# -*- coding: utf-8 -*-

"""Solvers for Conda configuration entries, that are not valid."""

from __future__ import annotations

__all__ = ()

import html
import typing

from anaconda_navigator.utils import configurations
from anaconda_navigator.utils import notifications
from . import core

if typing.TYPE_CHECKING:
    from .. import types as conda_types


@core.POOL.register_function(filters={'exception_name': 'CustomValidationError', 'parameter_name': 'ssl_verify'})
def solve_ssl_verify(error: 'conda_types.CondaValidationErrorOutput') -> notifications.Notification:
    """
    Solve issues with invalid `ssl_verify` preference.

    Solution: remove invalid values.
    """
    updater: configurations.Updater
    with configurations.YamlUpdater(path=error['source']) as updater:
        message: str = 'SSL certificate(s) are no longer available'

        first: bool = True
        value: str | None
        for value in (
                updater.content.pop('ssl_verify', None),
                updater.content.pop('verify_ssl', None),
        ):
            if value is not None:
                if first:
                    message += ':'
                    first = False
                message += f'<br>- {html.escape(value)}'

        return notifications.Notification(
            message=message,
            caption='Broken Conda configuration',
            tags=('conda', 'ssl_certificate'),
        )
