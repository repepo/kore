# -*- coding: utf-8 -*-

"""Solvers for the invalid SSL certificate paths."""

from __future__ import annotations

__all__ = ()

import html
import os
import typing

from anaconda_navigator.utils import notifications
from . import core


@core.POOL.register
def solve_ssl_paths(configuration: typing.Any) -> notifications.Notification | None:
    """Remove invalid paths to SSL certificates."""
    broken_certificates: set[str] = set()

    key: str
    for key in ('ssl_verify', 'verify_ssl'):
        value: str | bool | None = configuration.get(key, None)
        if isinstance(value, str) and (not os.path.exists(value)):
            broken_certificates.add(value)
            del configuration[key]

    if broken_certificates:
        message: str = 'SSL certificate(s) are no longer available:'
        for key in broken_certificates:
            message += f'<br>- {html.escape(key)}'

        return notifications.Notification(
            message=message,
            caption='Broken Anaconda configuration',
            tags=('anaconda', 'ssl_certificate'),
        )

    return None
