# -*- coding: utf-8 -*-

"""Solvers for invalid ssl_certificate setting."""

from __future__ import annotations

__all__ = ()

import html
import os

from anaconda_navigator.utils import notifications
from . import core


@core.CONFIGURATION_POOL.register
def solve_missing_certificate(context: core.ConfigurationContext) -> notifications.Notification | None:
    """Check if `ssl_ceritificate` is missing from file system."""
    ssl_certificate: str = context.config.get('main', 'ssl_certificate')
    if (not ssl_certificate) or os.path.exists(ssl_certificate):
        return None

    context.config.set('main', 'ssl_certificate', '')
    return notifications.Notification(
        message=f'SSL certificate is no longer available:<br>- {html.escape(ssl_certificate)}',
        caption='Broken Navigator configuration',
        tags=('navigator', 'ssl_certificate'),
    )
