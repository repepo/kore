# -*- coding: utf8 -*-

"""Helper utilities to catch anaconda-client issues."""

from __future__ import annotations

__all__ = ['catch_and_notify']

import collections.abc
import contextlib
import warnings

import binstar_client

from anaconda_navigator.utils import notifications


@contextlib.contextmanager
def catch_and_notify(*, reraise: bool = False) -> collections.abc.Iterator[None]:
    """
    Catch exceptions/warnings and init :class:`~MessageBoxInformation` that will be passed to the notification queue.

    The dialog will not be passed to the notification queue if `once` is set to `True` and dialog is already queued.
    """
    try:
        warns: list[warnings.WarningMessage]
        with warnings.catch_warnings(record=True) as warns:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            yield

        warning: warnings.WarningMessage
        for warning in warns:
            notifications.NOTIFICATION_QUEUE.push(
                message=str(warning.message),  # pylint: disable=no-member
                caption='Anaconda warning',
                tags=('anaconda', 'warning')
            )

    except binstar_client.BinstarError as error:
        notifications.NOTIFICATION_QUEUE.push(
            message=str(error.message),  # pylint: disable=no-member
            caption='Anaconda error',
            tags=('anaconda', 'warning')
        )
        if reraise:
            raise
