# -*- coding: utf-8 -*-

"""Helper component to show messages from the Navigator."""

from __future__ import annotations

__all__ = ['NotificationsComponent']

import typing

from anaconda_navigator.utils import notifications
from anaconda_navigator.widgets import dialogs
from . import common

if typing.TYPE_CHECKING:
    from qtpy import QtWidgets
    from anaconda_navigator.widgets import main_window


class NotificationsComponent(common.Component):
    """Component for notification management."""

    __alias__ = 'notifications'

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`NotificationsComponent` instance."""
        super().__init__(parent=parent)
        self.__listener: typing.Final[notifications.NotificationListener] = notifications.NotificationListener(self)
        self.__listener.sig_notification.connect(self.__notification)

    def __notification(self, notification: notifications.Notification) -> None:
        """"""
        self.info(text=notification.message, title=notification.caption)

    def info(self, text: str = '', title: str = '') -> None:
        """Show a dialog with information message."""
        dialog: 'QtWidgets.QDialog' = dialogs.MessageBoxInformation(text=text, title=title, parent=self.main_window)
        dialog.setMinimumWidth(480)
        dialog.exec_()

    def setup(self, worker: typing.Any, output: typing.Any, error: str, initial: bool) -> None:
        """Perform component configuration from `conda_data`."""
        notifications.NOTIFICATION_QUEUE.attach(self.__listener)
