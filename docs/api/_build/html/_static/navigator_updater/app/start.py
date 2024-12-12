#!/usr/bin/env python

# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2016 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Application start."""

# Standard library imports
import signal
import sys

# Third party imports
from qtpy import PYQT5
from qtpy.QtCore import QCoreApplication, QEvent, QObject, Qt
from qtpy.QtGui import QIcon

# Local imports
from navigator_updater.config import LOCKFILE, MAC
from navigator_updater.external import filelock
from navigator_updater.static import images
from navigator_updater.static.fonts import load_fonts
from navigator_updater.utils.logs import setup_logger, LOGGER_CONFIG
from navigator_updater.utils.qthelpers import qapplication
from navigator_updater.widgets.dialogs import MessageBoxInformation
from navigator_updater.widgets.dialogs.main_dialog import MainDialog

# For retina displays on qt5
if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)


def except_hook(cls, exception, traceback):
    """Custom except hook to avoid crashes on PyQt5."""
    sys.__excepthook__(cls, exception, traceback)


def set_application_icon():
    """Set application icon."""
    global app  # pylint: disable=global-variable-not-assigned,invalid-name
    app_icon = QIcon(images.ANACONDA_LOGO)
    app.setWindowIcon(app_icon)


class EventEater(QObject):  # pylint: disable=too-few-public-methods
    """Event filter for application state."""

    def __init__(self, application):
        """Event filter for application state."""
        super().__init__()
        self.app = application

    def eventFilter(self, ob, event):  # pylint: disable=invalid-name
        """Qt override."""
        if (event.type() == QEvent.ApplicationActivate) and MAC and self.app.window.setup_ready:
            self.app.window.show()
            if self.app.window.isMaximized():
                self.app.window.showMaximized()
            elif self.app.window.isFullScreen():
                self.app.window.showFullScreen()
            else:
                self.app.window.showNormal()
            return True

        return super().eventFilter(ob, event)


def start_app(options):  # pragma: no cover
    """Main application entry point."""
    # Setup logger
    LOGGER_CONFIG.level = options.log_level
    setup_logger()

    # Monkey patching sys.excepthook to avoid crashes in PyQt 5.5+
    if PYQT5:
        sys.excepthook = except_hook

    global app  # pylint: disable=invalid-name,global-variable-undefined,global-statement
    app = qapplication(test_time=60)
    set_application_icon()
    load_fonts(app)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Create file lock
    lock = filelock.FileLock(LOCKFILE)
    try:
        with lock.acquire(timeout=1.0):
            dlg = MainDialog(
                latest_version=options.latest_version, prefix=options.prefix
            )
            app.window = dlg
            event_eater = EventEater(app)
            app.installEventFilter(event_eater)
            sys.exit(dlg.exec_())
    except filelock.Timeout:
        msgbox = MessageBoxInformation(
            title='Anaconda Navigator Updater Information',
            text='There is an instance of Anaconda Navigator Updater already running.',
        )
        sys.exit(msgbox.exec_())
