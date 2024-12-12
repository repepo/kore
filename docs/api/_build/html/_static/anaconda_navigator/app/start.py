#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Application start."""

import os
import signal
import sys

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from anaconda_navigator.config import CONF, LINUX, LOCKFILE, MAC, PIDFILE, UBUNTU
from anaconda_navigator.external import filelock
from anaconda_navigator.static import images
from anaconda_navigator.static.fonts import load_fonts
from anaconda_navigator.utils import misc
from anaconda_navigator.utils.logs import setup_logger, LOGGER_CONFIG
from anaconda_navigator.utils.qthelpers import qapplication
from anaconda_navigator.widgets.dialogs import MessageBoxInformation
from anaconda_navigator.widgets.dialogs.splash import SplashScreen
from anaconda_navigator.widgets.main_window import MainWindow
from anaconda_navigator.utils import styles


app: QtWidgets.QApplication

# For retina displays on qt5
if CONF.get('main', 'enable_high_dpi_scaling'):
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)


def except_hook(cls, exception, traceback):
    """Custom except hook to avoid crashes on PyQt5."""
    sys.__excepthook__(cls, exception, traceback)


def set_application_icon():
    """Set application icon."""
    global app  # pylint: disable=global-variable-not-assigned,invalid-name
    if LINUX and UBUNTU:
        app_icon = QtGui.QIcon(images.ANACONDA_LOGO_WHITE)
    else:
        app_icon = QtGui.QIcon(images.ANACONDA_LOGO)
    app.setWindowIcon(app_icon)


def run_app(splash):
    """Create and show Navigator's main window."""
    window = MainWindow(splash=splash)
    # window.setup()
    return window


class EventEater(QtCore.QObject):  # pylint: disable=too-few-public-methods
    """Event filter for application state."""

    def __init__(self, application):
        """Event filter for application state."""
        super().__init__()
        self.app = application

    def eventFilter(self, ob, event):  # pylint: disable=invalid-name
        """Qt override."""
        if (event.type() == QtCore.QEvent.ApplicationActivate) and MAC and self.app.window.setup_ready:
            self.app.window.show()
            if self.app.window.isMaximized():
                self.app.window.showMaximized()
            elif self.app.window.isFullScreen():
                self.app.window.showFullScreen()
            else:
                self.app.window.showNormal()
            return True

        return super().eventFilter(ob, event)


def start_core_app():  # pylint: disable=missing-function-docstring
    global app  # pylint: disable=global-statement,invalid-name
    app = qapplication()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.setStyleSheet(styles.load_style_sheet())


def start_app(options):  # cov-skip
    """Main application entry point."""
    LOGGER_CONFIG.level = options.log_level
    setup_logger()

    start_core_app()
    set_application_icon()
    load_fonts(app)

    # Check if running as root or with sudo on Unix
    if (MAC or LINUX) and os.environ.get('SUDO_UID', None) is not None:
        msgbox = MessageBoxInformation(
            title='Anaconda Navigator Information',
            text='Anaconda Navigator cannot be run with root user privileges.'
        )
        sys.exit(msgbox.exec_())

    # Create file lock
    lock = filelock.FileLock(LOCKFILE)
    try:
        load_pid = misc.load_pid()

        # This means a PSutil Access Denied error was raised
        if load_pid is False:
            msgbox = MessageBoxInformation(
                title='Anaconda Navigator Startup Error',
                text=(
                    'Navigator failed to start due to an incorrect shutdown. '
                    '<br><br>'
                    'We were unable to remove the pid & lock files. '
                    'Please manually remove the following files and restart '
                    'Anaconda Navigator:<br><ul>'
                    f'<li><pre>{LOCKFILE}</pre></li><li><pre>{PIDFILE}</pre></li></ul>'
                )
            )
            sys.exit(msgbox.exec_())
        elif load_pid is None:  # A stale lock might be around
            misc.remove_lock()

        with lock.acquire(timeout=3.0):  # timeout in seconds
            misc.save_pid()
            splash = SplashScreen()
            splash.show_message('Initializing...')
            window = run_app(splash)
            app.window = window
            event_eater = EventEater(app)
            app.installEventFilter(event_eater)

            if os.environ.get('TEST_CI') is not None:
                timer_shutdown = QtCore.QTimer(app)
                timer_shutdown.timeout.connect(window.close)
                timer_shutdown.start(60_000)

            sys.exit(app.exec_())
    except filelock.Timeout:
        msgbox = MessageBoxInformation(
            title='Anaconda Navigator Information',
            text='There is an instance of Anaconda Navigator already running.'
        )
        sys.exit(msgbox.exec_())
