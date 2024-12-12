# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Splash screen and intial startup splash."""

from __future__ import absolute_import, division, print_function

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPixmap
from qtpy.QtWidgets import QApplication, QGraphicsOpacityEffect, QSplashScreen  # pylint: disable=no-name-in-module
from anaconda_navigator.static.images import ANACONDA_ICON_256_PATH


class SplashScreen(QSplashScreen):
    """Splash screen for the main window."""
    def __init__(self, *args, **kwargs):
        """Splash screen for the main window."""
        super().__init__(*args, **kwargs)
        self._effect = QGraphicsOpacityEffect()
        self._font = self.font()
        self._pixmap = QPixmap(ANACONDA_ICON_256_PATH)
        self._message = ''

        # Setup
        self._font.setPixelSize(10)
        self._effect.setOpacity(0.9)
        self.setFont(self._font)
        self.setGraphicsEffect(self._effect)
        self.setPixmap(self._pixmap)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.SplashScreen | Qt.WindowStaysOnTopHint)

    def get_message(self):
        """Return currently displayed message."""
        return self._message

    def show_message(self, message):
        """Show message in the screen."""
        self._message = message
        message += '\n'
        self.show()
        self.showMessage(message, Qt.AlignBottom + Qt.AlignCenter + Qt.AlignAbsolute, QColor(Qt.white))
        QApplication.processEvents()
