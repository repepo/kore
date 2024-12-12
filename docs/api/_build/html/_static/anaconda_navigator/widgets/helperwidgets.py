# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Helper widgets."""

from __future__ import absolute_import, division, print_function, with_statement

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QSizePolicy, QToolButton
from anaconda_navigator.utils.qthelpers import update_pointer


class ButtonSearch(QPushButton):
    """Button used for CSS styling."""
    sig_entered = Signal()
    sig_left = Signal()

    def mousePressEvent(self, event):
        """Override Qt method."""
        super().mousePressEvent(event)
        update_pointer()
        self.setProperty('focused', False)

    def enterEvent(self, event):
        """Override Qt method."""
        if self.isEnabled():
            update_pointer(Qt.ArrowCursor)
            self.setProperty('focused', True)
            self.sig_entered.emit()
        else:
            self.setProperty('focused', False)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Override Qt method."""
        if self.isEnabled():
            update_pointer()
            self.setProperty('focused', False)
            self.sig_left.emit()
        else:
            self.setProperty('focused', False)
        super().leaveEvent(event)

    def setProperty(self, name, value):
        """Override Qt method."""
        super().setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class ButtonShow(ButtonSearch):  # pylint: disable=missing-class-docstring

    def mousePressEvent(self, event):
        """Override Qt method."""

    def mouseReleaseEvent(self, event):
        """Override Qt method."""

    def mouseDoubleClickEvent(self, event):
        """Override Qt method."""


class LineEditSearch(QLineEdit):
    """Lineedit search widget with clear button."""
    def __init__(self, *args, **kwargs):
        """Lineedit search widget with clear button."""
        super().__init__(*args, **kwargs)
        self._empty = True
        self._show_icons = False
        self.button_icon = ButtonSearch()

        # Setup
        self.button_icon.setDefault(True)
        self.button_icon.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_MacShowFocusRect, False)

        # Layouts
        layout = QHBoxLayout()
        layout.addWidget(self.button_icon, 0, Qt.AlignRight)
        layout.setSpacing(0)
        layout.addSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Signals
        self.textEdited.connect(self.update_box)
        self.button_icon.clicked.connect(self.clear_text)

        self.update_box(None)
        self.set_icon_size(16, 16)
        self.setTabOrder(self, self.button_icon)

    def set_icon_size(self, width, height):
        """Set clear button icon size."""
        self.button_icon.setMaximumSize(QSize(width, height))
        self.setStyleSheet(f'LineEditSearch {{padding-right: {width}px;}}')

    def set_icon_visibility(self, value):
        """Set clear button visibility."""
        self._show_icons = value
        self.update_box()

    def update_box(self, text=None):
        """Update icon visibility and status."""
        if text and self._show_icons:
            self.button_icon.setIcon(QIcon())
        else:
            if self._show_icons:
                self.button_icon.setIcon(QIcon())
        self._empty = not bool(text)
        self.button_icon.setDisabled(self._empty)

    def clear_text(self):
        """Clear all text in the line edit."""
        self.setText('')
        self.setFocus()
        self.update_box()

    def update_style_sheet(self):
        """Update custom CSS style sheet."""

    def keyPressEvent(self, event):
        """Override Qt method."""
        key = event.key()
        if key in [Qt.Key_Escape]:
            self.clear_text()
        else:
            super().keyPressEvent(event)


class ButtonToggleCollapse(QToolButton):  # pylint: disable=too-few-public-methods
    """Button to collapse the environment list."""
    def __init__(self, *args, **kwargs):
        """Button to collapse the environment list."""
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCheckable(True)

        self.clicked.connect(self._update_icon)

    def setProperty(self, name, value):
        """Override Qt method."""
        QToolButton.setProperty(self, name, value)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _update_icon(self):
        self.setProperty('checked', self.isChecked())
