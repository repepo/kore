# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Common components for environment dialogs."""

__all__ = ['LineEditEnvironment', 'EnvironmentActionsDialog']

import os
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from anaconda_navigator import widgets
from anaconda_navigator.widgets import common as global_commons
from anaconda_navigator.widgets import dialogs


class LineEditEnvironment(widgets.LineEditBase):
    """
    Custom line edit to handle regex for naming an environment.
    """

    VALID_RE = QtCore.QRegExp('^[A-Za-z][A-Za-z0-9_-]{0,30}$')  # pylint: disable=invalid-name

    sig_return_pressed = QtCore.Signal()
    sig_escape_pressed = QtCore.Signal()
    sig_copied = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        """Custom line edit for naming an environment."""
        super().__init__(*args, **kwargs)
        self._validator = QtGui.QRegExpValidator(self.VALID_RE)
        self.menu = QtWidgets.QMenu(parent=self)
        self.setValidator(self._validator)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def event(self, event):
        """Override Qt method."""
        if (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.buttons() & QtCore.Qt.RightButton
                and not self.isEnabled()
        ):
            self.show_menu(event.pos())
            return True
        return super().event(event)

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        """Override Qt method."""
        key = event.key()
        # Display a copy menu in case the widget is disabled.
        if event.matches(QtGui.QKeySequence.Paste):
            clipboard = QtWidgets.QApplication.clipboard()
            text = clipboard.text()
            if self.VALID_RE.exactMatch(text):
                self.setText(text)
                return
        else:
            if key in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter]:
                self.sig_return_pressed.emit()
            elif key in [QtCore.Qt.Key_Escape]:
                self.sig_escape_pressed.emit()
        super().keyPressEvent(event)

    def show_menu(self, pos):
        """Show copy menu for channel item."""
        self.menu.clear()
        copy = QtWidgets.QAction('&Copy', self.menu)
        copy.triggered.connect(self.copy_text)
        self.menu.addAction(copy)
        self.menu.setEnabled(True)
        self.menu.exec_(self.mapToGlobal(pos))

    def copy_text(self):
        """Copy environment text to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self.text())
        self.sig_copied.emit()


class EnvironmentActionsDialog(dialogs.DialogBase):  # pylint: disable=too-many-instance-attributes
    """Base dialog with common methods for all dialogs."""

    BASE_DIALOG_WIDTH = 480  # pylint: disable=invalid-name

    sig_setup_ready = QtCore.Signal()

    def __init__(self, parent=None, api=None):
        """Base dialog with common methods for all dialogs."""
        super().__init__(parent=parent)

        self.info = None
        self._packages = None
        self.envs_dirs = None
        self.environments = None
        self.api = api
        self.channels = None

        # Widgets to be defined on subclass __init__
        self.text_name = None
        self.label_prefix = None

    def setup(self, worker=None, conda_data=None, error=None):  # pylint: disable=unused-argument
        """Setup the dialog conda information as a result of a conda worker."""
        if conda_data:
            conda_info = conda_data.get('processed_info', {})
            self.info = conda_info
            self._packages = conda_data.get('packages')
            self.envs_dirs = conda_info['__envs_dirs_writable']
            self.environments = conda_info['__environments']
            self.channels = conda_info.get('channels')
            self.refresh()
            self.sig_setup_ready.emit()

    def update_location(self):
        """Update the location (prefix) text."""
        self.button_ok.setDisabled(True)
        if self.environments:
            fm = self.label_prefix.fontMetrics()  # pylint: disable=invalid-name
            prefix = fm.elidedText(self.prefix, QtCore.Qt.ElideLeft, 300)
            self.label_prefix.setText(prefix)
            self.label_prefix.setToolTip(self.prefix)

    def refresh(self):
        """Update the status of buttons based data entered."""
        raise NotImplementedError

    def is_valid_env_name(self, env_name):
        """
        Check that an environment has a valid name.
        On Windows is case insensitive.
        """
        env_names = self.environments.values() if self.environments else []
        if os.name == 'nt':  # cov-win
            env_name = env_name.lower()
            env_names = list(map(str.lower, env_names))

        return bool(env_name) and (env_name not in env_names)

    @staticmethod
    def align_labels(label_widgets):
        """Align label widgets to the right."""
        for widget in label_widgets:
            widget.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    @property
    def name(self):  # noqa: D401
        """Return the content without extra spaces for the name of the env."""
        text = ''
        if self.text_name:
            text = self.text_name.text().strip()
        return text

    @property
    def prefix(self):  # noqa: D401
        """Return the full prefix (location) as entered in the dialog."""
        result: str
        if self.envs_dirs and self.name:
            result = os.path.join(self.envs_dirs[0], self.name)
        else:
            result = self.name
        return result


class OpenIconButton(global_commons.IconButton):  # pylint: disable=too-few-public-methods
    """Button with "open" icon."""
