# -*- coding: utf-8 -*-

# pylint: disable=no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Quit application dialog."""

from qtpy.QtCore import QSize, Qt
from qtpy.QtSvg import QSvgWidget
from qtpy.QtWidgets import QDialogButtonBox, QHBoxLayout, QLabel, QListWidgetItem, QVBoxLayout
from anaconda_navigator.config import CONF
from anaconda_navigator.static import images
from anaconda_navigator.utils.styles import SASS_VARIABLES
from anaconda_navigator.widgets import (
    ButtonDanger, ButtonNormal, ButtonPrimary, CheckBoxBase, FrameBase, LabelBase, SpacerHorizontal, SpacerVertical,
)
from anaconda_navigator.widgets.dialogs import DialogBase
from anaconda_navigator.widgets.lists import ListWidgetBase


class QuitApplicationDialog(DialogBase):
    """Quit application confirmation dialog."""
    def __init__(self, *args, **kwargs):
        """Quit application confirmation dialog."""
        self.config = kwargs.pop('config', CONF)
        super().__init__(*args, **kwargs)
        self.widget_icon = QSvgWidget(images.ANACONDA_LOGO)
        self.label_about = QLabel('Quit Anaconda Navigator?')
        self.button_ok = ButtonPrimary('Yes')
        self.button_cancel = ButtonNormal('No')
        self.buttonbox = QDialogButtonBox(Qt.Horizontal)
        self.checkbox = CheckBoxBase("Don't show again")

        # Widgets setup
        self.setWindowTitle('Quit application')
        self.widget_icon.setFixedSize(QSize(100, 100))

        # Layouts
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.widget_icon, 0, Qt.AlignTop)
        h_layout.addWidget(SpacerHorizontal())
        h_layout.addWidget(self.label_about)

        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        main_layout = QVBoxLayout()
        main_layout.addLayout(h_layout)
        main_layout.addWidget(self.checkbox, 0, Qt.AlignRight)
        main_layout.addWidget(SpacerVertical())
        main_layout.addWidget(SpacerVertical())
        main_layout.addLayout(layout_buttons)
        self.setLayout(main_layout)

        # Signals
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        # Setup
        self.update_style_sheet()
        self.setup()
        self.button_cancel.setFocus()

    def setup(self):
        """Setup widget content."""
        # Widget setup
        if self.config.get('main', 'hide_quit_dialog'):
            hide_dialog = Qt.Checked
        else:
            hide_dialog = Qt.Unchecked
        self.checkbox.setChecked(hide_dialog)

    def accept(self):
        """
        Qt overide.

        Update the configuration preferences.
        """
        hide_dialog = self.checkbox.checkState() == Qt.Checked
        self.config.set('main', 'hide_quit_dialog', hide_dialog)
        super().accept()


class QuitBusyDialog(QuitApplicationDialog):
    """Dialog for quiting while navigator is busy."""
    def __init__(self, *args, **kwargs):
        """Quit application confirmation dialog."""
        super().__init__(*args, **kwargs)
        self.label_about.setText('Anaconda Navigator is still busy.<br><br>Do you want to quit?')
        self.setWindowTitle('Quit application')
        self.checkbox.setVisible(False)


class FrameRunningApps(FrameBase):
    """Main widget for the list items."""


class ListRunningApps(ListWidgetBase):
    """List of running apps."""

    def setup_item(self, item):
        """Override."""


class ListItemRunningApp(QListWidgetItem):
    """List item representing a running application."""
    def __init__(self, package):
        """List item representing a running application."""
        super().__init__()
        self.widget = FrameRunningApps()
        self.package = package
        self.label = LabelBase(package)
        self.checkbox = CheckBoxBase()

        layout_frame = QHBoxLayout()
        layout_frame.addWidget(self.checkbox)
        layout_frame.addWidget(self.label)
        layout_frame.addStretch()
        self.widget.setLayout(layout_frame)
        self.setSizeHint(self.widget_size())

    def set_checked(self, value):
        """Set the check state for the checkbox in the list item."""
        self.checkbox.setChecked(value)

    def checked(self):
        """Return True if checked otherwise return False."""
        return self.checkbox.isChecked()

    @staticmethod
    def set_loading(item):
        """Override."""

    def text(self):
        """Qt override."""
        return self.label.text()

    @staticmethod
    def set_selected(item):
        """Override."""

    @staticmethod
    def widget_size():
        """Return the size defined in the SASS file."""
        return QSize(SASS_VARIABLES.WIDGET_RUNNING_APPS_TOTAL_WIDTH, SASS_VARIABLES.WIDGET_RUNNING_APPS_TOTAL_HEIGHT)


class QuitRunningAppsDialog(DialogBase):  # pylint: disable=too-many-instance-attributes
    """Dialog for closing running apps if quiting navigator."""
    def __init__(self, *args, **kwargs):
        """Dialog for closing running apps if quiting navigator."""
        self.running_processes = kwargs.pop('running_processes', [])
        self.config = kwargs.pop('config', CONF)
        super().__init__(*args, **kwargs)

        self.list = ListRunningApps(self)
        self.label_about = QLabel(
            'There are some applications running. Please select \nthe applications you want to close on quit:'
        )
        self.button_close = ButtonDanger('Quit')
        self.button_cancel = ButtonNormal('Cancel')
        self.buttonbox = QDialogButtonBox(Qt.Horizontal)
        self.checkbox = CheckBoxBase('Don\'t show again')

        # Widget setup
        self.setWindowTitle('Close running applications')

        # Layouts
        layout_buttons = QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(SpacerHorizontal())
        layout_buttons.addWidget(self.button_close)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_about)
        main_layout.addWidget(SpacerVertical())
        main_layout.addWidget(self.list, 0, Qt.AlignCenter)
        main_layout.addWidget(SpacerVertical())
        main_layout.addWidget(self.checkbox, 0, Qt.AlignRight)
        main_layout.addWidget(SpacerVertical())
        main_layout.addWidget(SpacerVertical())
        main_layout.addLayout(layout_buttons)
        self.setLayout(main_layout)

        # Signals
        self.button_close.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        # Setup
        self.update_style_sheet()
        self.setup()
        self.button_cancel.setFocus()

    def setup(self):
        """Setup widget content."""
        # Widget setup
        packages = sorted({
            package.package
            for package in self.running_processes
        })
        checked_packages = self.config.get('main', 'running_apps_to_close')
        for package in packages:
            item = ListItemRunningApp(package)
            item.set_checked(package in checked_packages)
            self.list.addItem(item)

        if self.config.get('main', 'hide_running_apps_dialog'):
            hide_dialog = Qt.Checked
        else:
            hide_dialog = Qt.Unchecked
        self.checkbox.setChecked(hide_dialog)

    def accept(self):
        """
        Qt overide.

        Update the configuration preferences.
        """
        original_checked_packages = set(self.config.get('main', 'running_apps_to_close'))
        hide_dialog = self.checkbox.checkState() == Qt.Checked
        checked_packages = {
            item.text()
            for item in self.list.items()
            if item.checked()
        }
        packages = {
            item.text()
            for item in self.list.items()
        }
        delta = original_checked_packages - packages
        new_packages = sorted(list(checked_packages.union(delta)))
        self.config.set('main', 'running_apps_to_close', new_packages)
        self.config.set('main', 'hide_running_apps_dialog', hide_dialog)
        super().accept()


class ClosePackageManagerDialog(QuitApplicationDialog):
    """Confirm application quit if package manager is running."""
    def __init__(self, *args, **kwargs):
        """Confirm application quit if package manager is running."""
        super().__init__(*args, **kwargs)

        self.label_about.setText('Conda is still busy.\n\nDo you want to cancel the process?')
        self.setWindowTitle('Cancel Process')
        self.checkbox.setVisible(False)
