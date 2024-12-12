# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for cloning environments."""

__all__ = ['CloneDialog']

from qtpy import QtWidgets  # QGridLayout, QHBoxLayout, QVBoxLayout
from anaconda_navigator import widgets
from . import common


class CloneDialog(common.EnvironmentActionsDialog):  # pylint: disable=too-few-public-methods
    """Clone environment dialog."""

    def __init__(self, parent=None, clone_from_name=None):
        """Clone environment dialog."""
        super().__init__(parent=parent)

        # Widgets
        self.label_name = widgets.LabelBase('Name:')
        self.text_name = common.LineEditEnvironment()

        self.label_location = widgets.LabelBase('Location:')
        self.label_prefix = widgets.LabelBase()
        self.button_ok = widgets.ButtonPrimary('Clone')
        self.button_cancel = widgets.ButtonNormal('Cancel')

        # Widget setup
        self.align_labels([self.label_name, self.label_location])
        self.setMinimumWidth(self.BASE_DIALOG_WIDTH)
        self.setWindowTitle('Clone from environment: ' + clone_from_name)
        self.text_name.setPlaceholderText('New environment name')
        self.label_prefix.setObjectName('environment-location')

        # Layouts
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.label_name, 2, 0)
        grid.addWidget(widgets.SpacerHorizontal(), 2, 1)
        grid.addWidget(self.text_name, 2, 2)
        grid.addWidget(widgets.SpacerVertical(), 3, 0)
        grid.addWidget(self.label_location, 4, 0)
        grid.addWidget(self.label_prefix, 4, 2)

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(widgets.SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(widgets.SpacerVertical())
        layout.addWidget(widgets.SpacerVertical())
        layout.addLayout(layout_buttons)

        self.setLayout(layout)

        # Signals
        self.text_name.textChanged.connect(self.refresh)
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

        # Setup
        self.text_name.setFocus()
        self.refresh()

    def refresh(self, text=''):  # pylint: disable=unused-argument
        """Update status of buttons based on combobox selection."""
        name = self.name
        self.update_location()
        if self.environments:
            self.button_ok.setDisabled(not self.is_valid_env_name(name))
